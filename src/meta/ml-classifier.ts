/**
 * ML Meta-Confidence Classifier — Phase 4
 *
 * A lightweight binary classifier that predicts whether a routing
 * decision needs escalation, based on signal + valence features.
 *
 * DESIGN PHILOSOPHY:
 * - GPU kickstart: model is TRAINED using Python + sklearn on GPU
 * - RAM-resident: exported weights live in RAM as a JSON file
 * - CPU inference: prediction is one dot product + sigmoid (~microseconds)
 * - Zero ML dependencies: no onnxruntime, no tensorflow, no native bindings
 *
 * The classifier takes the same inputs the heuristic meta-confidence model
 * uses (signals + valence) and outputs a probability that escalation is
 * needed. This replaces the hand-tuned thresholds and similarity matching
 * in the original MetaConfidenceModel.
 *
 * Model file format (JSON):
 * {
 *   "type": "logistic_regression",
 *   "version": 1,
 *   "features": ["tokenProbabilitySpread", "semanticVelocity", ...],
 *   "weights": [0.45, -0.12, ...],
 *   "intercept": -0.83,
 *   "threshold": 0.5,
 *   "trainedAt": "2026-04-16T...",
 *   "trainingExamples": 150,
 *   "accuracy": 0.87
 * }
 *
 * For gradient-boosted trees (future upgrade), the format extends to:
 * {
 *   "type": "gradient_boosted_tree",
 *   "version": 1,
 *   "trees": [{ "feature": 0, "threshold": 0.5, "left": ..., "right": ... }],
 *   ...
 * }
 */

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import pino from 'pino';
import type { EscalationSignals, ValenceScore } from '../tiers/tier-interface.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MODEL_DIR = resolve(__dirname, '../../models');
const MODEL_FILE = resolve(MODEL_DIR, 'meta-classifier.json');

const log = pino({ level: process.env.KINDLING_LOG_LEVEL ?? 'info' });

// ─── Model schemas ──────────────────────────────────────────────

interface LogisticRegressionModel {
  type: 'logistic_regression';
  version: number;
  features: string[];
  weights: number[];
  intercept: number;
  threshold: number;
  trainedAt: string;
  trainingExamples: number;
  accuracy: number;
}

interface TreeNode {
  feature: number;       // index into feature vector
  threshold: number;     // split threshold
  left: TreeNode | number;  // left child or leaf value
  right: TreeNode | number; // right child or leaf value
}

interface GradientBoostedTreeModel {
  type: 'gradient_boosted_tree';
  version: number;
  features: string[];
  trees: TreeNode[];
  learningRate: number;
  baseScore: number;
  threshold: number;
  trainedAt: string;
  trainingExamples: number;
  accuracy: number;
}

type ModelDefinition = LogisticRegressionModel | GradientBoostedTreeModel;

// ─── Feature extraction ─────────────────────────────────────────

/** Standard feature vector for the classifier */
const FEATURE_NAMES = [
  'tokenProbabilitySpread',
  'semanticVelocity',
  'surpriseScore',
  'attentionAnomalyScore',
  'valenceUrgency',
  'valenceComplexity',
  'valenceStakes',
  'valenceComposite',
  // Interaction features (signal × valence)
  'spreadXcomplexity',
  'surpriseXstakes',
] as const;

export function extractFeatures(
  signals: EscalationSignals,
  valence: ValenceScore
): number[] {
  return [
    signals.tokenProbabilitySpread,
    signals.semanticVelocity,
    signals.surpriseScore,
    signals.attentionAnomalyScore,
    valence.urgency,
    valence.complexity,
    valence.stakes,
    valence.composite,
    // Interaction features help capture non-linear relationships
    signals.tokenProbabilitySpread * valence.complexity,
    signals.surpriseScore * valence.stakes,
  ];
}

// ─── Math primitives ────────────────────────────────────────────

function sigmoid(x: number): number {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function traverseTree(node: TreeNode | number, features: number[]): number {
  if (typeof node === 'number') return node;
  const featureValue = features[node.feature];
  return featureValue <= node.threshold
    ? traverseTree(node.left, features)
    : traverseTree(node.right, features);
}

// ─── Classifier ─────────────────────────────────────────────────

export interface ClassifierPrediction {
  escalationNeeded: boolean;
  probability: number;
  modelType: string;
  confidence: number; // how far from the threshold (0 = borderline, 1 = certain)
}

export class MLClassifier {
  private model: ModelDefinition | null = null;
  private loaded = false;

  constructor() {
    this.tryLoad();
  }

  /**
   * Predict whether escalation is needed for the given signals + valence.
   * Returns null if no trained model is available (falls back to heuristics).
   */
  predict(
    signals: EscalationSignals,
    valence: ValenceScore
  ): ClassifierPrediction | null {
    if (!this.model) return null;

    const features = extractFeatures(signals, valence);
    let probability: number;

    if (this.model.type === 'logistic_regression') {
      const logit = dotProduct(features, this.model.weights) + this.model.intercept;
      probability = sigmoid(logit);
    } else if (this.model.type === 'gradient_boosted_tree') {
      let rawScore = this.model.baseScore;
      for (const tree of this.model.trees) {
        rawScore += this.model.learningRate * traverseTree(tree, features);
      }
      probability = sigmoid(rawScore);
    } else {
      return null;
    }

    const threshold = this.model.threshold;
    const confidence = Math.abs(probability - threshold) / Math.max(threshold, 1 - threshold);

    return {
      escalationNeeded: probability >= threshold,
      probability,
      modelType: this.model.type,
      confidence: Math.min(1, confidence),
    };
  }

  /** Check if a trained model is loaded */
  isLoaded(): boolean {
    return this.loaded;
  }

  /** Get model metadata */
  getModelInfo(): {
    type: string;
    trainedAt: string;
    trainingExamples: number;
    accuracy: number;
    features: string[];
  } | null {
    if (!this.model) return null;
    return {
      type: this.model.type,
      trainedAt: this.model.trainedAt,
      trainingExamples: this.model.trainingExamples,
      accuracy: this.model.accuracy,
      features: this.model.features,
    };
  }

  /** Reload model from disk (e.g., after a training pass) */
  reload(): boolean {
    this.model = null;
    this.loaded = false;
    return this.tryLoad();
  }

  /** Feature names for training data export */
  static getFeatureNames(): readonly string[] {
    return FEATURE_NAMES;
  }

  private tryLoad(): boolean {
    if (!existsSync(MODEL_FILE)) {
      log.debug('No ML classifier model found — using heuristic meta-confidence');
      return false;
    }

    try {
      const raw = readFileSync(MODEL_FILE, 'utf-8');
      const parsed = JSON.parse(raw) as ModelDefinition;

      // Validate basic structure
      if (!parsed.type || !parsed.features || !Array.isArray(parsed.features)) {
        log.warn('ML classifier model file is malformed');
        return false;
      }

      if (parsed.type === 'logistic_regression') {
        if (!Array.isArray(parsed.weights) || typeof parsed.intercept !== 'number') {
          log.warn('Logistic regression model missing weights or intercept');
          return false;
        }
        if (parsed.weights.length !== parsed.features.length) {
          log.warn(
            { expected: parsed.features.length, got: parsed.weights.length },
            'Weight count does not match feature count'
          );
          return false;
        }
      }

      this.model = parsed;
      this.loaded = true;
      log.info(
        {
          type: parsed.type,
          features: parsed.features.length,
          examples: parsed.trainingExamples,
          accuracy: parsed.accuracy,
        },
        'ML classifier loaded — replacing heuristic meta-confidence'
      );
      return true;
    } catch (err) {
      log.warn({ err }, 'Failed to load ML classifier model');
      return false;
    }
  }
}

export function getModelFilePath(): string {
  return MODEL_FILE;
}

export function getModelDir(): string {
  return MODEL_DIR;
}
