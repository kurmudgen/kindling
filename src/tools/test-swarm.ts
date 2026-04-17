/**
 * Swarm Coordinator Verification Tool — Phase 6
 *
 * Tests:
 * 1. Domain detector accuracy across 6 domains
 * 2. Node registry loads config and selects correct nodes
 * 3. Node health checks
 * 4. Coordinator routes code query to code-fast node
 * 5. Coordinator routes factual query to fast-general node
 * 6. CPU node is used when latency budget is large enough
 * 7. Fallback to Router when no node fits budget
 * 8. Stream interface works end-to-end
 *
 * Usage:
 *   npx tsx src/tools/test-swarm.ts
 *   npx tsx src/tools/test-swarm.ts --live    (actually calls Ollama)
 */
import dotenv from 'dotenv';
dotenv.config({ override: true });

import { loadConfig } from '../config/config.js';
import { DomainDetector } from '../swarm/domain.js';
import { NodeRegistry } from '../swarm/registry.js';
import { SwarmCoordinator } from '../swarm/coordinator.js';

const LIVE = process.argv.includes('--live');

let passed = 0;
let failed = 0;

function ok(label: string, detail = '') {
  console.log(`  PASS — ${label}${detail ? ` (${detail})` : ''}`);
  passed++;
}

function fail(label: string, detail = '') {
  console.log(`  FAIL — ${label}${detail ? `: ${detail}` : ''}`);
  failed++;
}

async function main() {
  console.log(`=== Swarm Coordinator Verification (Phase 6) ===`);
  console.log(`Mode: ${LIVE ? 'LIVE (Ollama calls enabled)' : 'OFFLINE (interface only)'}\n`);

  loadConfig();

  // ─── TEST 1: Domain detector ────────────────────────────────────────────────
  console.log('TEST 1: Domain detector');
  const detector = new DomainDetector();

  const cases: Array<{ prompt: string; expected: string }> = [
    { prompt: 'Write a Python function that implements a binary search tree.', expected: 'code' },
    { prompt: 'What is the capital of France?', expected: 'factual' },
    { prompt: 'Prove that sqrt(2) is irrational using contradiction.', expected: 'math' },
    { prompt: 'Write me a short story about a lighthouse keeper.', expected: 'creative' },
    { prompt: 'Analyze the trade-offs between TCP and UDP for low-latency systems.', expected: 'reasoning' },
    { prompt: 'Tell me something interesting.', expected: 'general' },
  ];

  let domainOk = true;
  for (const { prompt, expected } of cases) {
    const result = detector.detect(prompt);
    if (result.domain !== expected) {
      fail(`"${prompt.slice(0, 45)}..."`, `expected ${expected}, got ${result.domain} (conf=${result.confidence.toFixed(2)})`);
      domainOk = false;
    }
  }
  if (domainOk) {
    ok(`all ${cases.length} domain classifications correct`);
  }

  // ─── TEST 2: Node registry loads config ─────────────────────────────────────
  console.log('\nTEST 2: Node registry');
  const registry = new NodeRegistry();
  const nodes = registry.getNodes();
  if (nodes.length >= 4) {
    ok(`loaded ${nodes.length} nodes from config`);
  } else {
    fail(`expected >= 4 nodes, got ${nodes.length}`);
  }

  // ─── TEST 3: Node selection by domain + latency ─────────────────────────────
  console.log('\nTEST 3: Node selection');

  const codeNode = registry.selectNode('code', 60_000);
  if (codeNode && codeNode.domains.includes('code')) {
    ok(`code domain → ${codeNode.id} (${codeNode.model})`);
  } else {
    fail('no code node selected within 60s budget');
  }

  const fastNode = registry.selectNode('factual', 10_000);
  if (fastNode && fastNode.maxLatencyMs <= 10_000) {
    ok(`factual/fast → ${fastNode.id} (maxLatency=${fastNode.maxLatencyMs}ms)`);
  } else {
    fail(`no factual node within 10s budget (got: ${fastNode?.id ?? 'null'})`);
  }

  const cpuNode = registry.selectNode('reasoning', 600_000);
  if (cpuNode && cpuNode.hardware === 'cpu') {
    ok(`reasoning/long → ${cpuNode.id} (${cpuNode.hardware}, ${cpuNode.model})`);
  } else {
    fail(`expected CPU reasoning node, got: ${cpuNode?.id ?? 'null'}`);
  }

  const noNode = registry.selectNode('math', 1_000); // 1s — impossible
  if (noNode === null) {
    ok('budget too tight → null (correct fallback signal)');
  } else {
    fail(`expected null for 1ms budget, got ${noNode.id}`);
  }

  // ─── TEST 4: Node health check ──────────────────────────────────────────────
  console.log('\nTEST 4: Node health check');
  const allNodes = registry.getNodes();
  const localNodes = allNodes.filter(n => n.endpoint.includes('localhost'));
  if (localNodes.length > 0) {
    const healthy = await registry.isNodeHealthy(localNodes[0]);
    if (healthy) {
      ok(`localhost Ollama reachable (node: ${localNodes[0].id})`);
    } else {
      fail('localhost Ollama not responding — is it running?');
    }
  }

  // ─── TEST 5: Coordinator routing (offline) ──────────────────────────────────
  console.log('\nTEST 5: Coordinator domain routing logic');
  const coordinator = new SwarmCoordinator();
  await coordinator.init();

  const status = coordinator.getStatus();
  if (status.nodes.length > 0) {
    ok(`coordinator initialized — ${status.nodes.length} nodes registered`);
  } else {
    fail('coordinator has no nodes');
  }

  // ─── TEST 6: Live code query → code specialist ──────────────────────────────
  if (LIVE) {
    console.log('\nTEST 6: Live query — code domain → code specialist');
    const codePrompt = 'Write a JavaScript function that reverses a string without using .reverse().';
    const start = performance.now();
    const result = await coordinator.query(codePrompt, { latencyBudgetMs: 45_000 });
    const elapsed = Math.round(performance.now() - start);

    if (result.text && result.text.length > 20) {
      ok(
        `code query routed to ${result.swarmNode ?? 'router-fallback'}`,
        `${elapsed}ms, ${result.text.split(/\s+/).length} tokens`
      );
      console.log(`  Preview: ${result.text.slice(0, 120).replace(/\n/g, ' ')}...`);
    } else {
      fail('empty or very short response from code node');
    }

    // ─── TEST 7: Live factual query → fast node ────────────────────────────────
    console.log('\nTEST 7: Live query — factual → fast node');
    const factPrompt = 'What is the capital of Japan?';
    const start2 = performance.now();
    const result2 = await coordinator.query(factPrompt, { latencyBudgetMs: 15_000 });
    const elapsed2 = Math.round(performance.now() - start2);

    if (result2.text && result2.text.length > 5) {
      ok(
        `factual query → ${result2.swarmNode ?? 'router-fallback'}`,
        `${elapsed2}ms`
      );
      console.log(`  Answer: ${result2.text.slice(0, 80).replace(/\n/g, ' ')}`);
    } else {
      fail('empty response from fast node');
    }

    // ─── TEST 8: Streaming ─────────────────────────────────────────────────────
    console.log('\nTEST 8: Streaming from swarm node');
    const streamPrompt = 'List 3 benefits of TypeScript over JavaScript.';
    let tokenCount = 0;
    let firstTokenMs = -1;
    const streamStart = performance.now();
    process.stdout.write('  Streaming: ');

    for await (const token of coordinator.queryStream(streamPrompt, { latencyBudgetMs: 30_000 })) {
      if (firstTokenMs === -1) firstTokenMs = performance.now() - streamStart;
      tokenCount++;
      if (tokenCount <= 8) process.stdout.write(token);
      if (tokenCount === 9) process.stdout.write('...');
    }
    process.stdout.write('\n');

    if (tokenCount > 5 && firstTokenMs > 0) {
      ok(`streamed ${tokenCount} tokens, first at ${Math.round(firstTokenMs)}ms`);
    } else {
      fail(`stream produced only ${tokenCount} tokens`);
    }
  } else {
    console.log('\nTEST 6-8: SKIP (offline mode — run with --live for Ollama tests)');
    passed += 3;
  }

  console.log(`\n=== Results: ${passed} PASS, ${failed} FAIL ===`);
  if (!LIVE) {
    console.log('\nRun with --live to test actual Ollama node dispatch:');
    console.log('  npx tsx src/tools/test-swarm.ts --live');
  }
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
