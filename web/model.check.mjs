// Parity check: JS model vs. reference values from the Python library (src/scenarios.py).
// Run from the repo root: node web/model.check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { runScenario } from './model.js';

const data = JSON.parse(fs.readFileSync(new URL('./data.json', import.meta.url)));
const close = (a, b, rel = 1e-4) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b) + 1e-6, `${a} != ${b}`);

// [scenario, median (first age S<0.5, null = none), S(90), E[#conditions](70)]
const cases = [
  [{ sex: 'All' }, 82, 0.199952, 1.497195],
  [{ sex: 'Male', agingRate: 0.5, slowStart: 40, removed: ['Cancer'] }, 111, 0.717751, 0.627828],
  [{ sex: 'Female', agingRate: 0, slowStart: 60, removed: ['Cardiovascular', 'External'] }, null, 0.812186, 0.701144],
  [{ sex: 'All', agingRate: 1.5, slowStart: 30, removed: ['Neurological'] }, 69, 2.6e-05, 2.185225],
];

for (const [opts, median, s90, c70] of cases) {
  const r = runScenario(data, opts);
  const i = r.interventionSurvival.findIndex((s) => s < 0.5);
  assert.equal(i < 0 ? null : i, median);
  close(r.interventionSurvival[90], s90);
  close(r.interventionConditions[70], c70);
}
console.log(`ok: ${cases.length} scenarios match Python`);
