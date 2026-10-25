import { runScenario, conditionalSurvival, medianFromSurvival, chanceAtAge, MEDIAN_CAP_AGE } from './model.js';

const COLOR_BASELINE = '#6c757d';
const COLOR_INTERVENTION = '#1f77b4';
const COLOR_GAIN = '#2ca02c';
const COLOR_LOSS = '#dc3545';
const DEFAULT_SLOW_START = 25;
// The GBD smoother puts the "95+" bucket midpoint at 97; past that the curve is flat-filled.
const GBD_DATA_MAX_AGE = 97;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const fin = Number.isFinite;

const data = await (await fetch('data.json')).json();
const selected = [];
let slowStartOverridden = false;

// --- Inputs ---

const num = (el) => (el.value === '' ? null : Number(el.value));

function readInputs() {
  const userAge = num($('user-age'));
  const slowInput = num($('slow-start'));
  const floor = userAge != null ? Math.trunc(userAge) : DEFAULT_SLOW_START;
  return {
    name: $('user-name').value,
    userAge,
    sex: $('sex').value || 'All',
    agingPct: Number($('aging-rate').value),
    slowInput,
    slowStart: slowInput == null ? floor : Math.max(Math.trunc(slowInput), floor),
    removeAccidents: $('remove-accidents').checked,
    padTo: num($('pad-to')) ?? 120,
  };
}

function scenarioFor(inp) {
  return runScenario(data, {
    sex: inp.sex, agingRate: inp.agingPct / 100, slowStart: inp.slowStart,
    removed: selected, padTo: inp.padTo, removeAccidentsForFit: inp.removeAccidents,
  });
}

// --- Formatting helpers ---

function possessive(name) {
  const nm = (name || '').trim();
  if (!nm) return '';
  return esc(nm.endsWith('s') ? `${nm}'` : `${nm}'s`);
}
const cardTitle = (rest, poss) => (poss ? `${poss} ${rest}` : rest[0].toUpperCase() + rest.slice(1));

function comparisonBody(today, scenario, delta, color, showScenario) {
  let html = `<div class="mb-1"><span class="text-muted small">Today</span><span class="float-end">${today}</span></div>`;
  if (showScenario) {
    html += `<div><span class="small" style="color:${color}">With your changes</span><span class="float-end fw-bold" style="color:${color}">${scenario}</span></div>`;
    if (delta) html += `<div class="small text-muted mt-1">${delta}</div>`;
  }
  return html;
}

function reachBody(basePct, intPct, showScenario, color) {
  if (basePct == null) return '<div class="small text-muted">Already past this age</div>';
  let delta = null;
  if (showScenario && basePct > 0.05) {
    const ratio = intPct / basePct;
    if (ratio >= 1.05) delta = `${ratio.toFixed(1)}× more likely`;
    else if (ratio <= 0.95) delta = `${ratio.toFixed(2)}× as likely`;
  }
  return comparisonBody(`${basePct.toFixed(1)}%`, intPct != null ? `${intPct.toFixed(1)}%` : '—', delta, color, showScenario);
}

function fmtGompertz(x) {
  if (!fin(x)) return '—';
  if (Math.abs(x) >= 0.001 && Math.abs(x) < 1000) return x.toFixed(4);
  const exp = Math.floor(Math.log10(Math.abs(x)));
  return `${(x / 10 ** exp).toFixed(2)} × 10<sup>${exp}</sup>`;
}

function gompertzRows(base, int, show, color) {
  let html = `<div><span class="text-muted">Baseline: </span>${fmtGompertz(base)}</div>`;
  if (show) html += `<div><span style="color:${color}">Adjusted: </span><span style="color:${color};font-weight:600">${fmtGompertz(int)}</span></div>`;
  return html;
}

function hoverSurv(ages, pcts, mort) {
  return ages.map((age, i) => {
    const mx = mort[age];
    const mortTxt = fin(mx) ? `${(mx * 1000).toFixed(1)} in 1,000 die this year` : '';
    return `<b>Age ${age}</b><br>${pcts[i].toFixed(1)}% still alive<br><span style='color:#6c757d'>${mortTxt}</span>`;
  });
}

// --- Main render (port of app.py update_dashboard) ---

const PLOT_CONFIG = { displayModeBar: false, responsive: true };
const axis = { showgrid: true, gridcolor: '#eef0f2', zeroline: false, fixedrange: true };
const baseLayout = {
  template: 'plotly_white', hovermode: 'x unified', dragmode: false,
  legend: { orientation: 'h', yanchor: 'top', y: -0.18, xanchor: 'center', x: 0.5 },
  margin: { l: 60, r: 30, t: 60, b: 90 }, plot_bgcolor: 'white', paper_bgcolor: 'white',
  font: { family: '"Open Sans", verdana, arial, sans-serif' },
};

function render() {
  const inp = readInputs();
  const { userAge } = inp;
  const agingRate = inp.agingPct / 100;
  const r = scenarioFor(inp);

  const baseSurv = conditionalSurvival(r.baselineSurvival, userAge);
  const intSurv = conditionalSurvival(r.interventionSurvival, userAge);
  const active = selected.length > 0 || agingRate !== 1;

  // --- Lifespan numbers ---
  const baseMedian = medianFromSurvival(baseSurv, r.baselineMortality);
  const intMedian = medianFromSurvival(intSurv, r.interventionMortality);
  const capped = intMedian === Infinity;
  const intMedianEff = capped ? MEDIAN_CAP_AGE : intMedian;
  const gained = fin(baseMedian) && fin(intMedianEff) ? intMedianEff - baseMedian : NaN;
  const intColor = fin(gained) && gained < -0.05 ? COLOR_LOSS : COLOR_INTERVENTION;
  const poss = possessive(inp.name);

  // --- KPI cards ---
  const todayLife = fin(baseMedian) ? `${baseMedian.toFixed(1)} yrs` : '—';
  const scenLife = capped ? `>${MEDIAN_CAP_AGE} yrs` : fin(intMedian) ? `${intMedian.toFixed(1)} yrs` : '—';
  let lifeDelta = null;
  if (fin(gained) && Math.abs(gained) > 0.05) {
    const prefix = capped && gained > 0 ? '>+' : gained > 0 ? '+' : '';
    lifeDelta = `${prefix}${gained.toFixed(1)} years vs. today`;
  }
  $('card-lifespan-title').innerHTML = cardTitle('expected lifespan', poss);
  $('card-lifespan-body').innerHTML = comparisonBody(todayLife, scenLife, lifeDelta, intColor, active);

  $('card-gain-title').innerHTML = cardTitle('years gained', poss);
  if (!active) {
    $('card-gain-body').innerHTML = '<h3 class="mb-0">—</h3><div class="small text-muted mt-1">No changes yet</div>';
  } else if (fin(gained)) {
    const prefix = capped && gained > 0 ? '>+' : gained >= 0 ? '+' : '';
    const color = gained > 0 ? COLOR_GAIN : gained < 0 ? COLOR_LOSS : COLOR_BASELINE;
    let sub;
    if (fin(baseMedian) && baseMedian > 0 && gained > 0) sub = `${capped ? '>' : ''}${((gained / baseMedian) * 100).toFixed(0)}% longer than today`;
    else sub = gained < 0 ? 'Shorter than today' : 'Same as today';
    $('card-gain-body').innerHTML = `<h3 class="mb-0 fw-bold" style="color:${color}">${prefix}${gained.toFixed(1)} yrs</h3><div class="small text-muted mt-1">${sub}</div>`;
  } else {
    $('card-gain-body').innerHTML = '<h3 class="mb-0">—</h3>';
  }

  for (const target of [80, 100]) {
    const [b, i] = chanceAtAge(target, r.baselineSurvival, r.interventionSurvival, userAge);
    $(`card-${target}-title`).innerHTML = cardTitle(`chance of reaching ${target}`, poss);
    $(`card-${target}-body`).innerHTML = reachBody(b, i, active, intColor);
  }

  shareText = active && fin(gained) && gained > 0.05
    ? `My median lifespan could go up by ${capped ? 'over ' : ''}${gained.toFixed(1)} years in this scenario. Try the Longevity Simulator:`
    : 'How long could you live if we cured diseases or slowed aging? Try the Longevity Simulator:';
  writeUrl(inp);

  // --- Gompertz ---
  $('gompertz-a-values').innerHTML = gompertzRows(r.baseFit.a, r.intFit.a, active, intColor);
  $('gompertz-b-values').innerHTML = gompertzRows(r.baseFit.b, r.intFit.b, active, intColor);

  // --- Survival figure ---
  const valsB = baseSurv.vals.map((v) => v * 100);
  const valsI = intSurv.vals.map((v) => v * 100);
  const diff = valsI.map((v, k) => v - valsB[k]);
  const dMin = Math.min(...diff), dMax = Math.max(...diff);
  let fill = {};
  if (active && dMin >= -1e-9 && dMax > 1e-9) fill = { fill: 'tonexty', fillcolor: 'rgba(44, 160, 44, 0.10)' };
  else if (active && dMax <= 1e-9 && dMin < -1e-9) fill = { fill: 'tonexty', fillcolor: 'rgba(220, 53, 69, 0.10)' };

  const survTraces = [{
    x: baseSurv.ages, y: valsB, mode: 'lines', name: 'Today', line: { color: COLOR_BASELINE, width: 2 },
    text: hoverSurv(baseSurv.ages, valsB, r.baselineMortality), hoverinfo: 'text',
  }];
  if (active) survTraces.push({
    x: intSurv.ages, y: valsI, mode: 'lines', name: 'With your changes', line: { color: intColor, width: 3 },
    text: hoverSurv(intSurv.ages, valsI, r.interventionMortality), hoverinfo: 'text', ...fill,
  });

  const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 50, y1: 50, line: { color: '#adb5bd', width: 1, dash: 'dot' } }];
  const annotations = [{ xref: 'paper', x: 0.01, y: 50, yref: 'y', text: '50% alive', showarrow: false, font: { color: '#6c757d', size: 10 }, yshift: 10 }];
  const marker = (x, y, label, color) => {
    shapes.push({ type: 'line', xref: 'x', x0: x, x1: x, yref: 'y', y0: 0, y1: 50, line: { color, width: 1, dash: 'dot' } });
    annotations.push({ x, y, xref: 'x', yref: 'y', text: `${label}: ${x.toFixed(0)}`, showarrow: false, font: { color, size: 10 }, bgcolor: 'rgba(248,249,250,0.85)' });
  };
  if (fin(baseMedian)) marker(baseMedian, 2, 'Today', COLOR_BASELINE);
  if (active && fin(intMedian) && intMedian <= baseSurv.ages[baseSurv.ages.length - 1]) marker(intMedian, 6, 'Scenario', intColor);

  Plotly.react('survival-graph', survTraces, {
    ...baseLayout,
    title: { text: cardTitle('chance of still being alive', poss), font: { size: 17 } },
    xaxis: { ...axis, title: { text: 'Age (years)' } },
    yaxis: { ...axis, title: { text: '% still alive' }, range: [0, 100], ticksuffix: '%' },
    shapes, annotations,
  }, PLOT_CONFIG);

  // --- Healthspan figure ---
  const xLo = userAge != null && userAge > 0 ? userAge : 0;
  const xHi = 95;
  const srcB = r.baselineConditions, srcI = r.interventionConditions;

  // Accelerated aging maps past the last GBD anchor and flattens; replace that tail with a
  // dashed linear extrapolation using the slope over the 5 yrs of real data before the cutoff.
  let extrapX = null, extrapY = null, cutoff = null;
  if (active && agingRate > 1) {
    const c = Math.floor(inp.slowStart + (GBD_DATA_MAX_AGE - inp.slowStart) / agingRate);
    if (c < xHi && c < srcI.length) {
      const s = Math.max(0, c - 5);
      if (c - s > 0) {
        const slope = (srcI[c] - srcI[s]) / (c - s);
        cutoff = c;
        extrapX = []; extrapY = [];
        for (let x = c; x <= xHi; x++) { extrapX.push(x); extrapY.push(srcI[c] + slope * (x - c)); }
      }
    }
  }

  const trim = (vals) => {
    const xs = [], ys = [];
    vals.forEach((v, age) => { if (age >= xLo && age <= xHi) { xs.push(age); ys.push(v); } });
    return [xs, ys];
  };
  const [hbX, hbY] = trim(srcB);
  let [siX, siY] = trim(srcI);
  if (cutoff != null) {
    const keep = siX.map((x) => x <= cutoff);
    siY = siY.filter((_, k) => keep[k]);
    siX = siX.filter((_, k) => keep[k]);
  }
  let fullX = siX, fullY = siY;
  if (extrapX && extrapX.length > 1) { fullX = siX.concat(extrapX.slice(1)); fullY = siY.concat(extrapY.slice(1)); }

  // Lower condition count is healthier, so fill polarity is flipped vs. the survival chart.
  let fillH = null;
  if (active && fullX.length && hbX.length) {
    const baseAt = (x) => (x <= hbX[0] ? hbY[0] : x >= hbX[hbX.length - 1] ? hbY[hbY.length - 1] : hbY[x - hbX[0]]);
    const d = fullX.map((x, k) => fullY[k] - baseAt(x));
    const mn = Math.min(...d), mx = Math.max(...d);
    if (mx <= 1e-9 && mn < -1e-9) fillH = 'rgba(44, 160, 44, 0.12)';
    else if (mn >= -1e-9 && mx > 1e-9) fillH = 'rgba(220, 53, 69, 0.12)';
  }

  const hoverH = 'Age %{x}<br>%{y:.2f} chronic conditions (expected)<extra></extra>';
  const hTraces = [{ x: hbX, y: hbY, mode: 'lines', name: 'Today', line: { color: COLOR_BASELINE, width: 2 }, hovertemplate: hoverH }];
  if (active) {
    if (fillH) hTraces.push({ x: fullX, y: fullY, mode: 'lines', line: { color: 'rgba(0,0,0,0)', width: 0 }, fill: 'tonexty', fillcolor: fillH, hoverinfo: 'skip', showlegend: false });
    hTraces.push({ x: siX, y: siY, mode: 'lines', name: 'With your changes', line: { color: intColor, width: 3 }, hovertemplate: hoverH });
    if (extrapX && extrapX.length > 1) hTraces.push({
      x: extrapX, y: extrapY, mode: 'lines', line: { color: COLOR_LOSS, width: 2, dash: 'dash' }, name: 'Extrapolated',
      showlegend: false, hovertemplate: 'Age %{x}<br>~%{y:.2f} (extrapolated)<extra></extra>',
    });
  }
  Plotly.react('healthspan-graph', hTraces, {
    ...baseLayout,
    title: { text: cardTitle('expected number of chronic conditions', poss), font: { size: 17 } },
    xaxis: { ...axis, title: { text: 'Age (years)' }, range: [xLo, xHi] },
    yaxis: { ...axis, title: { text: 'Expected # of chronic conditions' }, rangemode: 'tozero' },
  }, PLOT_CONFIG);
}

// --- Small callbacks ---

function renderAgingLabel() {
  const v = Number($('aging-rate').value);
  const s = num($('slow-start')) ?? DEFAULT_SLOW_START;
  const sfx = ` starting at age ${Math.trunc(s)}`;
  const age = num($('user-age'));
  const isCurrent = age != null && Math.trunc(s) === Math.trunc(age);
  $('slow-start-label').textContent = `Start at age ${Math.trunc(s)}${isCurrent ? ' (current age)' : ''}`;
  $('aging-rate-label').textContent =
    v === 100 ? 'Normal aging (100%)'
    : v === 0 ? `Frozen aging (0%)${sfx}`
    : v < 100 ? `Slow aging by ${100 - v}%${sfx}`
    : `Accelerated aging (${v}%)${sfx}`;
}

// Slow-start auto-tracks the user's age (floor 25) unless manually set to a valid override.
function syncSlowStart() {
  const userAge = num($('user-age'));
  const floor = Math.max(userAge != null ? Math.trunc(userAge) : 0, 25);
  const el = $('slow-start');
  el.min = floor;
  $('slow-start-min').textContent = floor;
  const cur = num(el);
  if (!slowStartOverridden || cur == null || cur < floor) el.value = floor;
}

function buildCauseButtons() {
  const wrap = $('cause-buttons');
  for (const { key, label } of data.causes) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    const paint = (on) => {
      b.className = `btn btn-sm ${on ? 'btn-success' : 'btn-outline-primary'} me-2 mb-2 rounded-pill`;
      b.setAttribute('aria-pressed', String(on));
    };
    paint(selected.includes(key));
    b.onclick = () => {
      const i = selected.indexOf(key);
      if (i >= 0) selected.splice(i, 1); else selected.push(key);
      paint(i < 0);
      render();
    };
    wrap.appendChild(b);
  }
}

// --- Shareable URL ---
// Name is deliberately left out of the URL so it isn't shared by accident.

function writeUrl(inp) {
  const q = new URLSearchParams();
  if (inp.userAge != null) q.set('age', inp.userAge);
  if ($('sex').value) q.set('sex', $('sex').value);
  if (selected.length) q.set('cure', selected.join(','));
  if (inp.agingPct !== 100) q.set('rate', inp.agingPct);
  if (slowStartOverridden) q.set('start', inp.slowStart);
  if (!inp.removeAccidents) q.set('acc', '1');
  if (inp.padTo !== 120) q.set('to', inp.padTo);
  const qs = q.toString().replace(/%2C/g, ',');
  history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

function readUrl() {
  const q = new URLSearchParams(location.search);
  const set = (id, k) => { if (q.has(k)) $(id).value = q.get(k); };
  set('user-age', 'age'); set('sex', 'sex'); set('aging-rate', 'rate'); set('pad-to', 'to');
  const keys = new Set(data.causes.map((c) => c.key));
  for (const k of (q.get('cure') ?? '').split(',')) if (keys.has(k)) selected.push(k);
  if (q.get('acc') === '1') $('remove-accidents').checked = false;
  if (q.has('start')) { slowStartOverridden = true; syncSlowStart(); $('slow-start').value = q.get('start'); }
}

let shareText = '';

async function share() {
  const url = location.href;
  if (navigator.share) {
    try { await navigator.share({ title: 'Longevity Simulator', text: shareText, url }); return; }
    catch (e) { if (e.name === 'AbortError') return; }
  }
  await navigator.clipboard.writeText(url);
  const b = $('btn-share');
  b.textContent = '✓ Link copied';
  setTimeout(() => { b.textContent = '↗ Share'; }, 2000);
}

const SOCIAL = {
  x: (u, t) => `https://x.com/intent/post?url=${u}&text=${t}`,
  linkedin: (u) => `https://www.linkedin.com/sharing/share-offsite/?url=${u}`,
  facebook: (u) => `https://www.facebook.com/sharer/sharer.php?u=${u}`,
  reddit: (u, t) => `https://www.reddit.com/submit?url=${u}&title=${t}`,
};

function openSocial(site) {
  const u = encodeURIComponent(location.href), t = encodeURIComponent(shareText);
  window.open(SOCIAL[site](u, t), '_blank', 'noopener,width=600,height=600');
}

// --- CSV export ---

function csvQuote(v) {
  const s = v == null ? '' : String(v);
  return /[,"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function download(target) {
  const inp = readInputs();
  const r = scenarioFor(inp);
  const labels = Object.fromEntries(data.causes.map((c) => [c.key, c.label]));
  const params = [
    ['Starting age', inp.userAge ?? ''],
    ['Sex', inp.sex],
    ['Diseases cured', selected.map((c) => labels[c] ?? c).join('; ')],
    ['Aging rate', inp.agingPct],
    ['Slow/speed aging starting at', inp.slowStart],
  ];
  const [cols, base, intv, filename] = target === 'lifespan'
    ? [['Age', 'Baseline % Chance Alive', 'Scenario % Chance Alive'],
       r.baselineSurvival.map((v) => v * 100), r.interventionSurvival.map((v) => v * 100), 'lifespan_curves.csv']
    : [['Age', 'Baseline Expected # Chronic Diseases', 'Scenario Expected # Chronic Diseases'],
       r.baselineConditions, r.interventionConditions, 'healthspan_curves.csv'];
  const lines = params.map(([k, v]) => `${csvQuote(k)},${csvQuote(v)}`);
  lines.push('', cols.map(csvQuote).join(','));
  base.forEach((b, age) => lines.push(`${age},${b.toFixed(4)},${intv[age].toFixed(4)}`));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// --- Wire up ---

readUrl();
buildCauseButtons();
$('user-age').addEventListener('input', () => { syncSlowStart(); renderAgingLabel(); render(); });
$('slow-start').addEventListener('input', () => {
  const s = num($('slow-start')), a = num($('user-age'));
  slowStartOverridden = s != null && a != null && Math.trunc(s) !== Math.trunc(a);
  renderAgingLabel(); render();
});
$('aging-rate').addEventListener('input', () => { renderAgingLabel(); render(); });
for (const id of ['user-name', 'sex', 'remove-accidents', 'pad-to']) $(id).addEventListener('input', render);
$('btn-download-lifespan').onclick = () => download('lifespan');
$('btn-download-healthspan').onclick = () => download('healthspan');
$('btn-share').onclick = share;
for (const b of document.querySelectorAll('[data-share]')) b.onclick = () => openSocial(b.dataset.share);

syncSlowStart();
renderAgingLabel();
render();
