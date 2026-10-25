// Port of src/ (scenarios, causes, interventions, survival, healthspan).
// All series are plain arrays indexed by age, starting at age 0.

export const MEDIAN_CAP_AGE = 1000;

const last = (a) => a[a.length - 1];

function pad(a, padTo) {
  const out = a.slice();
  while (out.length < padTo + 1) out.push(last(a));
  return out;
}

// Ages past the fractions table reuse the last available fraction.
function removeCause(mx, frac) {
  return mx.map((m, age) => m * (1 - (age < frac.length ? frac[age] : last(frac))));
}

function stopAging(mx, finalAge, padTo) {
  const out = mx.slice();
  if (finalAge < out.length - 1) out.fill(out[finalAge], finalAge);
  return pad(out, padTo);
}

function slowAging(mx, factor, start, padTo) {
  const max = mx.length - 1;
  const out = mx.map((_, age) => {
    const bio = age < start ? age : Math.trunc(start + (age - start) * factor);
    return mx[Math.min(Math.max(bio, 0), max)];
  });
  return pad(out, padTo);
}

function applyAging(mx, rate, start, padTo) {
  if (rate === 1) return pad(mx, padTo);
  if (rate === 0) return stopAging(mx, start, padTo);
  return slowAging(mx, rate, start, padTo);
}

export function survivalCurve(mx) {
  let s = 1;
  return mx.map((m) => (s *= 1 - m));
}

// Linear interpolation, clamped at the ends (np.interp on integer ages 0..n-1).
function interp(x, ys) {
  if (x <= 0) return ys[0];
  if (x >= ys.length - 1) return last(ys);
  const i = Math.floor(x);
  return ys[i] + (ys[i + 1] - ys[i]) * (x - i);
}

function remapAging(series, rate, start) {
  if (rate === 1) return series.slice();
  return series.map((_, age) => interp(age < start ? age : start + (age - start) * rate, series));
}

function conditionCount(prev, buckets, padTo) {
  const n = Math.max(...Object.values(prev).map((a) => a.length));
  const sum = Array.from({ length: n }, (_, age) =>
    buckets.reduce((s, b) => s + (prev[b][Math.min(age, prev[b].length - 1)] ?? 0), 0));
  return pad(sum, padTo);
}

// Equivalent of LongevityScenario(...).get_data(pad_to) for both targets.
export function runScenario(data, { sex = 'All', agingRate = 1, slowStart = 25, removed = [], padTo = 120 }) {
  const baseMx = data.mortality[sex];
  const fractions = data.fractions[sex];
  const strip = (mx, causes) => causes.reduce((m, c) => (fractions[c] ? removeCause(m, fractions[c]) : m), mx);

  const baselineMortality = pad(baseMx, padTo);
  const interventionMortality = applyAging(strip(baseMx, removed), agingRate, slowStart, padTo);

  const prev = data.prevalence[sex];
  const buckets = data.healthspanBuckets;
  const baselineConditions = conditionCount(prev, buckets, padTo);
  const kept = buckets.filter((b) => !removed.includes(b));
  const interventionConditions = remapAging(
    kept.length ? conditionCount(prev, kept, padTo) : baselineConditions.map(() => 0),
    agingRate, slowStart);

  return {
    baselineMortality,
    interventionMortality,
    baselineSurvival: survivalCurve(baselineMortality),
    interventionSurvival: survivalCurve(interventionMortality),
    baselineConditions,
    interventionConditions,
  };
}

export function survivalAt(surv, age) {
  if (!surv.length) return 0;
  if (age >= surv.length - 1) return last(surv);
  return surv[Math.max(0, Math.round(age))];
}

// S(t | alive at anchorAge) for t >= anchorAge, as { ages, vals }.
export function conditionalSurvival(surv, anchorAge) {
  const all = { ages: surv.map((_, i) => i), vals: surv };
  if (anchorAge == null || anchorAge <= 0) return all;
  const anchor = survivalAt(surv, anchorAge);
  if (anchor <= 0) return all;
  const ages = [], vals = [];
  surv.forEach((s, age) => {
    if (age >= anchorAge) { ages.push(age); vals.push(Math.min(s / anchor, 1)); }
  });
  return { ages, vals };
}

// First age where survival < 0.5; otherwise extrapolate with the last mortality rate held constant.
export function medianFromSurvival({ ages, vals }, mort) {
  const i = vals.findIndex((v) => v < 0.5);
  if (i >= 0) return ages[i];
  if (!vals.length) return NaN;
  const lastS = last(vals), lastMx = last(mort);
  if (lastS <= 0 || !(lastMx > 0 && lastMx < 1)) return NaN;
  const median = last(ages) + Math.log(0.5 / lastS) / Math.log(1 - lastMx);
  return median > MEDIAN_CAP_AGE ? Infinity : median;
}

// Probability (%) of reaching targetAge, conditional on userAge. [null, null] if already past it.
export function chanceAtAge(target, baseSurv, intSurv, userAge) {
  if (userAge != null && userAge >= target) return [null, null];
  const pct = (surv) => {
    const t = survivalAt(surv, target);
    if (userAge != null && userAge > 0) {
      const anchor = survivalAt(surv, userAge);
      if (anchor > 0) return Math.min(Math.max((t / anchor) * 100, 0), 100);
    }
    return Math.min(Math.max(t * 100, 0), 100);
  };
  return [pct(baseSurv), pct(intSurv)];
}
