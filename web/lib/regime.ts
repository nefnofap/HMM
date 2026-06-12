// Shape of data/regime.json (emitted by scripts/export_regime.py). Keep this in
// sync with that script's `payload` dict.

export interface StateMeta {
  id: number; // raw HMM state id (Viterbi label)
  rank: number; // 0 = most bearish .. n-1 = most bullish (sorted by mean return)
  label: string; // human label derived from rank ("Bear", "Neutral", "Bull"...)
  hue: string; // hex colour for this state across the whole UI
  meanLogReturn: number; // empirical mean log-return while in this state
  annReturnPct: number; // annualised return while in this state (%)
  annVolPct: number; // annualised volatility while in this state (%)
  sharePct: number; // % of the lookback window spent in this state
}

export interface ForecastStep {
  step: number; // t+step
  expectedLogReturn: number;
  expectedPct: number; // expected per-bar % move
  distribution: number[]; // P(state) at this horizon, indexed by raw state id
}

export interface RegimeData {
  schemaVersion: string;
  generatedAt: string; // ISO 8601
  ticker: string;
  interval: string; // e.g. "1h"
  lookbackDays: number;
  nComponents: number;
  refit: boolean; // true = model re-fit this run, false = saved model decoded
  // Down-sampled series for the ribbon (newest last). All same length.
  timestamps: string[];
  prices: number[];
  states: number[]; // raw state id per timestamp
  current: {
    stateId: number;
    label: string;
    confidence: number; // 0..1 posterior for the current state
    distribution: number[]; // full posterior, indexed by raw state id
  };
  forecast: {
    horizon: number;
    steps: ForecastStep[];
  };
  stateMeta: StateMeta[]; // one per raw state id, index === id
  transitionMatrix: number[][]; // row i -> P(next = j)
}

export function metaById(data: RegimeData, id: number): StateMeta | undefined {
  return data.stateMeta.find((m) => m.id === id);
}

export function pct(value: number, digits = 2): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}
