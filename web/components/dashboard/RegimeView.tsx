"use client";

import { useMemo, useState } from "react";
import {
  type RegimeData,
  metaById,
  pct,
  timeAgo,
} from "@/lib/regime";

export default function RegimeView({ data }: { data: RegimeData }) {
  const cur = metaById(data, data.current.stateId);
  const curHue = cur?.hue ?? "#8A93A6";

  return (
    <div className="animate-fade-up mt-6 space-y-6">
      <Hero data={data} curHue={curHue} />
      <Forecast data={data} />
      <Ribbon data={data} />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <TransitionMatrix data={data} />
        <StateLegend data={data} />
      </div>
    </div>
  );
}

/* ── Current regime ───────────────────────────────────────────────── */

function Hero({ data, curHue }: { data: RegimeData; curHue: string }) {
  const conf = Math.round(data.current.confidence * 100);
  const updated = timeAgo(data.generatedAt);
  return (
    <section className="panel p-6 sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="label-mono">current regime</div>
          <div className="mt-2 flex items-center gap-3">
            <span
              className="h-3.5 w-3.5 rounded-full"
              style={{ backgroundColor: curHue }}
            />
            <h1
              className="text-4xl font-semibold tracking-tight sm:text-5xl"
              style={{ color: curHue }}
            >
              {data.current.label}
            </h1>
          </div>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-ink-300">
            The model is <span className="stat-figure">{conf}%</span> confident{" "}
            {data.ticker} is in its{" "}
            <span style={{ color: curHue }}>{data.current.label.toLowerCase()}</span>{" "}
            state right now.
          </p>
        </div>

        <div className="text-right">
          <div className="label-mono">{data.ticker} · {data.interval}</div>
          <div className="mt-1 text-xs text-ink-500">updated {updated}</div>
          <div className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-ink-600 px-2.5 py-1">
            <span
              className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse-soft"
              aria-hidden
            />
            <span className="label-mono !tracking-[0.12em] text-ink-300">
              {data.refit ? "re-fit" : "decoded"}
            </span>
          </div>
        </div>
      </div>

      {/* Posterior over states */}
      <div className="mt-6">
        <div className="label-mono mb-2">state probability</div>
        <div className="space-y-1.5">
          {data.current.distribution.map((p, id) => {
            const m = metaById(data, id);
            return (
              <div key={id} className="flex items-center gap-3">
                <span className="w-20 shrink-0 truncate text-xs text-ink-400">
                  {m?.label ?? `S${id}`}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-800">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(p * 100, 1)}%`,
                      backgroundColor: m?.hue ?? "#8A93A6",
                    }}
                  />
                </div>
                <span className="stat-figure w-12 shrink-0 text-right text-xs">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ── Forecast ───────────────────────────────────────────────────── */

function Forecast({ data }: { data: RegimeData }) {
  return (
    <section className="panel p-6">
      <div className="label-mono">forecast · next {data.forecast.horizon} bars</div>
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {data.forecast.steps.map((s) => {
          // dominant state at this horizon
          let topId = 0;
          s.distribution.forEach((p, i) => {
            if (p > s.distribution[topId]) topId = i;
          });
          const m = metaById(data, topId);
          const up = s.expectedPct >= 0;
          return (
            <div key={s.step} className="rounded-lg border border-ink-600 p-4">
              <div className="label-mono text-ink-500">t+{s.step}</div>
              <div
                className="stat-figure mt-1 text-xl"
                style={{ color: up ? "#54C98C" : "#F26D5B" }}
              >
                {pct(s.expectedPct, 3)}
              </div>
              <div className="mt-2 flex items-center gap-1.5">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: m?.hue ?? "#8A93A6" }}
                />
                <span className="text-xs text-ink-400">
                  leans {m?.label.toLowerCase() ?? `s${topId}`}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ── Regime ribbon (signature) ─────────────────────────────────── */

function Ribbon({ data }: { data: RegimeData }) {
  const [hover, setHover] = useState<number | null>(null);
  const n = data.states.length;
  const W = 1000;
  const H = 120;

  // price polyline scaled into the band
  const { min, max } = useMemo(() => {
    let mn = Infinity,
      mx = -Infinity;
    for (const p of data.prices) {
      if (p < mn) mn = p;
      if (p > mx) mx = p;
    }
    return { min: mn, max: mx };
  }, [data.prices]);

  const pad = 10;
  const x = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const y = (p: number) =>
    H - pad - ((p - min) / Math.max(max - min, 1e-9)) * (H - 2 * pad);
  const line = data.prices.map((p, i) => `${x(i)},${y(p)}`).join(" ");

  const segW = W / n;
  const hoverMeta =
    hover != null ? metaById(data, data.states[hover]) : null;

  return (
    <section className="panel p-6">
      <div className="flex items-center justify-between">
        <div className="label-mono">decoded path · last {n} bars</div>
        <div className="text-xs text-ink-500">
          {hover != null ? (
            <span>
              <span style={{ color: hoverMeta?.hue }}>
                {hoverMeta?.label}
              </span>{" "}
              · {new Date(data.timestamps[hover]).toLocaleString()} ·{" "}
              <span className="stat-figure">
                {data.prices[hover].toLocaleString()}
              </span>
            </span>
          ) : (
            <span>hover the tape</span>
          )}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="mt-4 h-32 w-full"
        onMouseLeave={() => setHover(null)}
      >
        {/* regime segments */}
        {data.states.map((sid, i) => {
          const m = metaById(data, sid);
          return (
            <rect
              key={i}
              x={i * segW}
              y={0}
              width={segW + 0.6}
              height={H}
              fill={m?.hue ?? "#8A93A6"}
              opacity={hover == null || hover === i ? 0.32 : 0.18}
              onMouseEnter={() => setHover(i)}
            />
          );
        })}
        {/* price line */}
        <polyline
          points={line}
          fill="none"
          stroke="#E7ECF3"
          strokeWidth={1.4}
          vectorEffect="non-scaling-stroke"
          opacity={0.9}
        />
        {/* now marker */}
        <line
          x1={W - 1}
          y1={0}
          x2={W - 1}
          y2={H}
          stroke="#5EE6C7"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-1 flex justify-between text-[10px] text-ink-500">
        <span>{new Date(data.timestamps[0]).toLocaleDateString()}</span>
        <span className="text-signal">now</span>
      </div>
    </section>
  );
}

/* ── Transition matrix heatmap (structural device) ────────────── */

function TransitionMatrix({ data }: { data: RegimeData }) {
  const m = data.transitionMatrix;
  const labels = data.stateMeta.map((s) => s.label);
  return (
    <section className="panel p-6">
      <div className="label-mono">transition matrix · P(next | now)</div>
      <p className="mt-1 text-xs text-ink-500">
        Each row is a regime now; cells are the odds of the next bar&apos;s
        regime. Bright diagonals mean states persist.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-collapse text-center">
          <tbody>
            {m.map((row, i) => (
              <tr key={i}>
                <th className="pr-2 text-right text-[11px] font-normal text-ink-400">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: data.stateMeta[i]?.hue }}
                    />
                    {labels[i]}
                  </span>
                </th>
                {row.map((p, j) => (
                  <td key={j} className="p-0.5">
                    <div
                      className="flex aspect-square items-center justify-center rounded text-[11px]"
                      style={{
                        backgroundColor: data.stateMeta[j]?.hue ?? "#8A93A6",
                        opacity: 0.12 + p * 0.78,
                        color: p > 0.5 ? "#0B0E14" : "#C2CBD8",
                      }}
                      title={`${labels[i]} → ${labels[j]}: ${(p * 100).toFixed(1)}%`}
                    >
                      {(p * 100).toFixed(0)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ── Per-state legend ───────────────────────────────────────── */

function StateLegend({ data }: { data: RegimeData }) {
  const ordered = [...data.stateMeta].sort((a, b) => a.rank - b.rank);
  return (
    <section className="panel p-6">
      <div className="label-mono">regimes · {data.nComponents} states</div>
      <div className="mt-4 space-y-3">
        {ordered.map((m) => (
          <div key={m.id} className="flex items-center gap-3">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: m.hue }}
            />
            <span className="w-24 shrink-0 text-sm text-ink-200">{m.label}</span>
            <div className="flex flex-1 items-center justify-end gap-4 text-xs">
              <span className="text-ink-500">
                ann.{" "}
                <span
                  className="stat-figure"
                  style={{ color: m.annReturnPct >= 0 ? "#54C98C" : "#F26D5B" }}
                >
                  {pct(m.annReturnPct, 0)}
                </span>
              </span>
              <span className="text-ink-500">
                vol <span className="stat-figure text-ink-300">{m.annVolPct.toFixed(0)}%</span>
              </span>
              <span className="w-12 text-right text-ink-500">
                <span className="stat-figure text-ink-300">{m.sharePct.toFixed(0)}%</span>
              </span>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[11px] text-ink-500">
        Annualised return &amp; volatility are empirical, measured over bars the
        model assigned to each state. Share is time spent in each regime over the
        lookback.
      </p>
    </section>
  );
}
