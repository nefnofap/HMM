"use client";

import { useMemo, useState } from "react";
import {
  type RegimeBundle,
  comboId,
  sourcesForAsset,
} from "@/lib/regime";
import RegimeView from "@/components/dashboard/RegimeView";

const SOURCE_LABEL: Record<string, string> = { spot: "Spot", perp: "Perp" };

export default function RegimeExplorer({ bundle }: { bundle: RegimeBundle }) {
  const firstAsset = bundle.assets[0]?.key ?? "";
  const [asset, setAsset] = useState(firstAsset);

  const sources = useMemo(() => sourcesForAsset(bundle, asset), [bundle, asset]);
  const [source, setSource] = useState<"spot" | "perp">(sources[0] ?? "spot");

  // If the chosen asset doesn't offer the current source, fall back.
  const activeSource = sources.includes(source) ? source : sources[0];
  const data = activeSource
    ? bundle.payloads[comboId(asset, activeSource)]
    : undefined;

  return (
    <div className="mt-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Asset selector */}
        <div className="flex flex-wrap gap-1.5">
          {bundle.assets.map((a) => {
            const on = a.key === asset;
            return (
              <button
                key={a.key}
                type="button"
                onClick={() => setAsset(a.key)}
                aria-pressed={on}
                className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
                  on
                    ? "border-signal/60 bg-signal/10 text-ink-100"
                    : "border-ink-600 text-ink-400 hover:border-ink-500 hover:text-ink-200"
                }`}
              >
                {a.key}
              </button>
            );
          })}
        </div>

        {/* Source toggle */}
        <div className="inline-flex rounded-full border border-ink-600 p-0.5">
          {(["spot", "perp"] as const).map((s) => {
            const available = sources.includes(s);
            const on = s === activeSource;
            return (
              <button
                key={s}
                type="button"
                disabled={!available}
                onClick={() => setSource(s)}
                aria-pressed={on}
                title={available ? "" : "No data for this asset yet"}
                className={`rounded-full px-3.5 py-1 text-sm transition-colors ${
                  on
                    ? "bg-ink-700 text-ink-100"
                    : available
                      ? "text-ink-400 hover:text-ink-200"
                      : "cursor-not-allowed text-ink-600"
                }`}
              >
                {SOURCE_LABEL[s]}
              </button>
            );
          })}
        </div>
      </div>

      {data ? (
        <RegimeView key={`${asset}:${activeSource}`} data={data} />
      ) : (
        <div className="panel mt-6 p-8 text-center text-sm text-ink-400">
          No regime data for {asset} yet. The next scheduled run will fill it in.
        </div>
      )}
    </div>
  );
}
