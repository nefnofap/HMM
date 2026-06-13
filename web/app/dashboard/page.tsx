// The dashboard. Reads the precomputed regime bundle (written by the GitHub
// Action that runs the HMM across assets × sources) and renders it. The route
// is guarded by middleware, but we re-check the session here too so the page
// never renders for a non-member even if middleware is misconfigured.
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import bundle from "@/data/regimes.json";
import type { RegimeBundle } from "@/lib/regime";
import RegimeExplorer from "@/components/dashboard/RegimeExplorer";
import SignOutButton from "@/components/auth/SignOutButton";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/");
  if (!session.user.inGuild) redirect("/denied");

  const data = bundle as RegimeBundle;

  return (
    <main className="mx-auto min-h-screen w-full max-w-5xl px-5 py-8 sm:px-8">
      <header className="flex items-center justify-between">
        <div className="label-mono flex items-center gap-2">
          <span className="h-px w-5 bg-ink-500" />
          HMM · MARKET REGIMES
        </div>
        <div className="flex items-center gap-4">
          <span className="hidden text-sm text-ink-400 sm:inline">
            {session.user.name}
          </span>
          <SignOutButton />
        </div>
      </header>

      <RegimeExplorer bundle={data} />

      <footer className="mt-12 border-t border-ink-600 pt-5">
        <p className="text-xs leading-relaxed text-ink-500">
          Research tool, not advice. Regimes are decoded by a Hidden Markov Model
          and are only locally stationary — they describe the recent past and a
          short-horizon lean, not a guarantee. Spot is sourced from Yahoo
          Finance; perps from public exchange data. Keep live trading behind your
          own risk controls.
        </p>
      </footer>
    </main>
  );
}
