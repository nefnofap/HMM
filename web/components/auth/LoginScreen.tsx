"use client";

// The login screen — this IS the home page ("/"). The whole app sits behind
// Discord login. The only action is "Sign in with Discord". After auth the user
// lands on /dashboard (or /denied if they aren't in the Discord server).
import { signIn } from "next-auth/react";

// Set to your server's invite. Used as the "not a member yet?" upsell.
const INVITE = "https://discord.com/invite/MSXdaexYdH";

// Regime ramp echoed from the dashboard, used as the brand motif here.
const RIBBON = [
  "#F26D5B",
  "#E8A13A",
  "#8A93A6",
  "#3FB6A8",
  "#54C98C",
  "#3FB6A8",
  "#8A93A6",
  "#54C98C",
  "#E8A13A",
  "#8A93A6",
  "#54C98C",
  "#3FB6A8",
];

function DiscordMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M20.317 4.369a19.79 19.79 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.211.375-.444.864-.608 1.249a18.27 18.27 0 0 0-5.487 0 12.6 12.6 0 0 0-.617-1.25.077.077 0 0 0-.079-.036A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.2 14.2 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.1 13.1 0 0 1-1.872-.892.077.077 0 0 1-.008-.128c.126-.094.252-.192.372-.291a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.061 0a.074.074 0 0 1 .078.009c.12.099.246.198.373.292a.077.077 0 0 1-.006.127 12.3 12.3 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.84 19.84 0 0 0 6.002-3.03.077.077 0 0 0 .032-.055c.5-5.177-.838-9.674-3.549-13.66a.06.06 0 0 0-.031-.028ZM8.02 15.331c-1.182 0-2.157-1.085-2.157-2.419 0-1.333.956-2.418 2.157-2.418 1.21 0 2.176 1.094 2.157 2.418 0 1.334-.956 2.419-2.157 2.419Zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.418 2.157-2.418 1.21 0 2.176 1.094 2.157 2.418 0 1.334-.946 2.419-2.157 2.419Z" />
    </svg>
  );
}

export default function LoginScreen() {
  return (
    <main className="bg-radial-fade relative flex min-h-[100vh] flex-col items-center justify-center overflow-hidden px-6">
      <div className="animate-fade-up relative z-10 w-full max-w-sm">
        {/* Regime ribbon — the signature motif, also the product's core artifact */}
        <div className="mb-8 flex h-8 w-full overflow-hidden rounded-md border border-ink-600">
          {RIBBON.map((c, i) => (
            <span
              key={i}
              className="h-full flex-1"
              style={{ backgroundColor: c, opacity: 0.85 }}
            />
          ))}
        </div>

        <div className="label-mono flex items-center gap-2">
          <span className="h-px w-5 bg-ink-500" />
          HMM · MARKET REGIMES
        </div>

        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-ink-100">
          Regime
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-ink-300">
          A Hidden Markov Model reads BTC into hidden states — bear, neutral,
          bull — and forecasts where it leans next. Members only; sign in with
          the Discord account you joined the server with.
        </p>

        <button
          type="button"
          onClick={() => signIn("discord", { callbackUrl: "/dashboard" })}
          className="pill pill-primary mt-7 inline-flex w-full justify-center gap-2 py-3"
        >
          <DiscordMark />
          Sign in with Discord
        </button>

        <p className="mt-4 text-xs text-ink-500">
          Not a member yet?{" "}
          <a
            className="text-ink-300 underline underline-offset-2 hover:text-ink-100"
            href={INVITE}
            target="_blank"
            rel="noreferrer"
          >
            Join the Discord
          </a>
        </p>
      </div>
    </main>
  );
}
