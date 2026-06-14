"use client";

// Shown to a signed-in user who is NOT a member of the required Discord guild.
// They have a session but no app access — their only paths are "join the
// Discord" or "sign in with a different account".
import { signOut } from "next-auth/react";

const INVITE = "https://discord.com/invite/MSXdaexYdH";

export default function DeniedPage() {
  return (
    <main className="bg-radial-fade flex min-h-screen items-center justify-center px-6">
      <div className="panel animate-fade-up w-full max-w-md p-8">
        <div className="label-mono">access · members only</div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-100">
          Join the Discord
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-ink-300">
          Regime is open to members of our Discord server only. Your account
          isn&apos;t in the server yet — join with the same Discord account, then
          sign in again to unlock the dashboard.
        </p>

        <div className="mt-6 flex flex-col gap-2">
          <a
            className="pill pill-primary justify-center"
            href={INVITE}
            target="_blank"
            rel="noreferrer"
          >
            Join the Discord server
          </a>
          <button
            type="button"
            onClick={() => signOut({ callbackUrl: "/" })}
            className="pill pill-ghost justify-center"
          >
            Sign in with a different account
          </button>
        </div>
      </div>
    </main>
  );
}
