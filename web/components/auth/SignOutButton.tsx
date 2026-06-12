"use client";

import { signOut } from "next-auth/react";

export default function SignOutButton() {
  return (
    <button
      type="button"
      onClick={() => signOut({ callbackUrl: "/" })}
      className="label-mono hover:text-ink-200 transition-colors"
    >
      Sign out
    </button>
  );
}
