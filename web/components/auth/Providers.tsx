"use client";

// Wraps the app in Auth.js's SessionProvider so client components (LoginScreen,
// SignOutButton) can call useSession()/signIn()/signOut(). Server components use
// auth() directly and don't need this.
import { SessionProvider } from "next-auth/react";

export default function Providers({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}
