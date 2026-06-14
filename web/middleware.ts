// Route protection. Everything under /dashboard requires a Discord session whose
// holder is a member of the guild. Non-members are bounced to /denied (join-the-
// Discord upsell); signed-out visitors go to the login page ("/").
//
// The session is read from the JWT cookie only — no Discord API call happens
// here (membership was resolved once at sign-in and cached on the token), so
// this stays edge-safe and fast.
import { auth } from "@/auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const session = req.auth;

  if (!session) {
    return NextResponse.redirect(new URL("/", req.nextUrl));
  }
  if (!session.user?.inGuild) {
    return NextResponse.redirect(new URL("/denied", req.nextUrl));
  }
  return NextResponse.next();
});

export const config = {
  // Only guard the app surface; "/", "/denied", and /api/auth stay public.
  matcher: ["/dashboard/:path*"],
};
