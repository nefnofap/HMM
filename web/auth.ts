// Auth.js (NextAuth v5) — Discord OAuth with guild-membership gating + role tiers.
//
// Access rules:
//   • You can only reach the dashboard if you're a member of DISCORD_GUILD_ID.
//     Non-members get a session flagged `inGuild:false` and are bounced to
//     /denied by the middleware (they never see regime content).
//   • Members holding DISCORD_PREMIUM_ROLE_ID get tier "premium"; everyone else
//     in the guild is "free". (Tiers are plumbed through for later gating; the
//     regime dashboard itself is open to any guild member for now.)
//
// Membership + roles are read via the OAuth `guilds.members.read` scope, which
// lets us call GET /users/@me/guilds/{guild}/member with the user's own token —
// no bot token required. The result is cached onto the JWT, so the Discord API
// is only hit once at sign-in, not on every request.

import NextAuth from "next-auth";
import Discord from "next-auth/providers/discord";

const GUILD_ID = process.env.DISCORD_GUILD_ID ?? "";
const PREMIUM_ROLE_ID = process.env.DISCORD_PREMIUM_ROLE_ID ?? "";

export type Tier = "free" | "premium";

/**
 * Resolve the signed-in user's membership for our guild using their OAuth
 * access token.
 *   • 200 → member; tier depends on whether they hold the premium role.
 *   • 404 → not a member of the guild.
 *   • anything else (rate limit, outage) → treat as not-in-guild, don't crash.
 * If DISCORD_GUILD_ID is unset we fail OPEN (inGuild:true) so local dev without
 * a configured guild still works.
 */
async function resolveMembership(
  accessToken: string,
): Promise<{ inGuild: boolean; tier: Tier }> {
  if (!GUILD_ID) return { inGuild: true, tier: "free" };
  try {
    const res = await fetch(
      `https://discord.com/api/users/@me/guilds/${GUILD_ID}/member`,
      { headers: { Authorization: `Bearer ${accessToken}` } },
    );
    if (!res.ok) return { inGuild: false, tier: "free" };
    const member = (await res.json()) as { roles?: string[] };
    const roles = member.roles ?? [];
    const tier: Tier =
      PREMIUM_ROLE_ID && roles.includes(PREMIUM_ROLE_ID) ? "premium" : "free";
    return { inGuild: true, tier };
  } catch {
    return { inGuild: false, tier: "free" };
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  // Trust the deployment host header. Auto-true on Vercel; set explicitly so
  // self-hosted deploys work too. Pair with AUTH_URL in prod.
  trustHost: true,
  providers: [
    Discord({
      authorization: {
        params: { scope: "identify guilds guilds.members.read" },
      },
    }),
  ],
  pages: {
    signIn: "/", // the landing page IS the login screen
  },
  callbacks: {
    // Runs on sign-in (when `account` is present) and on every subsequent token
    // read. We resolve membership only at sign-in and cache it.
    async jwt({ token, account }) {
      if (account?.access_token) {
        const { inGuild, tier } = await resolveMembership(account.access_token);
        token.inGuild = inGuild;
        token.tier = tier;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.inGuild = (token.inGuild as boolean | undefined) ?? false;
        session.user.tier = (token.tier as Tier | undefined) ?? "free";
      }
      return session;
    },
  },
});
