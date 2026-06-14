// Module augmentation so `session.user.tier` / `.inGuild` and the JWT fields are
// typed across the app. See auth.ts for how they're populated.
import type { Tier } from "@/auth";
import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      tier: Tier;
      inGuild: boolean;
    } & DefaultSession["user"];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    tier?: Tier;
    inGuild?: boolean;
  }
}
