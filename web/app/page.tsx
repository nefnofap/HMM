// Home page = the login screen. Already-signed-in members skip straight to the
// dashboard; signed-in non-members are sent to the join-the-Discord page.
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import LoginScreen from "@/components/auth/LoginScreen";

export const dynamic = "force-dynamic";

export default async function Page() {
  const session = await auth();
  if (session?.user) {
    redirect(session.user.inGuild ? "/dashboard" : "/denied");
  }
  return <LoginScreen />;
}
