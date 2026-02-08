import { NextRequest } from "next/server";
import { proxyGET } from "@/lib/bffProxy";

const DEFAULT_TARGET_GW = "25";

export async function GET(req: NextRequest) {
  // Add a default target_gw if missing (backend requires it)
  if (!req.nextUrl.searchParams.has("target_gw")) {
    req.nextUrl.searchParams.set("target_gw", DEFAULT_TARGET_GW);
  }
  return proxyGET(req, "/recommendations/squad");
}