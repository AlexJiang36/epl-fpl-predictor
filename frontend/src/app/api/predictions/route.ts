import { NextRequest } from "next/server";
import { proxyGET } from "@/lib/bffProxy";

export async function GET(req: NextRequest) {
  return proxyGET(req, "/predictions");
}