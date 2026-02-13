// frontend/src/app/api/teams/route.ts
import { proxyGET } from "../../../lib/proxy";

export async function GET(req: Request) {
  return proxyGET(req, "/teams");
}