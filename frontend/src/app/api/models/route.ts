// frontend/src/app/api/models/route.ts
import { proxyGET } from "../../../lib/proxy";

export async function GET(req: Request) {
  return proxyGET("/models", req);
}