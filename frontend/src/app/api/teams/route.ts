// frontend/src/app/api/teams/route.ts
export async function GET() {
  const base = process.env.BACKEND_BASE_URL || "http://127.0.0.1:8000";
  const upstream = `${base}/teams`;

  const res = await fetch(upstream, { cache: "no-store" });
  const text = await res.text();

  return new Response(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}