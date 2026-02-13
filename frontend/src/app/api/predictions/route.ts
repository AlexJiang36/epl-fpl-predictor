// frontend/src/app/api/predictions/route.ts
export async function GET(req: Request) {
  const base = process.env.BACKEND_BASE_URL || "http://127.0.0.1:8000";
  const url = new URL(req.url);

  // forward full query string
  const upstream = `${base}/predictions?${url.searchParams.toString()}`;

  const res = await fetch(upstream, { cache: "no-store" });
  const text = await res.text();

  return new Response(text, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") || "application/json" },
  });
}