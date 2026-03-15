import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  const upstream = `${BACKEND_URL}/match/predictions/models`;

  const res = await fetch(upstream, { cache: "no-store" });
  const text = await res.text();

  if (!res.ok) {
    return NextResponse.json(
      { error: `Upstream ${res.status}`, body: text.slice(0, 300) },
      { status: res.status }
    );
  }

  return new NextResponse(text, {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
