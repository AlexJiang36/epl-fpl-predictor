// src/app/api/recommendations/squad/route.ts

import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);

  // Backend base URL (local dev)
  const backendBase = process.env.BACKEND_BASE_URL ?? "http://127.0.0.1:8000";

  const url = `${backendBase}/recommendations/squad?${searchParams.toString()}`;

  const resp = await fetch(url, {
    method: "GET",
    // avoid caching during dev
    cache: "no-store",
  });

  const text = await resp.text();

  // If backend didn't return JSON, pass through a useful error
  const contentType = resp.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return NextResponse.json(
      {
        error: "Backend did not return JSON",
        status: resp.status,
        url,
        preview: text.slice(0, 200),
      },
      { status: 502 }
    );
  }

  // If JSON, forward it
  return new NextResponse(text, {
    status: resp.status,
    headers: { "content-type": "application/json" },
  });
}
