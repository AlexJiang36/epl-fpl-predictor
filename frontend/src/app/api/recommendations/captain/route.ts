import { NextRequest, NextResponse } from "next/server";

const BACKEND_BASE_URL =
  process.env.BACKEND_BASE_URL || "http://127.0.0.1:8000";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);

    const targetGw = searchParams.get("target_gw");
    const modelName =
      searchParams.get("model_name") || "baseline_rollavg_v0";
    const limit = searchParams.get("limit") || "5";

    if (!targetGw) {
      return NextResponse.json(
        { error: "target_gw is required" },
        { status: 400 }
      );
    }

    const backendUrl = new URL(
      `${BACKEND_BASE_URL}/recommendations/captain`
    );
    backendUrl.searchParams.set("target_gw", targetGw);
    backendUrl.searchParams.set("model_name", modelName);
    backendUrl.searchParams.set("limit", limit);

    const res = await fetch(backendUrl.toString(), {
      method: "GET",
      cache: "no-store",
    });

    const data = await res.json();

    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    console.error("BFF /api/recommendations/captain failed:", error);
    return NextResponse.json(
      { error: "Failed to fetch captain recommendations" },
      { status: 500 }
    );
  }
}
