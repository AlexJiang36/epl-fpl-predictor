"use client";

import React, { useState } from "react";

type Row = {
  fixture_id: number;
  gw: number;
  kickoff_time: string | null;
  home_team_name: string;
  away_team_name: string;
  pred_home_win: number | null;
  pred_draw: number | null;
  pred_away_win: number | null;
  pred_result: "H" | "D" | "A" | null;
  found: boolean;
};

export default function MatchPredictionsPage() {
  const [gw, setGw] = useState<number>(30);
  const [modelName, setModelName] = useState<string>("match_baseline_v0");
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onFetch() {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(
        `/api/match/predictions?gw=${gw}&model_name=${encodeURIComponent(modelName)}`,
        { cache: "no-store" }
      );
      const text = await res.text();
      if (!res.ok) throw new Error(text || `Request failed (${res.status})`);
      const data = JSON.parse(text);
      setRows(data.rows ?? []);
    } catch (e) {
      setRows([]);
      setErr(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function fmt3(x: number | null) {
    if (x === null || !Number.isFinite(x)) return "-";
    return (Math.round(x * 1000) / 1000).toFixed(3);
  }

  return (
    <main className="max-w-6xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-bold">Match Predictions</h1>

      <section className="border rounded-lg p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">Gameweek (gw)</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              min={1}
              value={gw}
              onChange={(e) => setGw(Number(e.target.value))}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">model_name</span>
            <input
              className="border rounded px-2 py-1"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </label>

          <div className="flex items-end">
            <button
              className="border rounded px-4 py-2 font-medium disabled:opacity-60"
              onClick={onFetch}
              disabled={loading}
            >
              {loading ? "Fetching..." : "Fetch Match Predictions"}
            </button>
          </div>
        </div>

        {err ? <div className="text-sm text-red-600">{err}</div> : null}
      </section>

      <section className="border rounded-lg p-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2 pr-3">Kickoff</th>
              <th className="py-2 pr-3">Fixture</th>
              <th className="py-2 pr-3">Pred (H/D/A)</th>
              <th className="py-2 pr-3">Result</th>
              <th className="py-2 pr-3">fixture_id</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.fixture_id} className="border-b">
                <td className="py-2 pr-3 whitespace-nowrap">
                  {r.kickoff_time ?? "-"}
                </td>
                <td className="py-2 pr-3">
                  {r.home_team_name} vs {r.away_team_name}
                </td>
                <td className="py-2 pr-3 tabular-nums whitespace-nowrap">
                  {fmt3(r.pred_home_win)} / {fmt3(r.pred_draw)} /{" "}
                  {fmt3(r.pred_away_win)}
                </td>
                <td className="py-2 pr-3 font-semibold">
                  {r.pred_result ?? "-"}
                </td>
                <td className="py-2 pr-3 tabular-nums">{r.fixture_id}</td>
              </tr>
            ))}

            {rows.length === 0 ? (
              <tr>
                <td className="py-4 text-gray-500" colSpan={5}>
                  No rows loaded. Select GW and click Fetch.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </section>
    </main>
  );
}