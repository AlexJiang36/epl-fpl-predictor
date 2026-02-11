"use client";

import React, { useEffect, useMemo, useState } from "react";

type PredictionRow = Record<string, any>;
type OrderBy = "points" | "value" | "cost";
type SortDir = "desc" | "asc";
type Position = "GKP" | "DEF" | "MID" | "FWD" | "";
type Status = "" | "a" | "i" | "u" | "s";

/**
 * Teams dropdown (team_id).
 * You confirmed: Arsenal = 3.
 *
 * If any IDs differ in your DB, adjust here later (Day18+).
 * UI shows only team name (and optionally short name), not team_id.
 */
const TEAMS: Array<{ id: number; short: string; name: string }> = [
  { id: 3, short: "ARS", name: "Arsenal" }, // confirmed
  { id: 9, short: "CHE", name: "Chelsea" }, // seen in your sample
  { id: 14, short: "LIV", name: "Liverpool" }, // seen in your sample
  { id: 21, short: "WHU", name: "West Ham" }, // seen previously
  { id: 22, short: "WOL", name: "Wolves" }, // seen previously
  { id: 7, short: "BRE", name: "Brentford" }, // seen previously
  { id: 8, short: "BHA", name: "Brighton" }, // seen previously
  { id: 6, short: "BOU", name: "Bournemouth" },
  { id: 13, short: "LEE", name: "Leeds" }, // from your squad output (LEE)
  { id: 19, short: "SUN", name: "Sunderland" }, // from your squad output (SUN)
  { id: 4, short: "AVL", name: "Aston Villa" }, // seen in your sample
  // Add more teams as you confirm IDs from your backend responses.
];

function num(v: any): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function pick(row: PredictionRow, keys: string[], fallback: any = undefined) {
  for (const k of keys) {
    if (row && row[k] !== undefined && row[k] !== null) return row[k];
  }
  return fallback;
}

function fmt1(n: number | null) {
  if (n === null) return "-";
  return (Math.round(n * 10) / 10).toString();
}

function extractRows(payload: any): PredictionRow[] {
  if (Array.isArray(payload)) return payload;
  if (payload?.items && Array.isArray(payload.items)) return payload.items;
  if (payload?.data && Array.isArray(payload.data)) return payload.data;
  if (payload?.rows && Array.isArray(payload.rows)) return payload.rows;
  return [];
}

export default function PredictionsPage() {
  // -----------------------
  // Draft (form inputs)
  // -----------------------
  const [targetGw, setTargetGw] = useState<number>(26);
  const [modelName, setModelName] = useState<string>("");
  const [position, setPosition] = useState<Position>("");
  const [status, setStatus] = useState<Status>("");
  const [teamId, setTeamId] = useState<number | "">("");
  const [maxCost, setMaxCost] = useState<number | "">("");
  const [minPredPts, setMinPredPts] = useState<number | "">("");
  const [orderBy, setOrderBy] = useState<OrderBy>("points");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [limit, setLimit] = useState<number>(20);

  // -----------------------
  // Applied (used for query)
  // -----------------------
  const [applied, setApplied] = useState({
    targetGw: 26,
    modelName: "",
    position: "" as Position,
    status: "" as Status,
    teamId: "" as number | "",
    maxCost: "" as number | "",
    minPredPts: "" as number | "",
    orderBy: "points" as OrderBy,
    sortDir: "desc" as SortDir,
    limit: 20,
  });

  // Pagination offset is "applied" (Prev/Next should refresh)
  const [offset, setOffset] = useState<number>(0);

  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Whether user has ever fetched at least once (to enable auto refresh on paging)
  const [hasFetched, setHasFetched] = useState(false);

  const canFetchDraft =
    Number.isFinite(targetGw) && targetGw > 0 && Number.isFinite(limit) && limit > 0;

  // Query string is based on APPLIED filters + offset
  const appliedQueryString = useMemo(() => {
    const sp = new URLSearchParams();
    sp.set("target_gw", String(applied.targetGw));
    sp.set("limit", String(applied.limit));
    sp.set("offset", String(offset));
    sp.set("order_by", applied.orderBy); // backend expects: points|cost|value

    if (applied.modelName.trim()) sp.set("model_name", applied.modelName.trim());
    if (applied.position) sp.set("position", applied.position);
    if (applied.status) sp.set("status", applied.status);
    if (applied.teamId !== "") sp.set("team_id", String(applied.teamId));
    if (applied.maxCost !== "") sp.set("max_cost", String(applied.maxCost));
    if (applied.minPredPts !== "") sp.set("min_predicted_points", String(applied.minPredPts));

    return sp.toString();
  }, [applied, offset]);

  async function doFetch(qs: string) {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`/api/predictions?${qs}`, { cache: "no-store" });
      const body = await res.json();

      if (!res.ok) {
        const msg = body?.error?.message || body?.message || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      setRows(extractRows(body));
      setHasFetched(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setErr(msg);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  // ✅ Auto-refresh ONLY when paging changes (Prev/Next), after user has fetched once
  useEffect(() => {
    if (!hasFetched) return;
    doFetch(appliedQueryString);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset]);

  function onApplyAndFetch() {
    if (!canFetchDraft) return;

    const nextApplied = {
      targetGw,
      modelName,
      position,
      status,
      teamId,
      maxCost,
      minPredPts,
      orderBy,
      sortDir,
      limit,
    };

    setApplied(nextApplied);
    setOffset(0);

    // Fetch using the same values (avoid waiting for state updates)
    const sp = new URLSearchParams();
    sp.set("target_gw", String(nextApplied.targetGw));
    sp.set("limit", String(nextApplied.limit));
    sp.set("offset", "0");
    sp.set("order_by", nextApplied.orderBy);

    if (nextApplied.modelName.trim()) sp.set("model_name", nextApplied.modelName.trim());
    if (nextApplied.position) sp.set("position", nextApplied.position);
    if (nextApplied.status) sp.set("status", nextApplied.status);
    if (nextApplied.teamId !== "") sp.set("team_id", String(nextApplied.teamId));
    if (nextApplied.maxCost !== "") sp.set("max_cost", String(nextApplied.maxCost));
    if (nextApplied.minPredPts !== "") sp.set("min_predicted_points", String(nextApplied.minPredPts));

    doFetch(sp.toString());
  }

  function onPrev() {
    setOffset((o) => Math.max(0, o - applied.limit));
  }

  function onNext() {
    setOffset((o) => o + applied.limit);
  }

  // Display mapping aligned to your backend response:
  // - web_name
  // - now_cost (int, 0.1m units)
  // - predicted_points
  // - team_short_name
  function rowView(r: PredictionRow) {
    const name = pick(r, ["name", "player_name", "web_name"], "");
    const pos = pick(r, ["position", "pos"], "");
    const teamShown = pick(r, ["team_short_name", "team", "team_name"], "");

    const nowCost = num(pick(r, ["now_cost"], null));
    const costM = nowCost !== null ? nowCost / 10 : num(pick(r, ["cost_m", "cost"], null));

    const pred = num(pick(r, ["predicted_points", "points", "pred_points"], null));

    const valueComputed = pred !== null && costM !== null && costM > 0 ? pred / costM : null;

    return { name, pos, team: teamShown, cost: costM, pred, value: valueComputed };
  }

  // ✅ Front-end sorting for current page (supports asc/desc even if backend doesn't)
  const sortedRows = useMemo(() => {
    const arr = [...rows];
    const dir = applied.sortDir;

    const keyFn = (r: PredictionRow) => {
      const pred = num(pick(r, ["predicted_points", "points", "pred_points"], null)) ?? -1;
      const nowCost = num(pick(r, ["now_cost"], null));
      const costM =
        nowCost !== null ? nowCost / 10 : (num(pick(r, ["cost_m", "cost"], null)) ?? -1);
      const value = costM > 0 ? pred / costM : -1;

      if (applied.orderBy === "points") return pred;
      if (applied.orderBy === "cost") return costM;
      return value; // "value"
    };

    arr.sort((a, b) => {
      const ka = keyFn(a);
      const kb = keyFn(b);
      return dir === "asc" ? ka - kb : kb - ka;
    });

    return arr;
  }, [rows, applied.orderBy, applied.sortDir]);

  const requestHref = useMemo(() => {
    return `http://localhost:3000/api/predictions?${appliedQueryString}`;
  }, [appliedQueryString]);

  return (
    <main className="max-w-6xl mx-auto p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Predictions</h1>
        <p className="text-sm text-gray-600">
          Day17: manual fetch; paging auto refresh; client-side asc/desc sorting for current page.
        </p>
      </header>

      {/* Filters */}
      <section className="border rounded-lg p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">target_gw</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              min={1}
              value={targetGw}
              onChange={(e) => setTargetGw(Number(e.target.value))}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">model_name</span>
            <input
              className="border rounded px-2 py-1"
              placeholder="(optional)"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">position</span>
            <select
              className="border rounded px-2 py-1"
              value={position}
              onChange={(e) => setPosition(e.target.value as Position)}
            >
              <option value="">(any)</option>
              <option value="GKP">GKP</option>
              <option value="DEF">DEF</option>
              <option value="MID">MID</option>
              <option value="FWD">FWD</option>
            </select>
          </label>

          {/* ✅ Status dropdown */}
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">status</span>
            <select
              className="border rounded px-2 py-1"
              value={status}
              onChange={(e) => setStatus(e.target.value as Status)}
              title="Common statuses: a=available, i=injured, u=unavailable, s=suspended"
            >
              <option value="">(any)</option>
              <option value="a">a (available)</option>
              <option value="i">i (injured)</option>
              <option value="u">u</option>
              <option value="s">s</option>
            </select>
          </label>

          {/* ✅ Team dropdown (team_id), short label */}
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">team</span>
            <select
              className="border rounded px-2 py-1"
              value={teamId === "" ? "" : String(teamId)}
              onChange={(e) => {
                const v = e.target.value;
                setTeamId(v === "" ? "" : Number(v));
              }}
            >
              <option value="">(any)</option>
              {TEAMS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.short})
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">order_by</span>
            <select
              className="border rounded px-2 py-1"
              value={orderBy}
              onChange={(e) => setOrderBy(e.target.value as OrderBy)}
            >
              <option value="points">points</option>
              <option value="value">value</option>
              <option value="cost">cost</option>
            </select>
          </label>

          {/* ✅ asc/desc (client-side sort for current page) */}
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">sort_dir</span>
            <select
              className="border rounded px-2 py-1"
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value as SortDir)}
              title="Client-side sorting for the current page"
            >
              <option value="desc">desc</option>
              <option value="asc">asc</option>
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">max_cost</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              step={0.1}
              placeholder="(optional)"
              value={maxCost}
              onChange={(e) => {
                const v = e.target.value;
                setMaxCost(v === "" ? "" : Number(v));
              }}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">min_predicted_points</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              step={0.1}
              placeholder="(optional)"
              value={minPredPts}
              onChange={(e) => {
                const v = e.target.value;
                setMinPredPts(v === "" ? "" : Number(v));
              }}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">limit</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              min={1}
              max={200}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>

          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium">offset</span>
            <div className="border rounded px-2 py-1 text-sm text-gray-700">{offset}</div>
          </div>

          <div className="md:col-span-2 flex items-end gap-3">
            <button
              className="border rounded px-4 py-2 font-medium disabled:opacity-60"
              onClick={onApplyAndFetch}
              disabled={loading || !canFetchDraft}
            >
              {loading ? "Loading..." : "Fetch Predictions"}
            </button>

            <div className="text-xs text-gray-500 break-all">
              Request:{" "}
              <a className="underline" href={requestHref} target="_blank" rel="noreferrer">
                /api/predictions?{appliedQueryString}
              </a>
            </div>
          </div>
        </div>

        {err ? (
          <div className="border rounded-md p-3 bg-red-50 text-red-800 text-sm">
            <div className="font-semibold">Error</div>
            <div className="break-words">{err}</div>
          </div>
        ) : null}
      </section>

      {/* Pagination */}
      <section className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          Showing <span className="font-medium">{sortedRows.length}</span> rows
        </div>
        <div className="flex gap-2">
          <button
            className="border rounded px-3 py-1 disabled:opacity-60"
            onClick={onPrev}
            disabled={loading || offset === 0 || !hasFetched}
          >
            Prev
          </button>
          <button
            className="border rounded px-3 py-1 disabled:opacity-60"
            onClick={onNext}
            disabled={loading || !hasFetched || rows.length < applied.limit}
            title={rows.length < applied.limit ? "No more pages (returned less than limit)" : ""}
          >
            Next
          </button>
        </div>
      </section>

      {/* Table */}
      <section className="border rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2 border-b">Name</th>
                <th className="text-left px-3 py-2 border-b">Pos</th>
                <th className="text-left px-3 py-2 border-b">Team</th>
                <th className="text-right px-3 py-2 border-b">Cost (m)</th>
                <th className="text-right px-3 py-2 border-b">PredPts</th>
                <th className="text-right px-3 py-2 border-b">Value</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.length === 0 ? (
                <tr>
                  <td className="px-3 py-3 text-gray-500" colSpan={6}>
                    {hasFetched ? "No rows for current filters." : 'No rows. Click "Fetch Predictions".'}
                  </td>
                </tr>
              ) : (
                sortedRows.map((r, i) => {
                  const v = rowView(r);
                  return (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-3 py-2 border-b whitespace-nowrap">
                        {v.name || <span className="text-gray-400">-</span>}
                      </td>
                      <td className="px-3 py-2 border-b">{v.pos || "-"}</td>
                      <td className="px-3 py-2 border-b">{v.team || "-"}</td>
                      <td className="px-3 py-2 border-b text-right">{fmt1(v.cost)}</td>
                      <td className="px-3 py-2 border-b text-right">{fmt1(v.pred)}</td>
                      <td className="px-3 py-2 border-b text-right">{fmt1(v.value)}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}