"use client";

import React, { useEffect, useMemo, useState } from "react";

type PredictionRow = Record<string, any>;
type OrderBy = "points" | "value" | "cost";
type SortDir = "desc" | "asc";
type Position = "GKP" | "DEF" | "MID" | "FWD" | "";
type Status = "" | "a" | "i" | "u" | "s";

type TeamOpt = { id: number; name: string; short_name: string; fpl_team_id?: number };

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

function extractTotal(payload: any): number | null {
  const t = payload?.meta?.total;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
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
  // Applied (used for the dataset fetch)
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

  // Day19: offset is ONLY for client-side pagination (slice)
  const [offset, setOffset] = useState<number>(0);

  // Day19: store the full dataset for current applied filters
  const [allRows, setAllRows] = useState<PredictionRow[]>([]);
  const [total, setTotal] = useState<number>(0);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Whether user has fetched at least once
  const [hasFetched, setHasFetched] = useState(false);

  // -----------------------
  // Day18: teams dropdown from backend
  // -----------------------
  const [teams, setTeams] = useState<TeamOpt[]>([]);
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [teamsErr, setTeamsErr] = useState<string | null>(null);

  async function loadTeams() {
    setTeamsLoading(true);
    setTeamsErr(null);
    try {
      const res = await fetch("/api/teams", { cache: "no-store" });
      const body = await res.json();

      if (!res.ok) {
        const msg = body?.error?.message || body?.message || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      const listRaw = Array.isArray(body?.teams) ? body.teams : [];
      const list: TeamOpt[] = listRaw
        .map((t: any) => ({
          id: Number(t.id),
          name: String(t.name ?? ""),
          short_name: String(t.short_name ?? ""),
          fpl_team_id: t.fpl_team_id !== undefined ? Number(t.fpl_team_id) : undefined,
        }))
        .filter((t: TeamOpt) => Number.isFinite(t.id) && t.name.length > 0);

      list.sort((a, b) => a.name.localeCompare(b.name));
      setTeams(list);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setTeamsErr(msg);
      setTeams([]);
    } finally {
      setTeamsLoading(false);
    }
  }

  useEffect(() => {
    loadTeams();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canFetchDraft =
    Number.isFinite(targetGw) && targetGw > 0 && Number.isFinite(limit) && limit > 0;

  // Build query string for the dataset fetch (Day19: we fetch "all rows" once)
  const datasetQueryString = useMemo(() => {
    const sp = new URLSearchParams();
    sp.set("target_gw", String(applied.targetGw));

    // fetch all: pick a high limit that exceeds total players (safe for your scale)
    // If you later grow to bigger datasets, we can loop pages; not needed now.
    sp.set("limit", "200");
    sp.set("offset", "0");

    // backend needs order_by, but Day19 does global sorting on client
    // keep it stable (doesn't matter much). Use cost for determinism.
    sp.set("order_by", "cost");

    if (applied.modelName.trim()) sp.set("model_name", applied.modelName.trim());
    if (applied.position) sp.set("position", applied.position);
    if (applied.status) sp.set("status", applied.status);
    if (applied.teamId !== "") sp.set("team_id", String(applied.teamId));
    if (applied.maxCost !== "") sp.set("max_cost", String(applied.maxCost));
    if (applied.minPredPts !== "") sp.set("min_predicted_points", String(applied.minPredPts));

    return sp.toString();
  }, [applied]);

  async function fetchAllPredictions(appliedQS: URLSearchParams) {
    setLoading(true);
    setErr(null);

    try {
        const PAGE_LIMIT = 200; // keep within backend validation
        let pageOffset = 0;

        const merged: PredictionRow[] = [];
        let totalSeen: number | null = null;

        while (true) {
        const sp = new URLSearchParams(appliedQS.toString());
        sp.set("limit", String(PAGE_LIMIT));
        sp.set("offset", String(pageOffset));
        // backend requires order_by; stable value is fine because we sort client-side later
        sp.set("order_by", "cost");

        const res = await fetch(`/api/predictions?${sp.toString()}`, { cache: "no-store" });
        const body = await res.json();

        if (!res.ok) {
            const msg = body?.error?.message || body?.message || `Request failed (${res.status})`;
            throw new Error(msg);
        }

        const rows = extractRows(body);
        const t = extractTotal(body);
        if (t !== null) totalSeen = t;

        merged.push(...rows);

        // stop conditions:
        // 1) backend tells total, and we reached it
        if (totalSeen !== null && merged.length >= totalSeen) break;

        // 2) backend returned fewer than PAGE_LIMIT rows => no more pages
        if (rows.length < PAGE_LIMIT) break;

        pageOffset += PAGE_LIMIT;

        // safety guard to avoid infinite loops
        if (pageOffset > 5000) break;
        }

        setAllRows(merged);
        setTotal(totalSeen ?? merged.length);
        setHasFetched(true);
    } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setErr(msg);
        setAllRows([]);
        setTotal(0);
    } finally {
        setLoading(false);
    }
}

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

    // Build QS from nextApplied (avoid waiting for state updates)
    const sp = new URLSearchParams();
    sp.set("target_gw", String(nextApplied.targetGw));
    sp.set("limit", "200");
    sp.set("offset", "0");
    sp.set("order_by", "cost");

    if (nextApplied.modelName.trim()) sp.set("model_name", nextApplied.modelName.trim());
    if (nextApplied.position) sp.set("position", nextApplied.position);
    if (nextApplied.status) sp.set("status", nextApplied.status);
    if (nextApplied.teamId !== "") sp.set("team_id", String(nextApplied.teamId));
    if (nextApplied.maxCost !== "") sp.set("max_cost", String(nextApplied.maxCost));
    if (nextApplied.minPredPts !== "") sp.set("min_predicted_points", String(nextApplied.minPredPts));

    fetchAllPredictions(sp);
  }

  function onPrev() {
    setOffset((o) => Math.max(0, o - applied.limit));
  }

  function onNext() {
    setOffset((o) => o + applied.limit);
  }

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

  // ✅ Day19: global sorting on ALL rows
  const globallySortedRows = useMemo(() => {
    const dir = applied.sortDir;
    const arr = [...allRows];

    const keyFn = (r: PredictionRow) => {
      const pred = num(pick(r, ["predicted_points", "points", "pred_points"], null)) ?? -1;

      const nowCost = num(pick(r, ["now_cost"], null));
      const costM =
        nowCost !== null ? nowCost / 10 : (num(pick(r, ["cost_m", "cost"], null)) ?? -1);

      const value = costM > 0 ? pred / costM : -1;

      if (applied.orderBy === "points") return pred;
      if (applied.orderBy === "cost") return costM;
      return value; // value
    };

    arr.sort((a, b) => {
      const ka = keyFn(a);
      const kb = keyFn(b);
      return dir === "asc" ? ka - kb : kb - ka;
    });

    return arr;
  }, [allRows, applied.orderBy, applied.sortDir]);

  // ✅ Day19: paginate AFTER sorting
  const pageRows = useMemo(() => {
    const start = offset;
    const end = offset + applied.limit;
    return globallySortedRows.slice(start, end);
  }, [globallySortedRows, offset, applied.limit]);

  const totalPagesInfo = useMemo(() => {
    const t = total || globallySortedRows.length;
    const start = t === 0 ? 0 : offset + 1;
    const end = Math.min(offset + applied.limit, t);
    return { t, start, end };
  }, [total, globallySortedRows.length, offset, applied.limit]);

  const requestHref = useMemo(() => {
    // show the dataset fetch request (single fetch)
    return `http://localhost:3000/api/predictions?${datasetQueryString}`;
  }, [datasetQueryString]);

  const disableNext = useMemo(() => {
    const t = totalPagesInfo.t;
    return !hasFetched || loading || offset + applied.limit >= t;
  }, [hasFetched, loading, offset, applied.limit, totalPagesInfo.t]);

  return (
    <main className="max-w-6xl mx-auto p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Predictions</h1>
        <p className="text-sm text-gray-600">
          Day19: fetch full dataset once → <b>global sort</b> → <b>client paginate</b>. Manual fetch; no
          auto refresh on typing.
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

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">status</span>
            <select
              className="border rounded px-2 py-1"
              value={status}
              onChange={(e) => setStatus(e.target.value as Status)}
              title="a=available, i=injured, u=unavailable, s=suspended"
            >
              <option value="">(any)</option>
              <option value="a">a (available)</option>
              <option value="i">i (injured)</option>
              <option value="u">u</option>
              <option value="s">s</option>
            </select>
          </label>

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
              <option value="">{teamsLoading ? "Loading teams..." : "(any)"}</option>
              {teams.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.short_name})
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

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">sort_dir</span>
            <select
              className="border rounded px-2 py-1"
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value as SortDir)}
              title="Global sorting is applied BEFORE pagination in Day19"
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
              Dataset request:{" "}
              <a className="underline" href={requestHref} target="_blank" rel="noreferrer">
                /api/predictions?{datasetQueryString}
              </a>
            </div>
          </div>
        </div>

        {teamsErr ? (
          <div className="border rounded-md p-3 bg-yellow-50 text-yellow-900 text-sm">
            <div className="font-semibold">Teams load warning</div>
            <div className="break-words">{teamsErr}</div>
          </div>
        ) : null}

        {err ? (
          <div className="border rounded-md p-3 bg-red-50 text-red-800 text-sm">
            <div className="font-semibold">Error</div>
            <div className="break-words">{err}</div>
          </div>
        ) : null}
      </section>

      {/* Pagination info */}
      <section className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          {hasFetched ? (
            <>
              Showing <span className="font-medium">{totalPagesInfo.start}</span>–
              <span className="font-medium">{totalPagesInfo.end}</span> of{" "}
              <span className="font-medium">{totalPagesInfo.t}</span> (global-sorted)
            </>
          ) : (
            <>No rows. Click “Fetch Predictions”.</>
          )}
        </div>

        <div className="flex gap-2">
          <button
            className="border rounded px-3 py-1 disabled:opacity-60"
            onClick={onPrev}
            disabled={loading || offset === 0 || !hasFetched}
          >
            Prev
          </button>
          <button className="border rounded px-3 py-1 disabled:opacity-60" onClick={onNext} disabled={disableNext}>
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
              {pageRows.length === 0 ? (
                <tr>
                  <td className="px-3 py-3 text-gray-500" colSpan={6}>
                    {hasFetched ? "No rows for current filters." : 'No rows. Click "Fetch Predictions".'}
                  </td>
                </tr>
              ) : (
                pageRows.map((r, i) => {
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