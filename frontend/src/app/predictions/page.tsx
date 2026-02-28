"use client";

import React, { useEffect, useMemo, useState } from "react";

type PredictionRow = Record<string, any>;

type OrderBy = "points" | "value" | "cost";

type SortDir = "desc" | "asc";

type Position = "GKP" | "DEF" | "MID" | "FWD" | "";

type Status = "" | "a" | "i" | "u" | "s";

type TeamOpt = { id: number; name: string; short_name: string; fpl_team_id?: number };

type ModelOpt = {
  model_name: string;
  label: string;
  source?: string;
  is_active?: boolean;
  notes?: string | null;
};

const DEFAULT_DRAFT = {
  targetGw: 26,
  modelName: "",
  position: "" as Position,
  status: "" as Status,
  teamId: "" as number | "",
  // IMPORTANT: UI uses "m" (e.g., 8.0, 15.0).
  // Backend filter expects FPL now_cost units (tenths), e.g. 80 means 8.0m.
  maxCostM: "" as number | "",
  minPredPts: "" as number | "",
  orderBy: "points" as OrderBy,
  sortDir: "desc" as SortDir,
  limit: 20,
};

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

function fmtDec(n: number | null, decimals: number) {
  if (n === null) return "-";
  const p = Math.pow(10, decimals);
  const v = Math.round(n * p) / p;
  return v.toFixed(decimals);
}

function fmtCost(n: number | null) {
  return fmtDec(n, 1); // e.g., 7.5m
}

function fmtPts(n: number | null) {
  return fmtDec(n, 1);
}

function fmtValue(n: number | null) {
  return fmtDec(n, 2);
}

function extractRows(payload: any): PredictionRow[] {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.rows)) return payload.rows;
  if (payload && Array.isArray(payload.items)) return payload.items;
  if (payload && Array.isArray(payload.data)) return payload.data;
  return [];
}

function extractMetaTotal(payload: any): number | null {
  const t = payload?.meta?.total;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

function costMToBackendUnits(costM: number): number {
  // 8.0 -> 80, 7.5 -> 75
  return Math.round(costM * 10);
}

export default function PredictionsPage() {
  // -----------------------
  // Draft (form inputs)
  // -----------------------
  const [targetGw, setTargetGw] = useState<number>(DEFAULT_DRAFT.targetGw);
  const [modelName, setModelName] = useState<string>(DEFAULT_DRAFT.modelName);
  const [position, setPosition] = useState<Position>(DEFAULT_DRAFT.position);
  const [status, setStatus] = useState<Status>(DEFAULT_DRAFT.status);
  const [teamId, setTeamId] = useState<number | "">(DEFAULT_DRAFT.teamId);
  const [maxCostM, setMaxCostM] = useState<number | "">(DEFAULT_DRAFT.maxCostM);
  const [minPredPts, setMinPredPts] = useState<number | "">(DEFAULT_DRAFT.minPredPts);
  const [orderBy, setOrderBy] = useState<OrderBy>(DEFAULT_DRAFT.orderBy);
  const [sortDir, setSortDir] = useState<SortDir>(DEFAULT_DRAFT.sortDir);
  const [limit, setLimit] = useState<number>(DEFAULT_DRAFT.limit);

  // -----------------------
  // Applied (used for the dataset fetch)
  // -----------------------
  const [applied, setApplied] = useState(() => ({ ...DEFAULT_DRAFT }));

  // models dropdown
  const [models, setModels] = useState<ModelOpt[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsErr, setModelsErr] = useState<string | null>(null);

  // teams dropdown
  const [teams, setTeams] = useState<TeamOpt[]>([]);
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [teamsErr, setTeamsErr] = useState<string | null>(null);

  // Day19: offset is ONLY for client-side pagination (slice)
  const [offset, setOffset] = useState<number>(0);

  // Day19: store the full dataset for current applied filters
  const [allRows, setAllRows] = useState<PredictionRow[]>([]);
  const [total, setTotal] = useState<number>(0);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Whether user has fetched at least once
  const [hasFetched, setHasFetched] = useState(false);

  // Load teams once
  useEffect(() => {
    let alive = true;
    async function load() {
      setTeamsLoading(true);
      setTeamsErr(null);
      try {
        const res = await fetch("/api/teams", { cache: "no-store" });
        const body = await res.json();
        if (!res.ok) throw new Error(body?.message || `teams ${res.status}`);

        const listRaw = Array.isArray(body) ? body : body?.rows ?? body?.teams ?? [];
        const list: TeamOpt[] = listRaw
          .map((t: any) => ({
            id: Number(t.id),
            name: String(t.name ?? ""),
            short_name: String(t.short_name ?? ""),
            fpl_team_id: t.fpl_team_id !== undefined ? Number(t.fpl_team_id) : undefined,
          }))
          .filter((t: TeamOpt) => Number.isFinite(t.id) && t.name.length > 0);

        list.sort((a, b) => a.name.localeCompare(b.name));
        if (alive) setTeams(list);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        if (alive) {
          setTeamsErr(msg);
          setTeams([]);
        }
      } finally {
        if (alive) setTeamsLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, []);

  // Load models once
  useEffect(() => {
    let alive = true;
    async function load() {
      setModelsLoading(true);
      setModelsErr(null);
      try {
        const res = await fetch("/api/models", { cache: "no-store" });
        const body = await res.json();
        if (!res.ok) throw new Error(body?.message || `models ${res.status}`);

        const listRaw = Array.isArray(body) ? body : body?.rows ?? body?.models ?? [];
        const list: ModelOpt[] = listRaw
          .map((m: any) => ({
            model_name: String(m.model_name ?? ""),
            label: String(m.label ?? m.model_name ?? ""),
            source: m.source ? String(m.source) : undefined,
            is_active: m.is_active !== undefined ? Boolean(m.is_active) : undefined,
            notes: m.notes ?? null,
          }))
          .filter((m: ModelOpt) => m.model_name.length > 0);

        list.sort((a, b) => a.label.localeCompare(b.label));
        if (alive) setModels(list);

        // Nice UX: if user hasn't chosen a model and baseline exists, preselect it.
        if (alive && !modelName) {
          const baseline = list.find((m) => m.model_name === "baseline_rollavg_v0");
          if (baseline) setModelName(baseline.model_name);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        if (alive) {
          setModelsErr(msg);
          setModels([]);
        }
      } finally {
        if (alive) setModelsLoading(false);
      }
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canFetchDraft =
    Number.isFinite(targetGw) && targetGw > 0 && Number.isFinite(limit) && limit > 0;

  // Build query string for the dataset fetch (Day19: fetch all rows once)
  const datasetQueryString = useMemo(() => {
    const sp = new URLSearchParams();
    sp.set("target_gw", String(applied.targetGw));

    // fetch all rows once, then global sort + client paginate
    sp.set("limit", "200");
    sp.set("offset", "0");

    // backend requires order_by; stable value is fine because we sort client-side later
    sp.set("order_by", "cost");

    if (applied.modelName.trim()) sp.set("model_name", applied.modelName.trim());
    if (applied.position) sp.set("position", applied.position);
    if (applied.status) sp.set("status", applied.status);
    if (applied.teamId !== "") sp.set("team_id", String(applied.teamId));

    // ✅ FIX: Convert cost in m to backend units (tenths).
    if (applied.maxCostM !== "") sp.set("max_cost", String(costMToBackendUnits(applied.maxCostM)));

    if (applied.minPredPts !== "") sp.set("min_predicted_points", String(applied.minPredPts));

    return sp.toString();
  }, [applied]);

  async function fetchAllPredictions(appliedQS: URLSearchParams) {
    setLoading(true);
    setErr(null);

    try {
      const PAGE_LIMIT = 200;
      let pageOffset = 0;

      const merged: PredictionRow[] = [];
      let totalSeen: number | null = null;

      while (true) {
        const sp = new URLSearchParams(appliedQS.toString());
        sp.set("limit", String(PAGE_LIMIT));
        sp.set("offset", String(pageOffset));
        sp.set("order_by", "cost");

        const res = await fetch(`/api/predictions?${sp.toString()}`, { cache: "no-store" });
        const body = await res.json();
        if (!res.ok) throw new Error(body?.error?.message || body?.message || `predictions ${res.status}`);

        const rows = extractRows(body);
        const t = extractMetaTotal(body);
        if (t !== null) totalSeen = t;

        merged.push(...rows);

        if (totalSeen !== null && merged.length >= totalSeen) break;
        if (rows.length < PAGE_LIMIT) break;

        pageOffset += PAGE_LIMIT;
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
      maxCostM,
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

    // ✅ FIX: Convert cost in m to backend units (tenths).
    if (nextApplied.maxCostM !== "") sp.set("max_cost", String(costMToBackendUnits(nextApplied.maxCostM)));

    if (nextApplied.minPredPts !== "") sp.set("min_predicted_points", String(nextApplied.minPredPts));

    fetchAllPredictions(sp);
  }

  function onResetFilters() {
    // Reset ONLY draft inputs, keep manual fetch behavior.
    setTargetGw(DEFAULT_DRAFT.targetGw);
    setModelName(DEFAULT_DRAFT.modelName);
    setPosition(DEFAULT_DRAFT.position);
    setStatus(DEFAULT_DRAFT.status);
    setTeamId(DEFAULT_DRAFT.teamId);
    setMaxCostM(DEFAULT_DRAFT.maxCostM);
    setMinPredPts(DEFAULT_DRAFT.minPredPts);
    setOrderBy(DEFAULT_DRAFT.orderBy);
    setSortDir(DEFAULT_DRAFT.sortDir);
    setLimit(DEFAULT_DRAFT.limit);

    setOffset(0);
  }

  function onPrev() {
    setOffset((o) => Math.max(0, o - applied.limit));
  }

  function onNext() {
    setOffset((o) => o + applied.limit);
  }

  function rowView(r: PredictionRow) {
    const name = pick(r, ["name", "player_name", "web_name"], "-");
    const pos = pick(r, ["position", "pos"], "");
    const teamShown = pick(r, ["team", "team_name", "team_short_name", "short_name"], "");

    // backend often stores cost as now_cost (integer tenths)
    const costRaw = num(pick(r, ["now_cost", "cost", "price"], null));
    const costM = costRaw !== null ? costRaw / 10 : null;

    const pred = num(pick(r, ["predicted_points", "points", "prediction", "pred_points"], null));
    const valueComputed = pred !== null && costM !== null && costM > 0 ? pred / costM : null;

    return { name, pos, team: teamShown, cost: costM, pred, value: valueComputed };
  }

  // Global sort on ALL rows
  const globallySortedRows = useMemo(() => {
    const dir = applied.sortDir;
    const arr = [...allRows];

    const keyFn = (r: PredictionRow) => {
      const v = rowView(r);
      if (applied.orderBy === "points") return v.pred ?? -Infinity;
      if (applied.orderBy === "value") return v.value ?? -Infinity;
      return v.cost ?? -Infinity;
    };

    arr.sort((a, b) => {
      const ka = keyFn(a);
      const kb = keyFn(b);
      return dir === "asc" ? ka - kb : kb - ka;
    });

    return arr;
  }, [allRows, applied.orderBy, applied.sortDir]);

  // paginate AFTER sorting
  const pageRows = useMemo(() => {
    const start = offset;
    const end = offset + applied.limit;
    return globallySortedRows.slice(start, end);
  }, [globallySortedRows, offset, applied.limit]);

  const pageStart = total === 0 ? 0 : offset + 1;
  const pageEnd = Math.min(offset + applied.limit, total);

  const requestHref = useMemo(() => `/api/predictions?${datasetQueryString}`, [datasetQueryString]);

  // Day25: show draft vs applied filter state clearly
  const hasDraftChanges = useMemo(() => {
    const dModel = modelName.trim();
    const aModel = applied.modelName.trim();

    return (
      targetGw !== applied.targetGw ||
      dModel !== aModel ||
      position !== applied.position ||
      status !== applied.status ||
      teamId !== applied.teamId ||
      maxCostM !== applied.maxCostM ||
      minPredPts !== applied.minPredPts ||
      orderBy !== applied.orderBy ||
      sortDir !== applied.sortDir ||
      limit !== applied.limit
    );
  }, [
    targetGw,
    modelName,
    position,
    status,
    teamId,
    maxCostM,
    minPredPts,
    orderBy,
    sortDir,
    limit,
    applied,
  ]);

  const appliedTeamLabel = useMemo(() => {
    if (applied.teamId === "") return null;
    const t = teams.find((x) => x.id === applied.teamId);
    if (!t) return `team ${applied.teamId}`;
    return `${t.name} (${t.short_name})`;
  }, [applied.teamId, teams]);

  const appliedModelLabel = useMemo(() => {
    const name = applied.modelName.trim();
    if (!name) return null;
    const m = models.find((x) => x.model_name === name);
    return m?.label || name;
  }, [applied.modelName, models]);

  const appliedChips = useMemo(() => {
    const chips: Array<{ k: string; v: string }> = [];

    chips.push({ k: "GW", v: String(applied.targetGw) });

    if (appliedTeamLabel) chips.push({ k: "Team", v: appliedTeamLabel });
    if (appliedModelLabel) chips.push({ k: "Model", v: appliedModelLabel });
    if (applied.position) chips.push({ k: "Pos", v: applied.position });
    if (applied.status) chips.push({ k: "Status", v: applied.status });

    if (applied.maxCostM !== "") chips.push({ k: "Max cost", v: `${applied.maxCostM}m` });
    if (applied.minPredPts !== "") chips.push({ k: "Min pts", v: String(applied.minPredPts) });

    chips.push({ k: "Order", v: `${applied.orderBy} ${applied.sortDir}` });

    return chips;
  }, [
    applied.targetGw,
    applied.teamId,
    applied.modelName,
    applied.position,
    applied.status,
    applied.maxCostM,
    applied.minPredPts,
    applied.orderBy,
    applied.sortDir,
    appliedTeamLabel,
    appliedModelLabel,
  ]);

  return (
    <main className="max-w-6xl mx-auto p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Predictions</h1>
        <p className="text-sm text-gray-600">Filter player predictions by gameweek, team, position, status, and model.</p>
      </header>

      {/* Filters */}
      <section className="border rounded-lg p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Gameweek</span>
            <input
              className="border rounded px-3 py-2"
              type="number"
              value={targetGw}
              onChange={(e) => setTargetGw(Number(e.target.value))}
              min={1}
            />
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Team</span>
            <select
              className="border rounded px-3 py-2"
              value={teamId}
              onChange={(e) => {
                const v = e.target.value;
                setTeamId(v === "" ? "" : Number(v));
              }}
              disabled={teamsLoading}
            >
              <option value="">{teamsLoading ? "Loading teams..." : "(any)"}</option>
              {teams.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.short_name})
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Position</span>
            <select
              className="border rounded px-3 py-2"
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

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Status</span>
            <select
              className="border rounded px-3 py-2"
              value={status}
              onChange={(e) => setStatus(e.target.value as Status)}
              title="a=available, i=injured, u=unavailable, s=suspended"
            >
              <option value="">(any)</option>
              <option value="a">a</option>
              <option value="i">i</option>
              <option value="u">u</option>
              <option value="s">s</option>
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm md:col-span-2">
            <span className="text-gray-700">Model</span>
            <select
              className="border rounded px-3 py-2"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={modelsLoading}
            >
              <option value="">{modelsLoading ? "Loading models..." : "(any)"}</option>
              {models.map((m) => (
                <option key={m.model_name} value={m.model_name}>
                  {m.label || m.model_name}
                </option>
              ))}
            </select>
            {modelsErr ? <div className="text-xs text-red-600">{modelsErr}</div> : null}
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Max cost (m)</span>
            <input
              className="border rounded px-3 py-2"
              type="number"
              value={maxCostM}
              onChange={(e) => {
                const v = e.target.value;
                setMaxCostM(v === "" ? "" : Number(v));
              }}
              min={0}
              step={0.1}
              title='Enter in "m" (e.g., 8.0). Backend stores cost as tenths: 8.0 -> 80.'
            />
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Min predicted points</span>
            <input
              className="border rounded px-3 py-2"
              type="number"
              value={minPredPts}
              onChange={(e) => {
                const v = e.target.value;
                setMinPredPts(v === "" ? "" : Number(v));
              }}
              min={0}
              step={0.1}
            />
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Order by</span>
            <select
              className="border rounded px-3 py-2"
              value={orderBy}
              onChange={(e) => setOrderBy(e.target.value as OrderBy)}
            >
              <option value="points">points</option>
              <option value="value">value</option>
              <option value="cost">cost</option>
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Sort dir</span>
            <select
              className="border rounded px-3 py-2"
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value as SortDir)}
              title="Sorting is applied to the full dataset before pagination"
            >
              <option value="desc">desc</option>
              <option value="asc">asc</option>
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-700">Limit</span>
            <input
              className="border rounded px-3 py-2"
              type="number"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              min={1}
              step={1}
            />
          </label>

          <div className="md:col-span-2 flex items-end gap-3">
            <button
              className="border rounded px-4 py-2 font-medium disabled:opacity-60"
              onClick={onApplyAndFetch}
              disabled={loading || !canFetchDraft}
            >
              {loading ? "Loading..." : "Fetch Predictions"}
            </button>

            <button
              type="button"
              className="border rounded px-4 py-2 font-medium disabled:opacity-60"
              onClick={onResetFilters}
              disabled={loading}
            >
              Reset Filters
            </button>

            {hasFetched && hasDraftChanges ? (
              <div className="text-xs text-amber-700">
                Draft differs from applied filters — click “Fetch Predictions” to apply.
              </div>
            ) : null}

            <div className="text-xs text-gray-500 break-all">
              Dataset request:{" "}
              <a className="underline" href={requestHref} target="_blank" rel="noreferrer">
                {requestHref}
              </a>
            </div>
          </div>
        </div>

        {teamsErr ? <div className="text-xs text-red-600">{teamsErr}</div> : null}
        {err ? <div className="text-sm text-red-600">{err}</div> : null}
      </section>

      {/* Results */}
      <section className="border rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            {hasFetched ? (
              <>
                Showing <span className="font-medium">{pageStart}</span>–{" "}
                <span className="font-medium">{pageEnd}</span> of{" "}
                <span className="font-medium">{total}</span>
              </>
            ) : (
              <>No rows. Click “Fetch Predictions”.</>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              className="border rounded px-3 py-1.5 text-sm disabled:opacity-60"
              onClick={onPrev}
              disabled={loading || offset === 0}
            >
              Prev
            </button>
            <button
              className="border rounded px-3 py-1.5 text-sm disabled:opacity-60"
              onClick={onNext}
              disabled={loading || offset + applied.limit >= total}
            >
              Next
            </button>
          </div>
        </div>

        {hasFetched ? (
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
            <span className="text-gray-500">Applied filters:</span>
            {appliedChips.map((c) => (
              <span key={c.k} className="inline-flex items-center rounded border px-2 py-0.5">
                <span className="text-gray-500 mr-1">{c.k}:</span>
                {c.v}
              </span>
            ))}
          </div>
        ) : null}

        <div className="overflow-x-auto overflow-y-auto max-h-[60vh]">
          <table className="min-w-full text-sm [font-variant-numeric:tabular-nums]">
            <thead>
              <tr className="border-b sticky top-0 bg-white z-10">
                <th className="text-left py-2 pr-4">Name</th>
                <th className="text-left py-2 pr-4">Pos</th>
                <th className="text-left py-2 pr-4">Team</th>
                <th className="text-right py-2 pr-4">Cost</th>
                <th className="text-right py-2 pr-4">Pred pts</th>
                <th className="text-right py-2 pr-2">Value</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.length === 0 ? (
                <tr>
                  <td className="py-4 text-gray-500" colSpan={6}>
                    {loading
                      ? "Loading…"
                      : err
                      ? "Couldn’t load predictions. Check your filters and try Fetch again."
                      : hasFetched
                      ? "No rows match the applied filters."
                      : 'No rows yet. Adjust filters, then click "Fetch Predictions".'}
                  </td>
                </tr>
              ) : (
                pageRows.map((r, idx) => {
                  const v = rowView(r);
                  return (
                    <tr key={idx} className="border-b last:border-b-0 hover:bg-gray-50">
                      <td className="py-2 pr-4">{v.name}</td>
                      <td className="py-2 pr-4">{v.pos}</td>
                      <td className="py-2 pr-4">{v.team}</td>
                      <td className="py-2 pr-4 text-right">{fmtCost(v.cost)}</td>
                      <td className="py-2 pr-4 text-right">{fmtPts(v.pred)}</td>
                      <td className="py-2 pr-2 text-right">{fmtValue(v.value)}</td>
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
