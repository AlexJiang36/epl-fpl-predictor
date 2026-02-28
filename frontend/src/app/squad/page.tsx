"use client";

import React, { useEffect, useMemo, useState } from "react";

type Status = "" | "a" | "i" | "u" | "s";
type ViewMode = "compact" | "full";

type Player = {
  name?: string;
  web_name?: string;
  position?: "GKP" | "DEF" | "MID" | "FWD" | string;
  team?: string;
  team_short_name?: string;
  team_name?: string;

  // cost
  cost_m?: number; // e.g. 5.9
  now_cost?: number; // e.g. 59 (x10)

  // points
  predicted_points?: number;

  // ids
  player_id?: number;
  fpl_player_id?: number;
  id?: number;
};

type SquadSummary = {
  spent_m?: number;
  remaining_m?: number;
  team_counts?: Record<string, number>;
  squad_counts?: Record<string, number>;
};

type SquadResponse = {
  target_gw?: number;
  model_name?: string;
  generated_at?: string;
  summary?: SquadSummary;

  starting_xi?: Record<string, Player[]>;
  bench?: Record<string, Player[]>;
  bench_list?: Player[];

  // tolerate older shapes
  squad_list?: Player[];
};

type ModelOpt = {
  model_name: string;
  label?: string;
};

function fmt1(n: number) {
  return (Math.round(n * 10) / 10).toFixed(1);
}
function fmt2(n: number) {
  return (Math.round(n * 100) / 100).toFixed(2);
}

function getName(p: Player): string {
  return String(p.name ?? p.web_name ?? "").trim();
}

function getTeam(p: Player): string {
  return String(p.team ?? p.team_short_name ?? p.team_name ?? "").trim();
}

function getPredPts(p: Player): number {
  const v = p.predicted_points ?? 0;
  return Number.isFinite(v) ? Number(v) : 0;
}

function getCostM(p: Player): number | null {
  if (typeof p.cost_m === "number" && Number.isFinite(p.cost_m)) return p.cost_m;
  if (typeof p.now_cost === "number" && Number.isFinite(p.now_cost)) return p.now_cost / 10;
  return null;
}

function playerKey(p: Player): string {
  if (p.fpl_player_id != null) return String(p.fpl_player_id);
  if (p.player_id != null) return String(p.player_id);
  if (p.id != null) return String(p.id);
  return `${getName(p)}|${getTeam(p)}|${p.position ?? ""}`;
}

function computeCaptainVice(starters: Player[]): { captainKey: string | null; viceKey: string | null } {
  const sorted = [...starters].sort((a, b) => {
    const pa = getPredPts(a);
    const pb = getPredPts(b);
    if (pb !== pa) return pb - pa;
    return getName(a).localeCompare(getName(b));
  });
  return {
    captainKey: sorted[0] ? playerKey(sorted[0]) : null,
    viceKey: sorted[1] ? playerKey(sorted[1]) : null,
  };
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center justify-center rounded-md bg-gray-100 px-2 py-0.5 text-xs font-semibold text-gray-700">
      {children}
    </span>
  );
}

function RoleBadge({ role }: { role: "C" | "VC" }) {
  return (
    <span className="inline-flex items-center justify-center rounded-md border border-gray-300 bg-white px-2 py-0.5 text-xs font-bold text-gray-800">
      {role}
    </span>
  );
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-semibold tabular-nums">{value}</span>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="text-xl font-bold">{children}</h2>;
}

function PlayerRow({
  p,
  role,
}: {
  p: Player;
  role?: "C" | "VC";
}) {
  const name = getName(p) || "-";
  const team = getTeam(p);
  const pos = p.position ?? "";
  const costM = getCostM(p);
  const pred = getPredPts(p);

  const value = costM && costM > 0 ? pred / costM : null;

  return (
    <div className="flex items-center justify-between py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium">{name}</span>
          {team ? <span className="text-gray-500">({team})</span> : null}
          {pos ? <Badge>{pos}</Badge> : null}
          {role ? <RoleBadge role={role} /> : null}
        </div>
      </div>

      <div className="flex items-center gap-6 text-sm text-gray-700 tabular-nums">
        <StatPill label="Cost:" value={costM == null ? "—" : `${fmt1(costM)}m`} />
        <StatPill label="Pred:" value={fmt1(pred)} />
        <StatPill label="Value:" value={value == null ? "—" : fmt2(value)} />
      </div>
    </div>
  );
}

export default function SquadPage() {
  const [targetGw, setTargetGw] = useState<number>(28);
  const [budgetM, setBudgetM] = useState<number>(100.0);
  const [maxPerTeam, setMaxPerTeam] = useState<number>(3);
  const [status, setStatus] = useState<Status>("a");
  const [modelName, setModelName] = useState<string>("baseline_rollavg_v0");
  const [view, setView] = useState<ViewMode>("compact");

  const [models, setModels] = useState<ModelOpt[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsErr, setModelsErr] = useState<string | null>(null);

  const [data, setData] = useState<SquadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function loadModels() {
    setModelsLoading(true);
    setModelsErr(null);
    try {
      const res = await fetch("/api/models", { cache: "no-store" });
      const body = await res.json();

      if (!res.ok) {
        const msg = body?.error?.message || body?.message || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      const raw = Array.isArray(body?.models) ? body.models : [];
      const list: ModelOpt[] = raw
        .map((m: any) => ({
          model_name: String(m.model_name ?? "").trim(),
          label: String(m.label ?? m.model_name ?? "").trim(),
        }))
        .filter((m: ModelOpt) => m.model_name.length > 0);

      list.sort((a, b) => (a.label ?? a.model_name).localeCompare(b.label ?? b.model_name));
      setModels(list);

      if (!modelName) {
        const baseline = list.find((x) => x.model_name === "baseline_rollavg_v0");
        if (baseline) setModelName(baseline.model_name);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setModelsErr(msg);
      setModels([]);
    } finally {
      setModelsLoading(false);
    }
  }

  useEffect(() => {
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const queryString = useMemo(() => {
    const sp = new URLSearchParams();
    sp.set("target_gw", String(targetGw));
    sp.set("budget_m", String(budgetM));
    sp.set("max_per_team", String(maxPerTeam));
    if (status) sp.set("status", status);
    if (modelName.trim()) sp.set("model_name", modelName.trim());
    sp.set("view", view);
    return sp.toString();
  }, [targetGw, budgetM, maxPerTeam, status, modelName, view]);

  async function onGenerate() {
    setLoading(true);
    setErr(null);

    try {
      const res = await fetch(`/api/squad?${queryString}`, { cache: "no-store" });
      const body = await res.json();

      if (!res.ok) {
        const msg = body?.error?.message || body?.message || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      setData(body as SquadResponse);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setErr(msg);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const startingXi = useMemo(() => {
    const sx = data?.starting_xi ?? {};
    const byPos = {
      GKP: Array.isArray((sx as any).GKP) ? (sx as any).GKP : [],
      DEF: Array.isArray((sx as any).DEF) ? (sx as any).DEF : [],
      MID: Array.isArray((sx as any).MID) ? (sx as any).MID : [],
      FWD: Array.isArray((sx as any).FWD) ? (sx as any).FWD : [],
    } as Record<"GKP" | "DEF" | "MID" | "FWD", Player[]>;
    return byPos;
  }, [data]);

  const startersFlat = useMemo(() => {
    return [
      ...startingXi.GKP,
      ...startingXi.DEF,
      ...startingXi.MID,
      ...startingXi.FWD,
    ];
  }, [startingXi]);

  const { captainKey, viceKey } = useMemo(() => computeCaptainVice(startersFlat), [startersFlat]);

  const bench = useMemo(() => {
    const bx = data?.bench ?? {};
    const byPos = {
      GKP: Array.isArray((bx as any).GKP) ? (bx as any).GKP : [],
      DEF: Array.isArray((bx as any).DEF) ? (bx as any).DEF : [],
      MID: Array.isArray((bx as any).MID) ? (bx as any).MID : [],
      FWD: Array.isArray((bx as any).FWD) ? (bx as any).FWD : [],
    } as Record<"GKP" | "DEF" | "MID" | "FWD", Player[]>;
    return byPos;
  }, [data]);

  const benchList = useMemo(() => {
    if (Array.isArray(data?.bench_list)) return data!.bench_list!;
    return [];
  }, [data]);

  return (
    <main className="max-w-6xl mx-auto p-8 space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">FPL Squad Recommendation</h1>
        <p className="text-gray-600">
          Generate a squad via <code className="px-1 py-0.5 bg-gray-100 rounded">/api/squad</code> and label{" "}
          <span className="font-semibold">Captain (C)</span> / <span className="font-semibold">Vice-Captain (VC)</span>{" "}
          based on predicted points (Starting XI only).
        </p>
      </header>

      {/* Controls */}
      <section className="rounded-xl border bg-white p-5 shadow-sm space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <label className="space-y-1">
            <div className="text-sm font-medium text-gray-700">target_gw</div>
            <input
              className="w-full rounded-md border px-3 py-2"
              type="number"
              min={1}
              value={targetGw}
              onChange={(e) => setTargetGw(Number(e.target.value))}
            />
          </label>

          <label className="space-y-1">
            <div className="text-sm font-medium text-gray-700">budget_m</div>
            <input
              className="w-full rounded-md border px-3 py-2"
              type="number"
              min={0}
              step={0.1}
              value={budgetM}
              onChange={(e) => setBudgetM(Number(e.target.value))}
            />
          </label>

          <label className="space-y-1">
            <div className="text-sm font-medium text-gray-700">max_per_team</div>
            <input
              className="w-full rounded-md border px-3 py-2"
              type="number"
              min={1}
              max={3}
              value={maxPerTeam}
              onChange={(e) => setMaxPerTeam(Number(e.target.value))}
            />
          </label>

          <label className="space-y-1">
            <div className="text-sm font-medium text-gray-700">status</div>
            <select
              className="w-full rounded-md border px-3 py-2"
              value={status}
              onChange={(e) => setStatus(e.target.value as Status)}
            >
              <option value="">(any)</option>
              <option value="a">a (available)</option>
              <option value="i">i (injured)</option>
              <option value="u">u (unavailable)</option>
              <option value="s">s (suspended)</option>
            </select>
          </label>

          <label className="space-y-1">
            <div className="text-sm font-medium text-gray-700">model_name</div>
            <select
              className="w-full rounded-md border px-3 py-2"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            >
              <option value="">
                {modelsLoading ? "Loading models..." : "(any)"}
              </option>
              {models.map((m) => (
                <option key={m.model_name} value={m.model_name}>
                  {m.label ?? m.model_name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            className="rounded-lg bg-gray-900 px-5 py-2.5 font-semibold text-white disabled:opacity-60"
            onClick={onGenerate}
            disabled={loading}
          >
            {loading ? "Generating..." : "Generate Squad"}
          </button>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-sm text-gray-600">view</span>
            <select
              className="rounded-md border px-3 py-2"
              value={view}
              onChange={(e) => setView(e.target.value as ViewMode)}
            >
              <option value="compact">compact</option>
              <option value="full">full</option>
            </select>
          </div>
        </div>

        {modelsErr ? <div className="text-sm text-yellow-700">{modelsErr}</div> : null}
        {err ? <div className="text-sm text-red-600">{err}</div> : null}
      </section>

      {/* Summary cards */}
      {data?.summary ? (
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-xl border bg-white p-5 shadow-sm space-y-1">
            <div className="text-sm text-gray-500">Spent</div>
            <div className="text-2xl font-bold tabular-nums">
              {fmt1(Number(data.summary.spent_m ?? 0))}m
            </div>
          </div>
          <div className="rounded-xl border bg-white p-5 shadow-sm space-y-1">
            <div className="text-sm text-gray-500">Remaining</div>
            <div className="text-2xl font-bold tabular-nums">
              {fmt1(Number(data.summary.remaining_m ?? 0))}m
            </div>
          </div>
          <div className="rounded-xl border bg-white p-5 shadow-sm space-y-2">
            <div className="text-sm text-gray-500">Team Count</div>
            <div className="flex flex-wrap gap-2">
              {data.summary.team_counts
                ? Object.entries(data.summary.team_counts).map(([k, v]) => (
                    <Badge key={k}>
                      {k}: {v}
                    </Badge>
                  ))
                : <div className="text-sm text-gray-500">—</div>}
            </div>
          </div>
        </section>
      ) : null}

      {/* Starting XI */}
      <section className="rounded-xl border bg-white p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <SectionTitle>Starting XI</SectionTitle>
          {startersFlat.length > 0 ? (
            <div className="text-sm text-gray-600 flex items-center gap-2">
              <span>Captain/Vice:</span>
              {captainKey ? <Badge>C</Badge> : <Badge>—</Badge>}
              {viceKey ? <Badge>VC</Badge> : <Badge>—</Badge>}
            </div>
          ) : null}
        </div>

        {startersFlat.length === 0 ? (
          <div className="text-sm text-gray-500">No squad yet. Click Generate Squad.</div>
        ) : (
          <div className="divide-y">
            {(["GKP", "DEF", "MID", "FWD"] as const).map((pos) => {
              const list = startingXi[pos] ?? [];
              if (list.length === 0) return null;
              return (
                <div key={pos} className="py-2">
                  <div className="flex items-center gap-2 pb-2">
                    <Badge>{pos}</Badge>
                    <span className="text-sm text-gray-500">{list.length} players</span>
                  </div>
                  <div className="divide-y">
                    {list.map((p) => {
                      const k = playerKey(p);
                      const role = k === captainKey ? "C" : k === viceKey ? "VC" : undefined;
                      return <PlayerRow key={k} p={p} role={role} />;
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {startersFlat.length > 0 ? (
          <div className="pt-2 text-xs text-gray-500">
            Captain (C) = highest predicted points in Starting XI. Vice-Captain (VC) = second highest.
          </div>
        ) : null}
      </section>

      {/* Bench */}
      <section className="rounded-xl border bg-white p-5 shadow-sm space-y-3">
        <SectionTitle>Bench</SectionTitle>

        {benchList.length > 0 ? (
          <>
            <div className="text-sm text-gray-600">
              Bench order (from backend <code className="px-1 py-0.5 bg-gray-100 rounded">bench_list</code>):
            </div>
            <div className="divide-y">
              {benchList.map((p, idx) => {
                const k = `${playerKey(p)}|bench|${idx}`;
                return (
                  <div key={k} className="flex items-center justify-between py-3">
                    <div className="flex items-center gap-2">
                      <Badge>{idx + 1}</Badge>
                      <span className="font-medium">{getName(p) || "-"}</span>
                      {getTeam(p) ? <span className="text-gray-500">({getTeam(p)})</span> : null}
                      {p.position ? <Badge>{p.position}</Badge> : null}
                    </div>
                    <div className="flex items-center gap-6 text-sm text-gray-700 tabular-nums">
                      <StatPill label="Cost:" value={getCostM(p) == null ? "—" : `${fmt1(getCostM(p) as number)}m`} />
                      <StatPill label="Pred:" value={fmt1(getPredPts(p))} />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="divide-y">
            {(["GKP", "DEF", "MID", "FWD"] as const).map((pos) => {
              const list = bench[pos] ?? [];
              if (list.length === 0) return null;
              return (
                <div key={pos} className="py-2">
                  <div className="flex items-center gap-2 pb-2">
                    <Badge>{pos}</Badge>
                    <span className="text-sm text-gray-500">{list.length} players</span>
                  </div>
                  <div className="divide-y">
                    {list.map((p) => (
                      <PlayerRow key={playerKey(p)} p={p} />
                    ))}
                  </div>
                </div>
              );
            })}
            <div className="pt-2 text-xs text-gray-500">
              Bench order not available (backend did not return bench_list).
            </div>
          </div>
        )}
      </section>
    </main>
  );
}