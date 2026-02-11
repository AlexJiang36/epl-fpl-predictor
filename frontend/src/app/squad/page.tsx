"use client";

import React, { useMemo, useState } from "react";

type SquadPlayer = {
  name: string;
  position: "GKP" | "DEF" | "MID" | "FWD";
  team: string;
  cost_m: number;
  predicted_points: number;
  value?: number;
  player_id?: number;
  fpl_player_id?: number;
  team_id?: number;
};

type SquadResponse = {
  target_gw: number;
  model_name?: string;
  generated_at?: string;
  filters?: any;
  summary?: {
    spent_m: number;
    remaining_m: number;
    team_counts: Record<string, number>;
    squad_counts: Record<string, number>;
  };
  starting_xi?: Record<string, SquadPlayer[]>;
  bench?: Record<string, SquadPlayer[]>;
  bench_list?: SquadPlayer[];
};

type OrderBy = "predicted_points" | "value" | "cost_m";
type ViewMode = "compact" | "full";

function fmt1(n: number) {
  return Math.round(n * 10) / 10;
}

function PositionSection({
  title,
  groups,
}: {
  title: string;
  groups?: Record<string, SquadPlayer[]>;
}) {
  const order: Array<SquadPlayer["position"]> = ["GKP", "DEF", "MID", "FWD"];
  return (
    <div className="border rounded-lg p-4 space-y-3">
      <h2 className="font-semibold text-lg">{title}</h2>

      {!groups ? (
        <div className="text-sm text-gray-500">No data</div>
      ) : (
        order.map((pos) => {
          const list = groups[pos] || [];
          if (list.length === 0) return null;
          return (
            <div key={pos} className="space-y-2">
              <div className="font-medium">{pos}</div>
              <div className="grid gap-2">
                {list.map((p, idx) => (
                  <div
                    key={`${p.name}-${idx}`}
                    className="flex items-center justify-between rounded-md border p-2"
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">
                        {p.name} <span className="text-gray-500">({p.team})</span>
                      </div>
                      <div className="text-xs text-gray-500">
                        Cost: {fmt1(p.cost_m)} · Pred: {fmt1(p.predicted_points)}
                        {typeof p.value === "number" ? ` · Value: ${fmt1(p.value)}` : ""}
                      </div>
                    </div>
                    <div className="text-sm text-gray-700">
                      {fmt1(p.predicted_points)} pts
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

export default function SquadPage() {
  // Defaults: keep them simple and demo-friendly
  const [targetGw, setTargetGw] = useState<number>(26);
  const [budgetM, setBudgetM] = useState<number>(100);
  const [maxPerTeam, setMaxPerTeam] = useState<number>(3);
  const [orderBy, setOrderBy] = useState<OrderBy>("predicted_points");
  const [view, setView] = useState<ViewMode>("compact");

  const [data, setData] = useState<SquadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const queryString = useMemo(() => {
    const sp = new URLSearchParams();
    sp.set("target_gw", String(targetGw));
    sp.set("budget_m", String(budgetM));
    sp.set("max_per_team", String(maxPerTeam));
    sp.set("order_by", orderBy);
    sp.set("view", view);
    return sp.toString();
  }, [targetGw, budgetM, maxPerTeam, orderBy, view]);

  async function onGenerate() {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`/api/squad?${queryString}`, { cache: "no-store" });
      const body = await res.json();

      if (!res.ok) {
        // BFF standardized error: { ok:false, error:{message,status} }
        const msg =
          body?.error?.message ||
          body?.message ||
          `Request failed (${res.status})`;
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

  return (
    <main className="max-w-5xl mx-auto p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Squad Generator</h1>
        <p className="text-sm text-gray-600">
          Day16: form → call <code className="px-1 py-0.5 border rounded">/api/squad</code> → render
          Starting XI / Bench / Summary.
        </p>
      </header>

      {/* Form */}
      <section className="border rounded-lg p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
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
            <span className="text-sm font-medium">budget_m</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              min={0}
              step={0.1}
              value={budgetM}
              onChange={(e) => setBudgetM(Number(e.target.value))}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">max_per_team</span>
            <input
              className="border rounded px-2 py-1"
              type="number"
              min={1}
              max={3}
              value={maxPerTeam}
              onChange={(e) => setMaxPerTeam(Number(e.target.value))}
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">order_by</span>
            <select
              className="border rounded px-2 py-1"
              value={orderBy}
              onChange={(e) => setOrderBy(e.target.value as OrderBy)}
            >
              <option value="predicted_points">predicted_points</option>
              <option value="value">value</option>
              <option value="cost_m">cost_m</option>
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium">view</span>
            <select
              className="border rounded px-2 py-1"
              value={view}
              onChange={(e) => setView(e.target.value as ViewMode)}
            >
              <option value="compact">compact</option>
              <option value="full">full</option>
            </select>
          </label>
        </div>

        <div className="flex items-center gap-3">
          <button
            className="border rounded px-4 py-2 font-medium disabled:opacity-60"
            onClick={onGenerate}
            disabled={loading || !Number.isFinite(targetGw) || targetGw <= 0}
          >
            {loading ? "Generating..." : "Generate Squad"}
          </button>

          <div className="text-xs text-gray-500 break-all">
            Request: <code>/api/squad?{queryString}</code>
          </div>
        </div>

        {err ? (
          <div className="border rounded-md p-3 bg-red-50 text-red-800 text-sm">
            <div className="font-semibold">Error</div>
            <div className="break-words">{err}</div>
          </div>
        ) : null}
      </section>

      {/* Summary */}
      <section className="border rounded-lg p-4 space-y-2">
        <h2 className="font-semibold text-lg">Summary</h2>
        {!data?.summary ? (
          <div className="text-sm text-gray-500">No summary yet. Click “Generate Squad”.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="border rounded p-3">
              <div className="text-gray-500">Spent</div>
              <div className="text-xl font-semibold">{fmt1(data.summary.spent_m)}m</div>
            </div>
            <div className="border rounded p-3">
              <div className="text-gray-500">Remaining</div>
              <div className="text-xl font-semibold">{fmt1(data.summary.remaining_m)}m</div>
            </div>
            <div className="border rounded p-3">
              <div className="text-gray-500">Model</div>
              <div className="font-medium">{data.model_name ?? "unknown"}</div>
              <div className="text-xs text-gray-500 break-all">
                GW {data.target_gw} · {data.generated_at ?? ""}
              </div>
            </div>

            <div className="border rounded p-3 md:col-span-3">
              <div className="text-gray-500 mb-1">Team counts</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(data.summary.team_counts || {}).map(([teamId, cnt]) => (
                  <span key={teamId} className="border rounded px-2 py-1 text-xs">
                    team {teamId}: {cnt}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Starting XI & Bench */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PositionSection title="Starting XI" groups={data?.starting_xi} />
        <PositionSection title="Bench" groups={data?.bench} />
      </section>

      {/* Bench order */}
      <section className="border rounded-lg p-4 space-y-3">
        <h2 className="font-semibold text-lg">Bench Order</h2>
        {!data?.bench_list?.length ? (
          <div className="text-sm text-gray-500">No bench_list in response.</div>
        ) : (
          <ol className="list-decimal pl-5 space-y-2">
            {data.bench_list.map((p, idx) => (
              <li key={`${p.name}-${idx}`} className="text-sm">
                <span className="font-medium">{p.name}</span>{" "}
                <span className="text-gray-500">({p.position}, {p.team})</span>{" "}
                — cost {fmt1(p.cost_m)} · pred {fmt1(p.predicted_points)}
              </li>
            ))}
          </ol>
        )}
      </section>
    </main>
  );
}