// src/app/page.tsx

"use client";

import { useMemo, useState } from "react";
import { fetchSquad } from "@/lib/api";
import type { CompactPlayer, Position, SquadResponse } from "@/lib/types";

type FormState = {
  target_gw: string; // allow empty
  budget_m: string;
  max_per_team: string;
  status: "a" | "all";
  model_name: string;
};

const POS_ORDER: Position[] = ["GKP", "DEF", "MID", "FWD"];

function fmtMoney(m: number) {
  return `${m.toFixed(1)}m`;
}

function PlayerRow({ p }: { p: CompactPlayer }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 py-2 border-b border-slate-200">
      <div className="font-medium">
        {p.name} <span className="text-slate-500">({p.team})</span>
        <span className="ml-2 text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700">
          {p.position}
        </span>
        {p.role && (
          <span className="ml-2 text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700">
            {p.role} #{p.slot ?? ""}
          </span>
        )}
      </div>

      <div className="text-sm text-slate-700 flex gap-4">
        <span>Cost: {fmtMoney(p.cost_m)}</span>
        <span>Pred: {p.predicted_points.toFixed(1)}</span>
        <span>Value: {p.value.toFixed(2)}</span>
      </div>
    </div>
  );
}

function PositionBlock({
  title,
  players,
}: {
  title: string;
  players: CompactPlayer[];
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">{title}</h3>
        <span className="text-sm text-slate-500">{players.length}</span>
      </div>

      <div className="mt-2">
        {players.length === 0 ? (
          <div className="text-sm text-slate-500">No players.</div>
        ) : (
          players.map((p) => <PlayerRow key={`${p.player_id}-${p.position}`} p={p} />)
        )}
      </div>
    </div>
  );
}

export default function Page() {
  const [form, setForm] = useState<FormState>({
    target_gw: "23",
    budget_m: "100.0",
    max_per_team: "3",
    status: "a",
    model_name: "baseline_rollavg_v0",
  });

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<SquadResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);

  const requestParams = useMemo(() => {
    const target_gw = form.target_gw.trim() ? Number(form.target_gw) : undefined;
    const budget_m = form.budget_m.trim() ? Number(form.budget_m) : 100.0;
    const max_per_team = form.max_per_team.trim() ? Number(form.max_per_team) : 3;

    return {
      target_gw,
      budget_m,
      max_per_team,
      status: form.status,
      model_name: form.model_name,
      view: "compact" as const,
      // Keep simple for Day14: fixed order_by
      order_by: "value" as const,
    };
  }, [form]);

  async function onGenerate() {
    setLoading(true);
    setErrorText(null);

    try {
      const res = await fetchSquad(requestParams);
      setData(res);

      if (res.error) {
        setErrorText(res.error);
      }
    } catch (e: any) {
      setErrorText(String(e?.message ?? e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const starting = data?.starting_xi;
  const benchList = data?.bench_list ?? [];
  const squadList = data?.squad_list ?? [];

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="max-w-5xl mx-auto px-4 py-10">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold">FPL Squad Recommendation</h1>
          <p className="text-slate-600">
            Minimal Day14 UI: calls <code className="px-1 py-0.5 bg-slate-100 rounded">/recommendations/squad</code>{" "}
            and displays a compact squad.
          </p>
        </header>

        {/* Controls */}
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700">target_gw</label>
              <input
                value={form.target_gw}
                onChange={(e) => setForm((s) => ({ ...s, target_gw: e.target.value }))}
                placeholder="e.g. 23"
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700">budget_m</label>
              <input
                value={form.budget_m}
                onChange={(e) => setForm((s) => ({ ...s, budget_m: e.target.value }))}
                placeholder="100.0"
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700">max_per_team</label>
              <input
                value={form.max_per_team}
                onChange={(e) => setForm((s) => ({ ...s, max_per_team: e.target.value }))}
                placeholder="3"
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700">status</label>
              <select
                value={form.status}
                onChange={(e) => setForm((s) => ({ ...s, status: e.target.value as "a" | "all" }))}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              >
                <option value="a">a (available)</option>
                <option value="all">all</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700">model_name</label>
              <input
                value={form.model_name}
                onChange={(e) => setForm((s) => ({ ...s, model_name: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={onGenerate}
              disabled={loading}
              className="rounded-xl bg-slate-900 text-white px-4 py-2 text-sm font-semibold disabled:opacity-60"
            >
              {loading ? "Generating..." : "Generate Squad"}
            </button>
            <div className="text-sm text-slate-600">
              Backend: <code className="px-1 py-0.5 bg-slate-100 rounded">http://127.0.0.1:8000</code>
            </div>
          </div>

          {errorText && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              <div className="font-semibold">Error</div>
              <div>{errorText}</div>
              {data?.diagnostics && (
                <pre className="mt-2 whitespace-pre-wrap text-xs text-red-700/80">
                  {JSON.stringify(data.diagnostics, null, 2)}
                </pre>
              )}
            </div>
          )}
        </section>

        {/* Summary */}
        {data && !data.error && (
          <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="text-sm text-slate-500">Spent</div>
              <div className="text-2xl font-bold">{fmtMoney(data.summary.spent_m)}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="text-sm text-slate-500">Remaining</div>
              <div className="text-2xl font-bold">{fmtMoney(data.summary.remaining_m)}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="text-sm text-slate-500">Team Count</div>
              <div className="mt-2 text-sm text-slate-700">
                {Object.keys(data.summary.team_counts).length === 0 ? (
                  <span className="text-slate-500">—</span>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(data.summary.team_counts).map(([teamId, n]) => (
                      <span key={teamId} className="px-2 py-1 rounded bg-slate-100">
                        team {teamId}: {n}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Starting XI */}
        {starting && !data?.error && (
          <section className="mt-6">
            <h2 className="text-2xl font-bold mb-3">Starting XI</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {POS_ORDER.map((pos) => (
                <PositionBlock key={pos} title={pos} players={starting[pos] ?? []} />
              ))}
            </div>
          </section>
        )}

        {/* Bench List */}
        {data && !data.error && (
          <section className="mt-6">
            <h2 className="text-2xl font-bold mb-3">Bench (fixed order)</h2>
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              {benchList.length === 0 ? (
                <div className="text-sm text-slate-500">No bench.</div>
              ) : (
                benchList.map((p) => (
                  <PlayerRow key={`${p.player_id}-${p.position}-bench`} p={p} />
                ))
              )}
            </div>
          </section>
        )}

        {/* Optional flat squad list */}
        {squadList.length > 0 && !data?.error && (
          <section className="mt-6">
            <h2 className="text-2xl font-bold mb-3">Squad List (15 flat)</h2>
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              {squadList.map((p, idx) => (
                <PlayerRow key={`${p.player_id}-${p.position}-${idx}`} p={p} />
              ))}
            </div>
          </section>
        )}

        <footer className="mt-10 text-sm text-slate-500">
          Day14 stop point: UI loads, calls backend through proxy, shows starting XI + bench_list + summary.
        </footer>
      </div>
    </main>
  );
}
