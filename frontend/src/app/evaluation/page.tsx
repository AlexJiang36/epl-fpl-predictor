"use client";

import { useEffect, useMemo, useState } from "react";

type MetricsSummary = Record<string, number>;

type EvaluationModelRow = {
  model_name: string;
  task_type: "player_points" | "match_result" | "match_goals" | string;
  feature_version: string | null;
  status: "active" | "experimental" | "archived" | string;
  is_active: boolean;
  is_production_default: boolean;
  selected_reason: string | null;
  notes: string | null;
  training_window_start_gw: number | null;
  training_window_end_gw: number | null;
  evaluation_start_gw: number | null;
  evaluation_end_gw: number | null;
  metrics_summary: MetricsSummary;
  updated_at: string;
};

type EvaluationSummaryResponse = {
  production_defaults: {
    player_default_model: string | null;
    match_default_model: string | null;
    player_backup_model: string | null;
    experimental_goals_model: string | null;
  };
  player_models: EvaluationModelRow[];
  match_result_models: EvaluationModelRow[];
  match_goals_models: EvaluationModelRow[];
  meta: {
    active_only: boolean;
    player_model_count: number;
    match_result_model_count: number;
    match_goals_model_count: number;
    source: string;
  };
};

function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function metricFromKeys(
  metrics: MetricsSummary,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    if (metrics[key] !== undefined) return metrics[key];
  }
  return undefined;
}

function statusPillClasses(status: string): string {
  if (status === "active") {
    return "bg-green-100 text-green-800 border-green-200";
  }
  if (status === "experimental") {
    return "bg-amber-100 text-amber-800 border-amber-200";
  }
  return "bg-slate-100 text-slate-700 border-slate-200";
}

function barWidth(value: number, maxValue: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0) {
    return "0%";
  }
  return `${Math.max(8, Math.round((value / maxValue) * 100))}%`;
}

function inverseBarWidth(value: number, minValue: number, maxValue: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return "0%";
  }
  if (maxValue <= minValue) return "100%";
  const scaled = (maxValue - value) / (maxValue - minValue);
  return `${Math.max(8, Math.round(scaled * 100))}%`;
}

function SummaryCard({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string | null;
  subtext?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="text-sm font-medium text-slate-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-slate-900">
        {value ?? "—"}
      </div>
      {subtext ? <div className="mt-2 text-sm text-slate-500">{subtext}</div> : null}
    </div>
  );
}

function PlayerMaeChart({ rows }: { rows: EvaluationModelRow[] }) {
  const chartRows = useMemo(() => {
    const prepared = rows
      .map((row) => ({
        model_name: row.model_name,
        status: row.status,
        is_active: row.is_active,
        is_production_default: row.is_production_default,
        mae: metricFromKeys(row.metrics_summary, [
          "val_mae",
          "overall_mae",
          "validation_mae",
        ]),
      }))
      .filter((row) => row.mae !== undefined) as Array<{
      model_name: string;
      status: string;
      is_active: boolean;
      is_production_default: boolean;
      mae: number;
    }>;

    return prepared.sort((a, b) => a.mae - b.mae);
  }, [rows]);

  const maes = chartRows.map((row) => row.mae);
  const minMae = maes.length ? Math.min(...maes) : 0;
  const maxMae = maes.length ? Math.max(...maes) : 1;

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">Player Model MAE</h2>
          <p className="mt-1 text-sm text-slate-500">
            Lower is better. Current production default should appear near the top.
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {chartRows.map((row) => (
          <div key={row.model_name}>
            <div className="mb-1 flex items-center justify-between gap-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-slate-900">{row.model_name}</span>
                {row.is_production_default ? (
                  <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
                    default
                  </span>
                ) : null}
                {row.is_active ? (
                  <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
                    active
                  </span>
                ) : null}
              </div>
              <span className="font-semibold text-slate-900">{formatMetric(row.mae)}</span>
            </div>

            <div className="h-3 rounded-full bg-slate-100">
              <div
                className="h-3 rounded-full bg-indigo-500"
                style={{ width: inverseBarWidth(row.mae, minMae, maxMae) }}
              />
            </div>
          </div>
        ))}

        {chartRows.length === 0 ? (
          <div className="text-sm text-slate-500">No MAE data available.</div>
        ) : null}
      </div>
    </section>
  );
}

function MatchMetricChart({
  rows,
  metricLabel,
  metricKeys,
  lowerIsBetter = false,
}: {
  rows: EvaluationModelRow[];
  metricLabel: string;
  metricKeys: string[];
  lowerIsBetter?: boolean;
}) {
  const chartRows = useMemo(() => {
    const prepared = rows
      .map((row) => ({
        model_name: row.model_name,
        status: row.status,
        is_active: row.is_active,
        is_production_default: row.is_production_default,
        value: metricFromKeys(row.metrics_summary, metricKeys),
      }))
      .filter((row) => row.value !== undefined) as Array<{
      model_name: string;
      status: string;
      is_active: boolean;
      is_production_default: boolean;
      value: number;
    }>;

    return prepared.sort((a, b) =>
      lowerIsBetter ? a.value - b.value : b.value - a.value,
    );
  }, [rows, metricKeys, lowerIsBetter]);

  const values = chartRows.map((row) => row.value);
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-xl font-semibold text-slate-900">{metricLabel}</h2>
        <p className="mt-1 text-sm text-slate-500">
          {lowerIsBetter ? "Lower is better." : "Higher is better."}
        </p>
      </div>

      <div className="space-y-4">
        {chartRows.map((row) => (
          <div key={row.model_name}>
            <div className="mb-1 flex items-center justify-between gap-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-slate-900">{row.model_name}</span>
                {row.is_production_default ? (
                  <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
                    default
                  </span>
                ) : null}
                {row.is_active ? (
                  <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
                    active
                  </span>
                ) : null}
              </div>
              <span className="font-semibold text-slate-900">{formatMetric(row.value)}</span>
            </div>

            <div className="h-3 rounded-full bg-slate-100">
              <div
                className={`h-3 rounded-full ${lowerIsBetter ? "bg-violet-500" : "bg-emerald-500"}`}
                style={{
                  width: lowerIsBetter
                    ? inverseBarWidth(row.value, minValue, maxValue)
                    : barWidth(row.value, maxValue),
                }}
              />
            </div>
          </div>
        ))}

        {chartRows.length === 0 ? (
          <div className="text-sm text-slate-500">No {metricLabel.toLowerCase()} data available.</div>
        ) : null}
      </div>
    </section>
  );
}

function ModelTable({
  title,
  rows,
  kind,
}: {
  title: string;
  rows: EvaluationModelRow[];
  kind: "player" | "match_result" | "match_goals";
}) {
  const columns = useMemo(() => {
    if (kind === "player") {
      return ["Model", "Status", "Feature", "MAE", "Eval GW", "Notes"];
    }
    if (kind === "match_result") {
      return ["Model", "Status", "Feature", "Accuracy", "Log Loss", "Eval GW"];
    }
    return ["Model", "Status", "Feature", "Goal MAE", "Goal RMSE", "Eval GW"];
  }, [kind]);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
        <div className="text-sm text-slate-500">{rows.length} models</div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-0">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="border-b border-slate-200 px-3 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-500"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const mae = metricFromKeys(row.metrics_summary, [
                "val_mae",
                "overall_mae",
                "validation_mae",
              ]);
              const accuracy = metricFromKeys(row.metrics_summary, [
                "val_accuracy",
                "accuracy",
              ]);
              const logloss = metricFromKeys(row.metrics_summary, [
                "val_logloss",
                "logloss",
              ]);
              const goalMae = metricFromKeys(row.metrics_summary, [
                "avg_goal_mae",
                "home_goals_mae",
              ]);
              const goalRmse = metricFromKeys(row.metrics_summary, [
                "avg_goal_rmse",
                "home_goals_rmse",
              ]);

              const evalGw =
                row.evaluation_start_gw && row.evaluation_end_gw
                  ? `${row.evaluation_start_gw}–${row.evaluation_end_gw}`
                  : "—";

              return (
                <tr key={row.model_name} className="align-top">
                  <td className="border-b border-slate-100 px-3 py-4">
                    <div className="flex flex-col gap-1">
                      <div className="font-medium text-slate-900">
                        {row.model_name}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {row.is_production_default ? (
                          <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
                            default
                          </span>
                        ) : null}
                        {row.is_active ? (
                          <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
                            active
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </td>

                  <td className="border-b border-slate-100 px-3 py-4">
                    <span
                      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${statusPillClasses(
                        row.status,
                      )}`}
                    >
                      {row.status}
                    </span>
                  </td>

                  <td className="border-b border-slate-100 px-3 py-4 text-sm text-slate-700">
                    {row.feature_version ?? "—"}
                  </td>

                  {kind === "player" ? (
                    <>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm font-medium text-slate-900">
                        {formatMetric(mae)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm text-slate-700">
                        {evalGw}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm text-slate-600">
                        {row.notes ?? row.selected_reason ?? "—"}
                      </td>
                    </>
                  ) : null}

                  {kind === "match_result" ? (
                    <>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm font-medium text-slate-900">
                        {formatMetric(accuracy)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm font-medium text-slate-900">
                        {formatMetric(logloss)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm text-slate-700">
                        {evalGw}
                      </td>
                    </>
                  ) : null}

                  {kind === "match_goals" ? (
                    <>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm font-medium text-slate-900">
                        {formatMetric(goalMae)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm font-medium text-slate-900">
                        {formatMetric(goalRmse)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-4 text-sm text-slate-700">
                        {evalGw}
                      </td>
                    </>
                  ) : null}
                </tr>
              );
            })}

            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-8 text-center text-sm text-slate-500"
                >
                  No rows found.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeOnly, setActiveOnly] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const apiBase =
          process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
        const url = `${apiBase}/evaluation/summary?active_only=${activeOnly}`;

        const response = await fetch(url, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(`Failed to load evaluation summary (${response.status})`);
        }

        const json: EvaluationSummaryResponse = await response.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [activeOnly]);

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
              Historical Evaluation
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
              Model Performance Dashboard
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              A lightweight, screenshot-ready view of player and match model
              performance, production defaults, and current model status.
            </p>
          </div>

          <label className="inline-flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm">
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={(e) => setActiveOnly(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300"
            />
            Show active models only
          </label>
        </div>

        {loading ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-slate-500 shadow-sm">
            Loading evaluation summary...
          </div>
        ) : null}

        {error ? (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-rose-700 shadow-sm">
            <div className="font-semibold">Could not load evaluation data</div>
            <div className="mt-2 text-sm">{error}</div>
            <div className="mt-2 text-sm">
              Check that the backend evaluation route is running and that
              <code className="mx-1 rounded bg-white px-1 py-0.5">
                NEXT_PUBLIC_API_BASE_URL
              </code>
              points to the correct API base if needed.
            </div>
          </div>
        ) : null}

        {!loading && !error && data ? (
          <div className="space-y-8">
            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <SummaryCard
                label="Player Default"
                value={data.production_defaults.player_default_model}
                subtext="Main player production model"
              />
              <SummaryCard
                label="Player Backup"
                value={data.production_defaults.player_backup_model}
                subtext="Explainable linear backup"
              />
              <SummaryCard
                label="Match Default"
                value={data.production_defaults.match_default_model}
                subtext="Main match production model"
              />
              <SummaryCard
                label="Experimental Goals"
                value={data.production_defaults.experimental_goals_model}
                subtext="Goals / scoreline foundation"
              />
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <SummaryCard
                label="Player Models"
                value={String(data.meta.player_model_count)}
              />
              <SummaryCard
                label="Match Result Models"
                value={String(data.meta.match_result_model_count)}
              />
              <SummaryCard
                label="Match Goals Models"
                value={String(data.meta.match_goals_model_count)}
              />
            </section>

            <section className="grid gap-6 xl:grid-cols-2">
              <PlayerMaeChart rows={data.player_models} />
              <div className="space-y-6">
                <MatchMetricChart
                  rows={data.match_result_models}
                  metricLabel="Match Model Accuracy"
                  metricKeys={["val_accuracy", "accuracy"]}
                />
                <MatchMetricChart
                  rows={data.match_result_models}
                  metricLabel="Match Model Log Loss"
                  metricKeys={["val_logloss", "logloss"]}
                  lowerIsBetter
                />
              </div>
            </section>

            <ModelTable
              title="Player Model Comparison"
              rows={data.player_models}
              kind="player"
            />

            <ModelTable
              title="Match Result Model Comparison"
              rows={data.match_result_models}
              kind="match_result"
            />

            <ModelTable
              title="Goals Prototype Comparison"
              rows={data.match_goals_models}
              kind="match_goals"
            />
          </div>
        ) : null}
      </div>
    </main>
  );
}
