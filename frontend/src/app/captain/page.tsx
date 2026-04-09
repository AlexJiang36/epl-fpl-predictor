"use client";

import { useState } from "react";

type MinutesStability = {
  label: string;
  avg_minutes: number | null;
  mins_60_plus_count: number;
  sample_size: number;
};

type FutureFactors = {
  fixture_difficulty: number | null;
  opponent_defense_strength: number | null;
  home_away: string | null;
  fixture_count: number | null;
  match_model_signal: number | null;
};

type CaptainCandidate = {
  player_id: number;
  web_name: string;
  team_name: string;
  team_short_name: string;
  position: string;
  now_cost: number;
  predicted_points: number;
  captain_label: string;
  recent_form_summary: string | null;
  minutes_stability: MinutesStability | null;
  explanation: string;
  future_factors: FutureFactors;
};

type CaptainResponse = {
  target_gw: number;
  model_name: string;
  captain: CaptainCandidate | null;
  vice_captain: CaptainCandidate | null;
  top_candidates: CaptainCandidate[];
  error?: string;
};

function formatCost(nowCost: number): string {
  return (nowCost / 10).toFixed(1);
}

function labelClass(label: string): string {
  if (label === "safe") {
    return "bg-green-100 text-green-800";
  }
  if (label === "upside") {
    return "bg-orange-100 text-orange-800";
  }
  return "bg-slate-100 text-slate-800";
}

function CandidateCard({
  title,
  candidate,
}: {
  title: string;
  candidate: CaptainCandidate | null;
}) {
  if (!candidate) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-sm font-medium text-slate-500">{title}</div>
        <div className="mt-3 text-slate-400">No recommendation available.</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium text-slate-500">{title}</div>
          <div className="mt-2 text-2xl font-semibold text-slate-900">
            {candidate.web_name}
          </div>
          <div className="mt-1 text-sm text-slate-600">
            {candidate.team_short_name} · {candidate.position} · £
            {formatCost(candidate.now_cost)}m
          </div>
        </div>

        <span
          className={`rounded-full px-3 py-1 text-xs font-medium ${labelClass(
            candidate.captain_label
          )}`}
        >
          {candidate.captain_label}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-slate-50 p-3">
          <div className="text-xs text-slate-500">Predicted Points</div>
          <div className="mt-1 text-lg font-semibold text-slate-900">
            {candidate.predicted_points.toFixed(2)}
          </div>
        </div>

        <div className="rounded-xl bg-slate-50 p-3">
          <div className="text-xs text-slate-500">Minutes Stability</div>
          <div className="mt-1 text-lg font-semibold text-slate-900">
            {candidate.minutes_stability?.label ?? "unknown"}
          </div>
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-700">
        <div className="font-medium text-slate-900">Recent Form</div>
        <div className="mt-1">
          {candidate.recent_form_summary ?? "No recent form summary."}
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-700">
        <div className="font-medium text-slate-900">Explanation</div>
        <div className="mt-1">{candidate.explanation}</div>
      </div>
    </div>
  );
}

export default function CaptainPage() {
  const [targetGw, setTargetGw] = useState("32");
  const [modelName, setModelName] = useState("baseline_rollavg_v0");
  const [data, setData] = useState<CaptainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchCaptainRecommendations() {
    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        target_gw: targetGw,
        model_name: modelName,
        limit: "5",
      });

      const res = await fetch(
        `/api/recommendations/captain?${params.toString()}`,
        { cache: "no-store" }
      );

      const json = await res.json();

      if (!res.ok) {
        throw new Error(json?.error || "Failed to fetch recommendations");
      }

      setData(json);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold text-slate-900">
            Captain Recommendations
          </h1>
          <p className="mt-2 text-slate-600">
            Simple, explainable captain and vice-captain suggestions based on
            existing player predictions.
          </p>
        </div>

        <div className="mb-8 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Target GW
              </label>
              <input
                type="number"
                value={targetGw}
                onChange={(e) => setTargetGw(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 outline-none focus:border-slate-500"
              />
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Model Name
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 outline-none focus:border-slate-500"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={fetchCaptainRecommendations}
                disabled={loading}
                className="w-full rounded-xl bg-slate-900 px-4 py-2.5 text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Loading..." : "Get Recommendations"}
              </button>
            </div>
          </div>

          {error ? (
            <div className="mt-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          ) : null}
        </div>

        {data ? (
          <>
            <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
              <CandidateCard title="Captain" candidate={data.captain} />
              <CandidateCard title="Vice-Captain" candidate={data.vice_captain} />
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">
                    Top Captain Candidates
                  </h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Ranked by predicted points for GW{data.target_gw}.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-sm text-slate-500">
                      <th className="py-3 pr-4">Rank</th>
                      <th className="py-3 pr-4">Player</th>
                      <th className="py-3 pr-4">Team</th>
                      <th className="py-3 pr-4">Pos</th>
                      <th className="py-3 pr-4">Predicted</th>
                      <th className="py-3 pr-4">Form</th>
                      <th className="py-3 pr-4">Minutes</th>
                      <th className="py-3 pr-4">Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_candidates.map((player, idx) => (
                      <tr
                        key={player.player_id}
                        className="border-b border-slate-100 text-sm text-slate-700"
                      >
                        <td className="py-3 pr-4 font-medium text-slate-900">
                          {idx + 1}
                        </td>
                        <td className="py-3 pr-4">{player.web_name}</td>
                        <td className="py-3 pr-4">{player.team_short_name}</td>
                        <td className="py-3 pr-4">{player.position}</td>
                        <td className="py-3 pr-4">
                          {player.predicted_points.toFixed(2)}
                        </td>
                        <td className="py-3 pr-4">
                          {player.recent_form_summary ?? "N/A"}
                        </td>
                        <td className="py-3 pr-4">
                          {player.minutes_stability?.label ?? "unknown"}
                        </td>
                        <td className="py-3 pr-4">
                          <span
                            className={`rounded-full px-2.5 py-1 text-xs font-medium ${labelClass(
                              player.captain_label
                            )}`}
                          >
                            {player.captain_label}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </main>
  );
}
