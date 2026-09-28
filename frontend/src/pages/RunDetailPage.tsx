import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useRun } from "../api/hooks";
import { StatCard } from "../components/StatCard";
import { fmtCurrency, fmtDateTime, fmtPercent } from "../lib/format";
import type { SegmentMetric } from "../api/types";

function groupBySegment(rows: SegmentMetric[]): Record<string, SegmentMetric[]> {
  const out: Record<string, SegmentMetric[]> = {};
  for (const r of rows) (out[r.segment] ??= []).push(r);
  return out;
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const { data: run, isLoading, error } = useRun(runId);

  if (isLoading) return <p className="text-sm text-slate-500">Loading…</p>;
  if (error) return <p className="text-sm text-red-600">{(error as Error).message}</p>;
  if (!run) return null;

  const defaultPortfolio = run.portfolio_runs.find((p) => p.policy_name === "default") ?? run.portfolio_runs[0];
  const segmentGroups = groupBySegment(run.segment_metrics);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">{run.run_id}</h1>
        <p className="text-sm text-slate-500">
          Started {fmtDateTime(run.started_at)} · {run.wall_time_seconds.toFixed(1)}s ·{" "}
          <Link className="underline" to={`/runs/${run.run_id}/recommendations`}>
            View recommendations →
          </Link>
        </p>
      </div>

      {defaultPortfolio && (
        <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="Increases approved" value={String(defaultPortfolio.n_increase_applied)} />
          <StatCard label="Decreases" value={String(defaultPortfolio.n_decrease)} />
          <StatCard label="Total EP uplift" value={fmtCurrency(defaultPortfolio.total_ep_uplift)} />
          <StatCard
            label="EAD budget used"
            value={fmtPercent(defaultPortfolio.used_ead / defaultPortfolio.ead_budget)}
            sub={`${fmtCurrency(defaultPortfolio.used_ead)} / ${fmtCurrency(defaultPortfolio.ead_budget)}`}
          />
        </section>
      )}

      {run.portfolio_runs.length > 1 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-slate-700">Policy comparison</h2>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">Policy</th>
                  <th className="px-3 py-2">PD max (increase)</th>
                  <th className="px-3 py-2">EAD budget</th>
                  <th className="px-3 py-2">Increases</th>
                  <th className="px-3 py-2">Total EP uplift</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {run.portfolio_runs.map((p) => (
                  <tr key={p.policy_name} className={p.policy_name === "default" ? "bg-slate-50/50" : ""}>
                    <td className="px-3 py-2 font-medium text-slate-900">{p.policy_name}</td>
                    <td className="px-3 py-2 text-slate-600">{fmtPercent(p.pd_increase_max, 0)}</td>
                    <td className="px-3 py-2 text-slate-600">{fmtCurrency(p.ead_budget)}</td>
                    <td className="px-3 py-2 text-slate-600">{p.n_increase_applied}</td>
                    <td className="px-3 py-2 text-slate-600">{fmtCurrency(p.total_ep_uplift)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {run.stress_test_results.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-slate-700">Stress test (PD shock)</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="h-64 rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={run.stress_test_results}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="pd_shock" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="n_increase" name="Increases" fill="#16a34a" />
                  <Bar dataKey="n_decrease" name="Decreases" fill="#dc2626" />
                  <Bar dataKey="n_hold" name="Holds" fill="#94a3b8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="h-64 rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={run.stress_test_results}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="pd_shock" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => fmtCurrency(v)} />
                  <Tooltip formatter={(v) => fmtCurrency(Number(v))} />
                  <Bar dataKey="total_ep_uplift" name="Total EP uplift" fill="#0f172a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      )}

      {Object.keys(segmentGroups).length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-slate-700">Segment breakdown</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {Object.entries(segmentGroups).map(([segment, rows]) => (
              <div key={segment} className="h-56 rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
                <div className="mb-1 text-xs font-medium text-slate-500">{segment}</div>
                <ResponsiveContainer width="100%" height="85%">
                  <BarChart data={rows} layout="vertical" margin={{ left: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" tickFormatter={(v) => fmtPercent(v, 0)} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="segment_value" width={90} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => fmtPercent(Number(v))} />
                    <Bar dataKey="increase_rate" name="Increase rate" fill="#16a34a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </section>
      )}

      {run.fairness_checks.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-slate-700">Fair-lending check</h2>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">Segment</th>
                  <th className="px-3 py-2">Min/max approval ratio</th>
                  <th className="px-3 py-2">Flag</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {run.fairness_checks.map((f) => (
                  <tr key={f.segment}>
                    <td className="px-3 py-2 text-slate-900">{f.segment}</td>
                    <td className="px-3 py-2 text-slate-600">{f.min_max_approval_ratio.toFixed(3)}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                          f.flag === "REVIEW" ? "bg-amber-100 text-amber-800" : "bg-green-100 text-green-800"
                        }`}
                      >
                        {f.flag}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {run.backtest_results.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-slate-700">Backtest (history-window comparison)</h2>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">Window (months)</th>
                  <th className="px-3 py-2">ROC-AUC</th>
                  <th className="px-3 py-2">PR-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {run.backtest_results.map((b) => (
                  <tr key={b.window_months}>
                    <td className="px-3 py-2 text-slate-900">{b.window_months}</td>
                    <td className="px-3 py-2 text-slate-600">{b.val_roc_auc.toFixed(4)}</td>
                    <td className="px-3 py-2 text-slate-600">{b.val_pr_auc.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
