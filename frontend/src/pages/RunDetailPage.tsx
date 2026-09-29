import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { Kpi } from "../components/ui/Kpi";
import { Card, CardHeader } from "../components/ui/Card";
import { ChartContainer } from "../components/ui/ChartContainer";
import { StatusBadge } from "../components/ui/StatusBadge";
import { Tabs } from "../components/ui/Tabs";
import { SkeletonKpiRow, SkeletonTable, ErrorState, EmptyState } from "../components/ui/States";
import { SegmentChart, groupBySegment } from "../components/SegmentChart";
import { fmtCurrency, fmtDateTime, fmtPercent } from "../lib/format";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "policy", label: "Policy Comparison" },
  { id: "stress", label: "Stress Test" },
  { id: "analytics", label: "Analytics" },
];

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const { data: run, isLoading, error } = useRun(runId);
  const [tab, setTab] = useState("overview");

  if (isLoading) {
    return (
      <div className="space-y-6">
        <SkeletonKpiRow />
        <SkeletonTable />
      </div>
    );
  }
  if (error) return <ErrorState error={error} fallback="Could not load this run." />;
  if (!run) return null;

  const defaultPolicy = run.portfolio_runs.find((p) => p.policy_name === "default") ?? run.portfolio_runs[0];
  const segmentGroups = groupBySegment(run.segment_metrics);
  const reviewFlags = run.fairness_checks.filter((f) => f.flag === "REVIEW").length;

  return (
    <div className="space-y-6">
      <PageHeader
        title={run.run_id}
        subtitle={
          <>
            Started {fmtDateTime(run.started_at)} · {run.wall_time_seconds.toFixed(1)}s ·{" "}
            <Link className="underline" to={`/runs/${run.run_id}/recommendations`}>
              View action queue →
            </Link>
          </>
        }
      />

      {defaultPolicy && (
        <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Kpi label="Increases approved" value={String(defaultPolicy.n_increase_applied)} tone="positive" />
          <Kpi label="Decreases" value={String(defaultPolicy.n_decrease)} tone="negative" />
          <Kpi label="Total EP uplift" value={fmtCurrency(defaultPolicy.total_ep_uplift)} />
          <Kpi
            label="EAD budget used"
            value={fmtPercent(defaultPolicy.used_ead / defaultPolicy.ead_budget)}
            sub={`${fmtCurrency(defaultPolicy.used_ead)} / ${fmtCurrency(defaultPolicy.ead_budget)}`}
            tone={defaultPolicy.used_ead / defaultPolicy.ead_budget > 0.9 ? "warning" : "neutral"}
          />
        </section>
      )}

      <Tabs
        items={TABS.map((t) => t.id === "analytics" && reviewFlags > 0
          ? { ...t, badge: <StatusBadge tone="warning">{reviewFlags}</StatusBadge> }
          : t)}
        activeId={tab}
        onChange={setTab}
      />

      {tab === "overview" && (
        <section className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader title="Exposure: current vs. recommended" subtitle="Total EAD across every customer in this run" />
            <ChartContainer height="h-64 p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[{
                    name: "EAD",
                    current: run.exposure_summary.total_current_ead,
                    recommended: run.exposure_summary.total_recommended_ead,
                  }]}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => fmtCurrency(v)} />
                  <Tooltip formatter={(v) => fmtCurrency(Number(v))} />
                  <Legend />
                  <Bar dataKey="current" name="Current" fill="#94a3b8" />
                  <Bar dataKey="recommended" name="Recommended" fill="#0f172a" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </Card>
          <Card>
            <CardHeader title="Model quality" subtitle="Validation metrics for this run's PD/EAD models" />
            <div className="grid grid-cols-2 gap-4 p-4 text-sm">
              <div><div className="text-xs text-slate-500">PD ROC-AUC</div><div className="text-lg font-semibold text-slate-900">{run.pd_roc_auc.toFixed(4)}</div></div>
              <div><div className="text-xs text-slate-500">PD PR-AUC</div><div className="text-lg font-semibold text-slate-900">{run.pd_pr_auc.toFixed(4)}</div></div>
              <div><div className="text-xs text-slate-500">EAD MAE</div><div className="text-lg font-semibold text-slate-900">{run.ead_mae.toFixed(1)}</div></div>
              <div><div className="text-xs text-slate-500">Git commit</div><div className="font-mono text-xs text-slate-600">{run.git_commit?.slice(0, 12) ?? "—"}</div></div>
            </div>
          </Card>
        </section>
      )}

      {tab === "policy" && (
        run.portfolio_runs.length > 0 ? (
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
        ) : (
          <EmptyState title="No named policy scenarios for this run" hint="Run python run_analytics.py's policy_compare against this run's models, then re-sync." />
        )
      )}

      {tab === "stress" && (
        run.stress_test_results.length > 0 ? (
          <div className="space-y-4">
            <p className="text-sm text-slate-500">
              Each shock level re-runs the full decision engine under a shocked PD (genuine re-simulation, not a
              rescaled approximation — see <code className="rounded bg-slate-100 px-1">src/stress_test.py</code>),
              so the recommended action itself can change under stress, not just its reported numbers.
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <Card>
                <CardHeader title="Actions by shock level" />
                <ChartContainer height="h-64 p-3">
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
                </ChartContainer>
              </Card>
              <Card>
                <CardHeader title="Total EP uplift by shock level" />
                <ChartContainer height="h-64 p-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={run.stress_test_results}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="pd_shock" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => fmtCurrency(v)} />
                      <Tooltip formatter={(v) => fmtCurrency(Number(v))} />
                      <Bar dataKey="total_ep_uplift" name="Total EP uplift" fill="#0f172a" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </Card>
            </div>
          </div>
        ) : (
          <EmptyState title="No stress test results for this run" />
        )
      )}

      {tab === "analytics" && (
        <div className="space-y-6">
          {Object.keys(segmentGroups).length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-semibold text-slate-700">Segment breakdown</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {Object.entries(segmentGroups).map(([segment, rows]) => (
                  <SegmentChart key={segment} segment={segment} rows={rows} />
                ))}
              </div>
            </section>
          )}

          {run.fairness_checks.length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-semibold text-slate-700">
                Fair-lending check
                <span className="ml-2 font-normal text-slate-400">
                  (a four-fifths-rule-style screening signal, not a compliance determination — see reason codes for detail)
                </span>
              </h2>
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
                          <StatusBadge tone={f.flag === "REVIEW" ? "warning" : "success"}>{f.flag}</StatusBadge>
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
              <h2 className="mb-2 text-sm font-semibold text-slate-700">
                History-window comparison
                <span className="ml-2 font-normal text-slate-400">
                  (not a time-series backtest — this dataset is one snapshot per customer, not repeated
                  observations over time; compares PD performance using only the N most recent months of behavior)
                </span>
              </h2>
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

          {run.segment_metrics.length === 0 && run.fairness_checks.length === 0 && run.backtest_results.length === 0 && (
            <EmptyState
              title="No analytics for this run yet"
              hint="Run python run_analytics.py against this run's models, then re-sync with sync_run_to_db.py."
            />
          )}
        </div>
      )}
    </div>
  );
}
