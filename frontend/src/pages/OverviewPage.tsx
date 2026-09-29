import { Link } from "react-router-dom";
import { useLatestRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { Kpi } from "../components/ui/Kpi";
import { Card, CardHeader } from "../components/ui/Card";
import { StatusBadge } from "../components/ui/StatusBadge";
import { SkeletonKpiRow, SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { SegmentChart, groupBySegment } from "../components/SegmentChart";
import { fmtCurrency, fmtDateTime, fmtPercent } from "../lib/format";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartContainer } from "../components/ui/ChartContainer";

export function OverviewPage() {
  const { data: run, isLoading, error, hasNoRuns } = useLatestRun();

  return (
    <div className="space-y-8">
      <PageHeader
        title="Portfolio Overview"
        subtitle={run ? (
          <>
            Latest run: <Link className="underline" to={`/runs/${run.run_id}`}>{run.run_id}</Link> ·{" "}
            {fmtDateTime(run.started_at)} ·{" "}
            <Link className="underline" to="/runs">all runs →</Link>
          </>
        ) : "No runs yet"}
      />

      {isLoading && (
        <div className="space-y-8">
          <SkeletonKpiRow count={6} />
          <SkeletonTable rows={4} />
        </div>
      )}

      {error && <ErrorState error={error} fallback="Could not load the latest run." />}

      {hasNoRuns && (
        <EmptyState
          title="No model runs yet"
          hint={<>Run <code className="rounded bg-slate-100 px-1">python run_all.py</code> and <code className="rounded bg-slate-100 px-1">python sync_run_to_db.py</code>, or trigger a run from the <Link className="underline" to="/runs">Runs</Link> page.</>}
        />
      )}

      {run && (
        <>
          {(() => {
            const p = run.portfolio_runs.find((x) => x.policy_name === "default") ?? run.portfolio_runs[0];
            const total = (p?.n_increase_applied ?? 0) + (p?.n_decrease ?? 0) + (p?.n_hold ?? 0);
            const exposureDelta = run.exposure_summary.total_recommended_ead - run.exposure_summary.total_current_ead;
            return (
              <section className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
                <Kpi label="Total customers" value={total.toLocaleString()} />
                <Kpi label="Increases" value={String(p?.n_increase_applied ?? 0)} tone="positive" />
                <Kpi label="Holds" value={String(p?.n_hold ?? 0)} />
                <Kpi label="Decreases" value={String(p?.n_decrease ?? 0)} tone="negative" />
                <Kpi
                  label="Recommended exposure"
                  value={fmtCurrency(run.exposure_summary.total_recommended_ead)}
                  sub={`${exposureDelta >= 0 ? "+" : ""}${fmtCurrency(exposureDelta)} vs current`}
                  tone={exposureDelta >= 0 ? "positive" : "negative"}
                />
                <Kpi
                  label="EAD budget used"
                  value={p ? fmtPercent(p.used_ead / p.ead_budget) : "—"}
                  sub={p ? `${fmtCurrency(p.used_ead)} / ${fmtCurrency(p.ead_budget)}` : undefined}
                  tone={p && p.used_ead / p.ead_budget > 0.9 ? "warning" : "neutral"}
                />
              </section>
            );
          })()}

          {(run.fairness_checks.some((f) => f.flag === "REVIEW") || run.stress_test_results.length > 0) && (
            <section className="flex flex-wrap gap-3">
              {run.fairness_checks.filter((f) => f.flag === "REVIEW").map((f) => (
                <div key={f.segment} className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                  <StatusBadge tone="warning">Fair-lending review</StatusBadge>
                  <span>{f.segment} ratio {f.min_max_approval_ratio.toFixed(3)} — see Run Detail → Analytics</span>
                </div>
              ))}
              {run.stress_test_results.length > 0 && (() => {
                const worst = run.stress_test_results[run.stress_test_results.length - 1];
                const baseline = run.stress_test_results[0];
                return (
                  <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                    <StatusBadge tone="info">Stress status</StatusBadge>
                    <span>
                      At {worst.pd_shock} PD shock, approvals go {baseline.n_increase} → {worst.n_increase} — see Run Detail → Stress Test
                    </span>
                  </div>
                );
              })()}
            </section>
          )}

          <section className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader title="Action distribution" subtitle="What the plan recommends for the whole book" />
              <ChartContainer height="h-64 p-3">
                {(() => {
                  const p = run.portfolio_runs.find((x) => x.policy_name === "default") ?? run.portfolio_runs[0];
                  const data = p ? [
                    { action: "increase", count: p.n_increase_applied },
                    { action: "hold", count: p.n_hold },
                    { action: "decrease", count: p.n_decrease },
                  ] : [];
                  return (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="action" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="count" name="Customers">
                          {data.map((d) => (
                            <Cell key={d.action} fill={d.action === "increase" ? "#16a34a" : d.action === "decrease" ? "#dc2626" : "#94a3b8"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  );
                })()}
              </ChartContainer>
            </Card>

            <Card>
              <CardHeader title="Exposure: current vs. recommended" subtitle="Total EAD across every customer, whole book" />
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
          </section>

          {run.segment_metrics.length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-semibold text-slate-700">Key segment: delinquency history</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {(() => {
                  const groups = groupBySegment(run.segment_metrics);
                  const preferred = groups["delinquency_tier"] ? "delinquency_tier" : Object.keys(groups)[0];
                  return preferred ? <SegmentChart segment={preferred} rows={groups[preferred]} /> : null;
                })()}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
