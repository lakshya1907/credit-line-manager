import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useDistributions, useCustomerSample, useLatestRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { Kpi } from "../components/ui/Kpi";
import { Card, CardHeader } from "../components/ui/Card";
import { StatusBadge } from "../components/ui/StatusBadge";
import { SkeletonBlock, SkeletonKpiRow, SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { ChartContainer } from "../components/ui/ChartContainer";
import { DistributionBarChart } from "../components/charts/DistributionBarChart";
import { PdVsUtilizationScatter, LimitComparisonScatter, PdVsProfitScatter } from "../components/charts/PortfolioScatters";
import { fmtCurrency, fmtDateTime, fmtPercent, fmtPd } from "../lib/format";

const ACTION_FILL: Record<string, string> = {
  increase: "var(--color-positive)",
  decrease: "var(--color-negative)",
  hold: "var(--color-neutral)",
};

export function OverviewPage() {
  const { data: run, isLoading, error, hasNoRuns } = useLatestRun();
  const distributions = useDistributions(run?.run_id);
  const sample = useCustomerSample(run?.run_id, 1500);

  const defaultPolicy = run?.portfolio_runs.find((p) => p.policy_name === "default") ?? run?.portfolio_runs[0];
  const total = defaultPolicy ? defaultPolicy.n_increase_applied + defaultPolicy.n_decrease + defaultPolicy.n_hold : 0;
  const exposureDelta = run ? run.exposure_summary.total_recommended_ead - run.exposure_summary.total_current_ead : 0;
  const portfolioUtilization = run && run.exposure_summary.total_current_limit > 0
    ? run.exposure_summary.total_current_ead / run.exposure_summary.total_current_limit
    : undefined;
  const approvalRate = defaultPolicy && total > 0 ? defaultPolicy.n_increase_applied / total : undefined;
  const avgSamplePd = sample.data && sample.data.length > 0
    ? sample.data.reduce((s, p) => s + p.pd_current, 0) / sample.data.length
    : undefined;

  const actionData = defaultPolicy
    ? [
        { action: "increase", label: "Increase", count: defaultPolicy.n_increase_applied },
        { action: "hold", label: "Hold", count: defaultPolicy.n_hold },
        { action: "decrease", label: "Decrease", count: defaultPolicy.n_decrease },
      ]
    : [];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Portfolio Overview"
        subtitle={run ? (
          <>
            Credit exposure, decision quality, and portfolio performance — run{" "}
            <Link className="underline" to={`/runs/${run.run_id}`}>{run.run_id}</Link> ·{" "}
            {fmtDateTime(run.started_at)} · <Link className="underline" to="/runs">all runs →</Link>
          </>
        ) : "Credit exposure, decision quality, and portfolio performance"}
      />

      {isLoading && (
        <div className="space-y-8">
          <SkeletonKpiRow count={4} />
          <SkeletonTable rows={4} />
        </div>
      )}

      {error && <ErrorState error={error} fallback="Could not load the latest run." />}

      {hasNoRuns && (
        <EmptyState
          title="No model runs yet"
          hint={<>Run <code className="rounded bg-zinc-100 px-1">python run_all.py</code> and <code className="rounded bg-zinc-100 px-1">python sync_run_to_db.py</code>, or trigger a run from the <Link className="underline" to="/runs">Runs</Link> page.</>}
        />
      )}

      {run && (
        <>
          <section className="space-y-3">
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <Kpi
                label="Recommended exposure"
                value={fmtCurrency(run.exposure_summary.total_recommended_ead)}
                sub={`${exposureDelta >= 0 ? "+" : ""}${fmtCurrency(exposureDelta)} vs current`}
                tone={exposureDelta >= 0 ? "positive" : "negative"}
              />
              <Kpi label="Total EP uplift" value={fmtCurrency(defaultPolicy?.total_ep_uplift ?? 0)} tone="positive" />
              <Kpi
                label="Approval rate"
                value={approvalRate !== undefined ? fmtPercent(approvalRate) : "—"}
                sub={`${defaultPolicy?.n_increase_applied ?? 0} of ${total.toLocaleString()} customers`}
              />
              <Kpi label="Customers" value={total.toLocaleString()} />
            </div>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Kpi size="supporting" label="Portfolio utilization" value={portfolioUtilization !== undefined ? fmtPercent(portfolioUtilization) : "—"} />
              <Kpi size="supporting" label="Avg. PD (sample)" value={avgSamplePd !== undefined ? fmtPd(avgSamplePd) : "—"} sub={sample.data ? `n=${sample.data.length.toLocaleString()}` : undefined} />
              <Kpi
                size="supporting"
                label="EAD budget used"
                value={defaultPolicy ? fmtPercent(defaultPolicy.used_ead / defaultPolicy.ead_budget) : "—"}
                tone={defaultPolicy && defaultPolicy.used_ead / defaultPolicy.ead_budget > 0.9 ? "warning" : "neutral"}
              />
              <Kpi size="supporting" label="Decreases" value={String(defaultPolicy?.n_decrease ?? 0)} tone="negative" />
            </div>
          </section>

          {(run.fairness_checks.some((f) => f.flag === "REVIEW") || run.stress_test_results.length > 0) && (
            <section className="flex flex-wrap gap-3">
              {run.fairness_checks.filter((f) => f.flag === "REVIEW").map((f) => (
                <div key={f.segment} className="flex items-center gap-2 rounded-lg border border-amber-200 bg-[var(--color-warning-soft)]/60 px-3 py-2 text-sm text-amber-900">
                  <StatusBadge tone="warning">Fair-lending review</StatusBadge>
                  <span>{f.segment} ratio {f.min_max_approval_ratio.toFixed(3)} — see <Link className="underline" to="/analytics">Analytics</Link></span>
                </div>
              ))}
              {run.stress_test_results.length > 0 && (() => {
                const worst = run.stress_test_results[run.stress_test_results.length - 1];
                const baseline = run.stress_test_results[0];
                return (
                  <div className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700">
                    <StatusBadge tone="info">Stress status</StatusBadge>
                    <span>
                      At {worst.pd_shock} PD shock, approvals go {baseline.n_increase} → {worst.n_increase} — see{" "}
                      <Link className="underline" to="/stress-testing">Stress Testing</Link>
                    </span>
                  </div>
                );
              })()}
            </section>
          )}

          <section className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader title="Risk distribution" subtitle="Customers by probability of default" />
              {distributions.data
                ? <DistributionBarChart data={distributions.data.risk_distribution} color="var(--color-warning)" />
                : <div className="h-56 p-3"><SkeletonBlock className="h-full w-full" /></div>}
            </Card>

            <Card>
              <CardHeader title="Credit action distribution" subtitle="What the plan recommends for the whole book" />
              <ChartContainer height="h-56 p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={actionData} margin={{ left: -12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} />
                    <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
                    <Tooltip formatter={(v) => [Number(v).toLocaleString(), "Customers"]} contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }} />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]} maxBarSize={48}>
                      {actionData.map((d) => <Cell key={d.action} fill={ACTION_FILL[d.action]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </Card>

            <Card>
              <CardHeader title="Utilization distribution" subtitle="Current exposure as a share of current limit" />
              {distributions.data
                ? <DistributionBarChart data={distributions.data.utilization_distribution} color="var(--color-brand-600)" />
                : <div className="h-56 p-3"><SkeletonBlock className="h-full w-full" /></div>}
            </Card>

            <Card>
              <CardHeader title="Recommended limit change" subtitle="Direction of the recommended move vs. current limit" />
              {distributions.data
                ? <DistributionBarChart data={distributions.data.limit_change_distribution} color="var(--color-neutral)" />
                : <div className="h-56 p-3"><SkeletonBlock className="h-full w-full" /></div>}
            </Card>
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader title="PD vs. utilization" subtitle="Observed model relationship across a sample of customers — not causal evidence" />
              {sample.data && <PdVsUtilizationScatter points={sample.data} />}
            </Card>

            <Card>
              <CardHeader title="Current vs. recommended limit" subtitle="Points above the diagonal are increases, below are decreases" />
              {sample.data && <LimitComparisonScatter points={sample.data} />}
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader title="PD vs. expected-profit uplift" subtitle="The risk/economic tradeoff the decision engine navigates — association, not a causal estimate" />
              {sample.data && <PdVsProfitScatter points={sample.data} />}
            </Card>
          </section>
        </>
      )}
    </div>
  );
}
