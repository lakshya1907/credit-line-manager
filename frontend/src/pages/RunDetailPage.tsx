import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { Kpi } from "../components/ui/Kpi";
import { Card, CardHeader } from "../components/ui/Card";
import { ChartContainer } from "../components/ui/ChartContainer";
import { StatusBadge } from "../components/ui/StatusBadge";
import { Tabs } from "../components/ui/Tabs";
import { SkeletonKpiRow, SkeletonTable, ErrorState } from "../components/ui/States";
import { PolicyComparisonSection } from "../components/sections/PolicyComparisonSection";
import { StressTestSection } from "../components/sections/StressTestSection";
import { AnalyticsSection } from "../components/sections/AnalyticsSection";
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
                  <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => fmtCurrency(v)} />
                  <Tooltip formatter={(v) => fmtCurrency(Number(v))} />
                  <Legend />
                  <Bar dataKey="current" name="Current" fill="var(--color-neutral)" />
                  <Bar dataKey="recommended" name="Recommended" fill="var(--color-brand-600)" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </Card>
          <Card>
            <CardHeader title="Model quality" subtitle="Validation metrics for this run's PD/EAD models" />
            <div className="grid grid-cols-2 gap-4 p-4 text-sm">
              <div><div className="text-xs text-zinc-500">PD ROC-AUC</div><div className="font-financial text-lg font-semibold text-zinc-900">{run.pd_roc_auc.toFixed(4)}</div></div>
              <div><div className="text-xs text-zinc-500">PD PR-AUC</div><div className="font-financial text-lg font-semibold text-zinc-900">{run.pd_pr_auc.toFixed(4)}</div></div>
              <div><div className="text-xs text-zinc-500">EAD MAE</div><div className="font-financial text-lg font-semibold text-zinc-900">{run.ead_mae.toFixed(1)}</div></div>
              <div><div className="text-xs text-zinc-500">Git commit</div><div className="font-financial text-xs text-zinc-600">{run.git_commit?.slice(0, 12) ?? "—"}</div></div>
            </div>
          </Card>
        </section>
      )}

      {tab === "policy" && <PolicyComparisonSection run={run} />}
      {tab === "stress" && <StressTestSection run={run} />}
      {tab === "analytics" && <AnalyticsSection run={run} />}
    </div>
  );
}
