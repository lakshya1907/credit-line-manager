import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelRunDetail } from "../../api/types";
import { Card, CardHeader } from "../ui/Card";
import { ChartContainer } from "../ui/ChartContainer";
import { EmptyState } from "../ui/States";
import { fmtCurrency } from "../../lib/format";

export function StressTestSection({ run }: { run: ModelRunDetail }) {
  if (run.stress_test_results.length === 0) {
    return <EmptyState title="No stress test results for this run" />;
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-500">
        Each shock level re-runs the full decision engine under a shocked PD (genuine re-simulation, not a
        rescaled approximation — see <code className="rounded bg-zinc-100 px-1 font-financial text-xs">src/stress_test.py</code>),
        so the recommended action itself can change under stress, not just its reported numbers.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader title="Actions by shock level" />
          <ChartContainer height="h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={run.stress_test_results}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
                <XAxis dataKey="pd_shock" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="n_increase" name="Increases" fill="var(--color-positive)" />
                <Bar dataKey="n_decrease" name="Decreases" fill="var(--color-negative)" />
                <Bar dataKey="n_hold" name="Holds" fill="var(--color-neutral)" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>
        <Card>
          <CardHeader title="Total EP uplift by shock level" />
          <ChartContainer height="h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={run.stress_test_results}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
                <XAxis dataKey="pd_shock" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => fmtCurrency(v)} />
                <Tooltip formatter={(v) => fmtCurrency(Number(v))} />
                <Bar dataKey="total_ep_uplift" name="Total EP uplift" fill="var(--color-brand-600)" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>
      </div>
    </div>
  );
}
