import {
  CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { Action, CustomerSamplePoint } from "../../api/types";
import { ChartContainer } from "../ui/ChartContainer";
import { fmtCurrency, fmtPd } from "../../lib/format";

const ACTION_COLOR: Record<Action, string> = {
  increase: "var(--color-positive)",
  decrease: "var(--color-negative)",
  hold: "var(--color-neutral)",
};

const ACTION_LABEL: Record<Action, string> = { increase: "Increase", decrease: "Decrease", hold: "Hold" };

function byAction(points: CustomerSamplePoint[]): Record<Action, CustomerSamplePoint[]> {
  const out: Record<Action, CustomerSamplePoint[]> = { increase: [], decrease: [], hold: [] };
  for (const p of points) out[p.action].push(p);
  return out;
}

function ActionSeries({ points }: { points: CustomerSamplePoint[] }) {
  const groups = byAction(points);
  return (
    <>
      {(Object.keys(groups) as Action[]).map((action) => (
        <Scatter key={action} name={ACTION_LABEL[action]} data={groups[action]} fill={ACTION_COLOR[action]} fillOpacity={0.55} />
      ))}
    </>
  );
}

const tooltipStyle = { fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" };

/** X = utilization, Y = PD. A correlation the PD model has learned from the
 * training data, not a causal claim -- see CLAUDE.md's note that
 * counterfactual PD drops sharply whenever utilization falls, which is
 * exactly the shape this chart is expected to show. */
export function PdVsUtilizationScatter({ points }: { points: CustomerSamplePoint[] }) {
  const data = points.filter((p) => p.utilization !== null);
  return (
    <ChartContainer height="h-72 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
          <XAxis
            type="number" dataKey="utilization" name="Utilization" domain={[0, "dataMax"]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11 }} label={{ value: "Utilization", position: "insideBottom", offset: -4, fontSize: 11, fill: "#71717a" }}
          />
          <YAxis
            type="number" dataKey="pd_current" name="PD" tickFormatter={(v) => fmtPd(v)} tick={{ fontSize: 11 }}
            label={{ value: "PD", angle: -90, position: "insideLeft", fontSize: 11, fill: "#71717a" }}
          />
          <ZAxis range={[24, 24]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={tooltipStyle}
            formatter={(value, name) => (name === "PD" ? fmtPd(Number(value)) : `${(Number(value) * 100).toFixed(1)}%`)}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ActionSeries points={data} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

/** Current vs. recommended limit -- the diagonal is "no change"; points
 * above it are increases, below are decreases, colored by the actual
 * decided action (guardrails/budget already applied). */
export function LimitComparisonScatter({ points }: { points: CustomerSamplePoint[] }) {
  return (
    <ChartContainer height="h-72 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
          <XAxis
            type="number" dataKey="current_limit" name="Current limit" tickFormatter={(v) => fmtCurrency(v)} tick={{ fontSize: 11 }}
            label={{ value: "Current limit", position: "insideBottom", offset: -4, fontSize: 11, fill: "#71717a" }}
          />
          <YAxis
            type="number" dataKey="recommended_limit" name="Recommended limit" tickFormatter={(v) => fmtCurrency(v)} tick={{ fontSize: 11 }}
            label={{ value: "Recommended limit", angle: -90, position: "insideLeft", fontSize: 11, fill: "#71717a" }}
          />
          <ZAxis range={[24, 24]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={tooltipStyle} formatter={(v) => fmtCurrency(Number(v))} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ActionSeries points={points} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

/** PD vs. expected-profit uplift -- the economic/risk tradeoff the decision
 * engine navigates. Association across customers, not a causal estimate of
 * what raising any one customer's PD would do to their profit. */
export function PdVsProfitScatter({ points }: { points: CustomerSamplePoint[] }) {
  return (
    <ChartContainer height="h-72 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
          <XAxis
            type="number" dataKey="pd_current" name="PD" tickFormatter={(v) => fmtPd(v)} tick={{ fontSize: 11 }}
            label={{ value: "PD", position: "insideBottom", offset: -4, fontSize: 11, fill: "#71717a" }}
          />
          <YAxis
            type="number" dataKey="ep_uplift" name="EP uplift" tickFormatter={(v) => fmtCurrency(v)} tick={{ fontSize: 11 }}
            label={{ value: "EP uplift", angle: -90, position: "insideLeft", fontSize: 11, fill: "#71717a" }}
          />
          <ZAxis range={[24, 24]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={tooltipStyle}
            formatter={(value, name) => (name === "PD" ? fmtPd(Number(value)) : fmtCurrency(Number(value)))}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ActionSeries points={points} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}
