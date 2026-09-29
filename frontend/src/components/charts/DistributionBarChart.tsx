import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DistributionBucket } from "../../api/types";
import { ChartContainer } from "../ui/ChartContainer";

/** A single bucketed-count histogram, shared by every distribution chart on
 * the Overview page. Buckets come from the API's SQL GROUP BY over the full
 * recommendation set for the run (see GET /runs/{id}/distributions) -- real
 * counts, not a client-side approximation. */
export function DistributionBarChart({ data, color = "var(--color-brand-600)" }: { data: DistributionBucket[]; color?: string }) {
  return (
    <ChartContainer height="h-56 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" vertical={false} />
          <XAxis dataKey="bucket" tick={{ fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#e4e4e7" }} />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
          <Tooltip
            formatter={(v) => [Number(v).toLocaleString(), "Customers"]}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e4e4e7" }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} fill={color} maxBarSize={40} />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}
