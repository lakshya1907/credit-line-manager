import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SegmentMetric } from "../api/types";
import { fmtPercent } from "../lib/format";
import { ChartContainer } from "./ui/ChartContainer";

export function groupBySegment(rows: SegmentMetric[]): Record<string, SegmentMetric[]> {
  const out: Record<string, SegmentMetric[]> = {};
  for (const r of rows) (out[r.segment] ??= []).push(r);
  return out;
}

export function SegmentChart({ segment, rows }: { segment: string; rows: SegmentMetric[] }) {
  return (
    <div className="h-56 rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-1 text-xs font-medium text-slate-500">{segment}</div>
      <ChartContainer height="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ left: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tickFormatter={(v) => fmtPercent(v, 0)} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="segment_value" width={90} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => fmtPercent(Number(v))} />
            <Bar dataKey="increase_rate" name="Increase rate" fill="#16a34a" />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}
