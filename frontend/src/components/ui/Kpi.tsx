export function Kpi({
  label, value, sub, tone = "neutral", size = "primary",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "neutral" | "positive" | "negative" | "warning";
  size?: "primary" | "supporting";
}) {
  const valueColor = {
    neutral: "text-zinc-900",
    positive: "text-[var(--color-positive)]",
    negative: "text-[var(--color-negative)]",
    warning: "text-[var(--color-warning)]",
  }[tone];

  if (size === "supporting") {
    return (
      <div className="rounded-lg border border-zinc-200 bg-white px-3.5 py-3">
        <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
        <div className={`mt-0.5 font-financial text-base font-semibold ${valueColor}`}>{value}</div>
        {sub && <div className="mt-0.5 text-[11px] text-zinc-500">{sub}</div>}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-1 font-financial text-2xl font-semibold ${valueColor}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}
