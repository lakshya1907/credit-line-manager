import type { ModelRunDetail } from "../../api/types";
import { EmptyState } from "../ui/States";
import { fmtCurrency, fmtPercent } from "../../lib/format";

/** Read-only: these are pre-computed named scenarios from run_analytics.py's
 * policy_compare.py, not a live slider -- see CLAUDE.md's frontend section
 * for why a real-time portfolio-level re-run isn't a frontend-only change. */
export function PolicyComparisonSection({ run }: { run: ModelRunDetail }) {
  if (run.portfolio_runs.length === 0) {
    return (
      <EmptyState
        title="No named policy scenarios for this run"
        hint="Run python run_analytics.py's policy_compare against this run's models, then re-sync."
      />
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
      <table className="w-full text-sm">
        <thead className="bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500">
          <tr>
            <th className="px-4 py-2.5 font-medium">Policy</th>
            <th className="px-4 py-2.5 font-medium">PD max (increase)</th>
            <th className="px-4 py-2.5 font-medium">EAD budget</th>
            <th className="px-4 py-2.5 text-right font-medium">Increases</th>
            <th className="px-4 py-2.5 text-right font-medium">Total EP uplift</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100">
          {run.portfolio_runs.map((p) => (
            <tr key={p.policy_name} className={p.policy_name === "default" ? "bg-[var(--color-brand-50)]/40" : ""}>
              <td className="px-4 py-2.5 font-medium text-zinc-900">{p.policy_name}</td>
              <td className="px-4 py-2.5 font-financial text-zinc-600">{fmtPercent(p.pd_increase_max, 0)}</td>
              <td className="px-4 py-2.5 font-financial text-zinc-600">{fmtCurrency(p.ead_budget)}</td>
              <td className="px-4 py-2.5 text-right font-financial text-zinc-600">{p.n_increase_applied}</td>
              <td className="px-4 py-2.5 text-right font-financial text-zinc-600">{fmtCurrency(p.total_ep_uplift)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
