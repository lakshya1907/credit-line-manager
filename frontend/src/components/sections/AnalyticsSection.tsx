import type { ModelRunDetail } from "../../api/types";
import { Card } from "../ui/Card";
import { StatusBadge } from "../ui/StatusBadge";
import { EmptyState } from "../ui/States";
import { SegmentChart, groupBySegment } from "../SegmentChart";

/** Segmentation + model-quality + fair-lending + history-window sections,
 * shared between a specific historical run (Run Detail's Analytics tab) and
 * the latest-run top-level Analytics page. "Model Performance" here is
 * built entirely from metrics the pipeline already persists (ROC-AUC,
 * PR-AUC, EAD MAE, the history-window backtest) -- true predicted-vs-
 * observed calibration would need a ground-truth default column this
 * schema doesn't persist (see CLAUDE.md), so it's deliberately left out
 * rather than fabricated. */
export function AnalyticsSection({ run }: { run: ModelRunDetail }) {
  const segmentGroups = groupBySegment(run.segment_metrics);

  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-3 text-sm font-semibold text-zinc-800">Model performance</h2>
        <Card>
          <div className="grid grid-cols-2 gap-4 p-4 text-sm sm:grid-cols-4">
            <div>
              <div className="text-xs text-zinc-500">PD ROC-AUC</div>
              <div className="mt-0.5 font-financial text-lg font-semibold text-zinc-900">{run.pd_roc_auc.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs text-zinc-500">PD PR-AUC</div>
              <div className="mt-0.5 font-financial text-lg font-semibold text-zinc-900">{run.pd_pr_auc.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs text-zinc-500">EAD MAE</div>
              <div className="mt-0.5 font-financial text-lg font-semibold text-zinc-900">{run.ead_mae.toFixed(1)}</div>
            </div>
            <div>
              <div className="text-xs text-zinc-500">Git commit</div>
              <div className="mt-0.5 font-financial text-xs text-zinc-600">{run.git_commit?.slice(0, 12) ?? "—"}</div>
            </div>
          </div>
        </Card>
      </section>

      {run.backtest_results.length > 0 && (
        <section>
          <h2 className="mb-1 text-sm font-semibold text-zinc-800">History-window comparison</h2>
          <p className="mb-3 text-xs text-zinc-500">
            Not a time-series backtest — this dataset is one snapshot per customer, not repeated observations over
            time. Compares PD performance using only the N most recent months of behavior via a small, deliberately
            restricted feature set (absolute metrics aren't comparable to the production model above).
          </p>
          <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Window (months)</th>
                  <th className="px-4 py-2.5 font-medium">ROC-AUC</th>
                  <th className="px-4 py-2.5 font-medium">PR-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {run.backtest_results.map((b) => (
                  <tr key={b.window_months}>
                    <td className="px-4 py-2.5 font-financial text-zinc-900">{b.window_months}</td>
                    <td className="px-4 py-2.5 font-financial text-zinc-600">{b.val_roc_auc.toFixed(4)}</td>
                    <td className="px-4 py-2.5 font-financial text-zinc-600">{b.val_pr_auc.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {Object.keys(segmentGroups).length > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold text-zinc-800">Portfolio segmentation</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {Object.entries(segmentGroups).map(([segment, rows]) => (
              <SegmentChart key={segment} segment={segment} rows={rows} />
            ))}
          </div>
        </section>
      )}

      {run.fairness_checks.length > 0 && (
        <section>
          <h2 className="mb-1 text-sm font-semibold text-zinc-800">
            Fair-lending check
          </h2>
          <p className="mb-3 text-xs text-zinc-500">
            SEX/EDUCATION/MARRIAGE are fed to the PD model as raw numeric features — a real ECOA fair-lending
            concern independent of intent. This is a four-fifths-rule-style screening signal, not a compliance
            determination.
          </p>
          <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Segment</th>
                  <th className="px-4 py-2.5 font-medium">Min/max approval ratio</th>
                  <th className="px-4 py-2.5 font-medium">Flag</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {run.fairness_checks.map((f) => (
                  <tr key={f.segment}>
                    <td className="px-4 py-2.5 text-zinc-900">{f.segment}</td>
                    <td className="px-4 py-2.5 font-financial text-zinc-600">{f.min_max_approval_ratio.toFixed(3)}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge tone={f.flag === "REVIEW" ? "warning" : "success"}>{f.flag}</StatusBadge>
                    </td>
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
  );
}
