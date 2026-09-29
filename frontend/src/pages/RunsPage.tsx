import { useState } from "react";
import { Link } from "react-router-dom";
import { useRuns, useTriggerRun, useJobPolling } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { StatusBadge } from "../components/ui/StatusBadge";
import { SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { fmtDateTime } from "../lib/format";

export function RunsPage() {
  const { data: runs, isLoading, error } = useRuns(50);
  const trigger = useTriggerRun();
  const [jobId, setJobId] = useState<string>();
  const [polling, setPolling] = useState(false);
  const { data: job } = useJobPolling(jobId, polling);

  if (job?.status === "completed" || job?.status === "failed") {
    if (polling) setPolling(false);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Runs"
        subtitle="Every run_all.py execution, newest first — each triggers a fresh retrain and portfolio re-scoring."
        actions={
          <button
            className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900 focus-visible:ring-offset-1"
            disabled={trigger.isPending || polling}
            onClick={() => {
              trigger.mutate(undefined, {
                onSuccess: (job) => {
                  setJobId(job.job_id);
                  setPolling(true);
                },
              });
            }}
          >
            {polling ? "Running pipeline…" : "Trigger new run"}
          </button>
        }
      />

      {job && (
        <div
          role="status"
          className={`flex items-center gap-2 rounded-md border p-3 text-sm ${
            job.status === "failed"
              ? "border-red-200 bg-red-50 text-red-800"
              : job.status === "completed"
                ? "border-green-200 bg-green-50 text-green-800"
                : "border-amber-200 bg-amber-50 text-amber-800"
          }`}
        >
          <StatusBadge tone={job.status === "failed" ? "danger" : job.status === "completed" ? "success" : "warning"}>
            {job.status}
          </StatusBadge>
          <span>Job {job.job_id.slice(0, 8)}</span>
          {job.run_id && <>— run <Link className="underline" to={`/runs/${job.run_id}`}>{job.run_id}</Link></>}
          {job.error && <>— {job.error}</>}
        </div>
      )}

      {isLoading && <SkeletonTable />}
      {error && <ErrorState error={error} fallback="Could not load run history." />}

      {runs && runs.length === 0 && (
        <EmptyState
          title="No runs yet"
          hint={<>Trigger one above, or run <code className="rounded bg-zinc-100 px-1">python run_all.py</code> and <code className="rounded bg-zinc-100 px-1">python sync_run_to_db.py</code> locally.</>}
        />
      )}

      {runs && runs.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 text-left text-xs uppercase text-zinc-500">
              <tr>
                <th className="px-4 py-2">Run ID</th>
                <th className="px-4 py-2">Started</th>
                <th className="px-4 py-2">Wall time</th>
                <th className="px-4 py-2">ROC-AUC</th>
                <th className="px-4 py-2">EAD MAE</th>
                <th className="px-4 py-2">Commit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {runs.map((run, i) => (
                <tr key={run.run_id} className="hover:bg-zinc-50">
                  <td className="px-4 py-2">
                    <Link className="font-medium text-zinc-900 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900 rounded" to={`/runs/${run.run_id}`}>
                      {run.run_id}
                    </Link>
                    {i === 0 && <span className="ml-2"><StatusBadge tone="info">latest</StatusBadge></span>}
                  </td>
                  <td className="px-4 py-2 text-zinc-600">{fmtDateTime(run.started_at)}</td>
                  <td className="px-4 py-2 text-zinc-600">{run.wall_time_seconds.toFixed(1)}s</td>
                  <td className="px-4 py-2 text-zinc-600">{run.pd_roc_auc.toFixed(4)}</td>
                  <td className="px-4 py-2 text-zinc-600">{run.ead_mae.toFixed(1)}</td>
                  <td className="px-4 py-2 font-mono text-xs text-zinc-400">
                    {run.git_commit?.slice(0, 7) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
