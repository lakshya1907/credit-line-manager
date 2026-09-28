import { useState } from "react";
import { Link } from "react-router-dom";
import { useRuns, useTriggerRun, useJobPolling } from "../api/hooks";
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
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-slate-900">Model Runs</h1>
        <button
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
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
      </div>

      {job && (
        <div
          className={`rounded-md border p-3 text-sm ${
            job.status === "failed"
              ? "border-red-200 bg-red-50 text-red-800"
              : job.status === "completed"
                ? "border-green-200 bg-green-50 text-green-800"
                : "border-amber-200 bg-amber-50 text-amber-800"
          }`}
        >
          Job {job.job_id.slice(0, 8)}: <strong>{job.status}</strong>
          {job.run_id && <> — run <Link className="underline" to={`/runs/${job.run_id}`}>{job.run_id}</Link></>}
          {job.error && <> — {job.error}</>}
        </div>
      )}

      {isLoading && <p className="text-sm text-slate-500">Loading…</p>}
      {error && <p className="text-sm text-red-600">{(error as Error).message}</p>}

      {runs && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-2">Run ID</th>
                <th className="px-4 py-2">Started</th>
                <th className="px-4 py-2">Wall time</th>
                <th className="px-4 py-2">ROC-AUC</th>
                <th className="px-4 py-2">EAD MAE</th>
                <th className="px-4 py-2">Commit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((run) => (
                <tr key={run.run_id} className="hover:bg-slate-50">
                  <td className="px-4 py-2">
                    <Link className="font-medium text-slate-900 hover:underline" to={`/runs/${run.run_id}`}>
                      {run.run_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-slate-600">{fmtDateTime(run.started_at)}</td>
                  <td className="px-4 py-2 text-slate-600">{run.wall_time_seconds.toFixed(1)}s</td>
                  <td className="px-4 py-2 text-slate-600">{run.pd_roc_auc.toFixed(4)}</td>
                  <td className="px-4 py-2 text-slate-600">{run.ead_mae.toFixed(1)}</td>
                  <td className="px-4 py-2 font-mono text-xs text-slate-400">
                    {run.git_commit?.slice(0, 7) ?? "—"}
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-slate-400">
                    No runs yet — trigger one above, or run `python run_all.py` + `python sync_run_to_db.py`.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
