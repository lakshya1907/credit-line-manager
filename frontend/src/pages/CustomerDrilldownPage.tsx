import { useState } from "react";
import { Link } from "react-router-dom";
import { useCustomerHistory, useScoreCustomer } from "../api/hooks";
import { ActionBadge } from "../components/ActionBadge";
import { fmtCurrency, fmtDateTime } from "../lib/format";
import { ApiError } from "../api/client";

export function CustomerDrilldownPage() {
  const [input, setInput] = useState("");
  const [customerId, setCustomerId] = useState<number>();
  const [newLimit, setNewLimit] = useState("");

  const history = useCustomerHistory(customerId);
  const score = useScoreCustomer(customerId);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const id = Number(input);
    if (id > 0) setCustomerId(id);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Customer Lookup</h1>
        <p className="text-sm text-slate-500">Recommendation history across every synced run, plus live what-if scoring.</p>
      </div>

      <form onSubmit={submit} className="flex gap-2">
        <input
          className="w-48 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
          placeholder="Customer ID"
          value={input}
          onChange={(e) => setInput(e.target.value.replace(/\D/g, ""))}
        />
        <button type="submit" className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white">
          Look up
        </button>
      </form>

      {customerId && (
        <div className="grid gap-6 lg:grid-cols-2">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-700">History</h2>
            {history.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
            {history.error && (
              <p className="text-sm text-red-600">
                {history.error instanceof ApiError && history.error.status === 404
                  ? `No history found for customer ${customerId}.`
                  : (history.error as Error).message}
              </p>
            )}
            {history.data && (
              <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
                    <tr>
                      <th className="px-3 py-2">Run</th>
                      <th className="px-3 py-2">Action</th>
                      <th className="px-3 py-2">Limit</th>
                      <th className="px-3 py-2">PD</th>
                      <th className="px-3 py-2">EP uplift</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {history.data.map((h) => (
                      <tr key={h.run_id}>
                        <td className="px-3 py-2">
                          <Link className="text-xs text-slate-500 hover:underline" to={`/runs/${h.run_id}`}>
                            {fmtDateTime(h.started_at)}
                          </Link>
                        </td>
                        <td className="px-3 py-2"><ActionBadge action={h.action} /></td>
                        <td className="px-3 py-2 text-slate-600">
                          {fmtCurrency(h.current_limit)} → {fmtCurrency(h.recommended_limit)}
                        </td>
                        <td className="px-3 py-2 text-slate-600">
                          {h.pd_current.toFixed(3)} → {h.pd_recommended.toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-slate-600">{fmtCurrency(h.ep_uplift)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-700">What-if scoring</h2>
            <p className="mb-3 text-xs text-slate-500">
              Against the currently loaded models (not the database). Leave the field blank to run the same
              best-candidate search the decision engine uses; set a limit to evaluate that exact scenario.
            </p>
            <div className="flex gap-2">
              <input
                className="w-48 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                placeholder="Hypothetical new limit (optional)"
                value={newLimit}
                onChange={(e) => setNewLimit(e.target.value.replace(/\D/g, ""))}
              />
              <button
                className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
                disabled={score.isPending}
                onClick={() => score.mutate(newLimit ? { new_limit: Number(newLimit) } : {})}
              >
                {score.isPending ? "Scoring…" : "Score"}
              </button>
            </div>

            {score.isError && (
              <p className="mt-3 text-sm text-red-600">
                {score.error instanceof ApiError ? score.error.message : "Failed to score customer."}
              </p>
            )}

            {score.data && (
              <div className="mt-3 space-y-2 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <ActionBadge action={score.data.action} />
                  {score.data.guardrail_blocked && (
                    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                      guardrail would block this increase
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm text-slate-600">
                  <div>Limit: {fmtCurrency(score.data.current_limit)} → {fmtCurrency(score.data.evaluated_limit)}</div>
                  <div>PD: {score.data.pd_current.toFixed(3)} → {score.data.pd_evaluated.toFixed(3)}</div>
                  <div>EAD: {fmtCurrency(score.data.ead_current)} → {fmtCurrency(score.data.ead_evaluated)}</div>
                  <div>EP uplift: {fmtCurrency(score.data.ep_uplift)}</div>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
