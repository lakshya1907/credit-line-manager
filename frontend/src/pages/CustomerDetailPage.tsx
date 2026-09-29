import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useCustomerHistory, useCustomerRiskProfile, useScoreCustomer } from "../api/hooks";
import { ActionBadge } from "../components/ActionBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { Card, CardHeader } from "../components/ui/Card";
import { Kpi } from "../components/ui/Kpi";
import { StatusBadge } from "../components/ui/StatusBadge";
import { SkeletonKpiRow, SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { fmtCurrency, fmtDateTime, fmtPd } from "../lib/format";
import { ApiError } from "../api/client";

function SearchForm({ initial }: { initial?: string }) {
  const [input, setInput] = useState(initial ?? "");
  const navigate = useNavigate();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const id = Number(input);
        if (id > 0) navigate(`/customers/${id}`);
      }}
      className="flex gap-2"
    >
      <input
        className="w-48 rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
        placeholder="Customer ID"
        value={input}
        onChange={(e) => setInput(e.target.value.replace(/\D/g, ""))}
        aria-label="Customer ID"
      />
      <button type="submit" className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-1">
        Look up
      </button>
    </form>
  );
}

export function CustomerDetailPage() {
  const params = useParams<{ customerId?: string }>();
  const customerId = params.customerId ? Number(params.customerId) : undefined;

  const risk = useCustomerRiskProfile(customerId);
  const history = useCustomerHistory(customerId);
  const [newLimit, setNewLimit] = useState("");
  const whatIf = useScoreCustomer(customerId);

  if (!customerId) {
    return (
      <div className="space-y-6">
        <PageHeader title="Customer Lookup" subtitle="Look up a customer to see their risk profile, recommendation, and history." />
        <SearchForm />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader
        title={`Customer ${customerId}`}
        subtitle="Risk profile, recommendation, and cross-run history"
        actions={<SearchForm initial={String(customerId)} />}
      />

      {risk.isLoading && <SkeletonKpiRow />}
      {risk.error && (
        <ErrorState
          error={risk.error}
          fallback={risk.error instanceof ApiError && risk.error.status === 404
            ? `Customer ${customerId} not found in the current feature set.`
            : "Could not score this customer."}
        />
      )}

      {risk.data && (
        <>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-700">Risk profile & recommendation</h2>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Kpi label="Probability of default" value={fmtPd(risk.data.pd_current)} tone={risk.data.pd_current > 0.2 ? "warning" : "neutral"} />
              <Kpi label="Current exposure (EAD)" value={fmtCurrency(risk.data.ead_current)} />
              <Kpi label="Expected loss" value={fmtCurrency(risk.data.pd_current * risk.data.ead_current)} sub="PD × EAD, current" />
              <Kpi label="EP uplift if applied" value={fmtCurrency(risk.data.ep_uplift)} tone={risk.data.ep_uplift >= 0 ? "positive" : "negative"} />
            </div>
          </section>

          <Card>
            <CardHeader title="Recommendation" />
            <div className="flex flex-wrap items-center gap-6 p-4">
              <ActionBadge action={risk.data.action} />
              <div className="text-sm text-slate-700">
                <span className="text-slate-500">Limit: </span>
                {fmtCurrency(risk.data.current_limit)} → <strong>{fmtCurrency(risk.data.evaluated_limit)}</strong>
                <span className="ml-2 text-xs text-slate-400">
                  ({risk.data.evaluated_limit >= risk.data.current_limit ? "+" : ""}
                  {fmtCurrency(risk.data.evaluated_limit - risk.data.current_limit)})
                </span>
              </div>
              <div className="text-sm text-slate-700">
                <span className="text-slate-500">PD: </span>{fmtPd(risk.data.pd_current)} → {fmtPd(risk.data.pd_evaluated)}
              </div>
            </div>
          </Card>
        </>
      )}

      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Historical decisions</h2>
        {history.isLoading && <SkeletonTable rows={3} />}
        {history.error && (
          <ErrorState
            error={history.error}
            fallback={history.error instanceof ApiError && history.error.status === 404
              ? `No recommendation history synced for customer ${customerId} yet.`
              : "Could not load history."}
          />
        )}
        {history.data && history.data.length === 0 && (
          <EmptyState title="No history yet" hint="This customer hasn't appeared in any synced run." />
        )}
        {history.data && history.data.length > 0 && (
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
                      {fmtPd(h.pd_current)} → {fmtPd(h.pd_recommended)}
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
        <h2 className="mb-1 text-sm font-semibold text-slate-700">What-if scoring</h2>
        <p className="mb-3 text-xs text-slate-500">
          Live, against the currently loaded models — not the database. Set a hypothetical limit to see exactly
          how the guardrails and economics respond; this is deliberate risk analysis, not a preview of what will
          be applied.
        </p>
        <div className="flex gap-2">
          <input
            className="w-56 rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
            placeholder="Hypothetical new limit"
            value={newLimit}
            onChange={(e) => setNewLimit(e.target.value.replace(/\D/g, ""))}
            aria-label="Hypothetical new limit"
          />
          <button
            className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-1"
            disabled={whatIf.isPending || !newLimit}
            onClick={() => whatIf.mutate({ new_limit: Number(newLimit) })}
          >
            {whatIf.isPending ? "Scoring…" : "Score this limit"}
          </button>
        </div>

        {whatIf.isError && <div className="mt-3"><ErrorState error={whatIf.error} fallback="Failed to score this scenario." /></div>}

        {whatIf.data && (
          <div className="mt-3 space-y-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs uppercase tracking-wide text-slate-400">Hypothetical input</span>
              <span className="text-sm text-slate-700">{fmtCurrency(whatIf.data.evaluated_limit)}</span>
              <span className="mx-2 text-slate-300">|</span>
              <span className="text-xs uppercase tracking-wide text-slate-400">Model result</span>
              <ActionBadge action={whatIf.data.action} />
              {whatIf.data.guardrail_blocked && (
                <StatusBadge tone="warning">guardrail would block this increase</StatusBadge>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm text-slate-600 sm:grid-cols-4">
              <div>PD: {fmtPd(whatIf.data.pd_current)} → {fmtPd(whatIf.data.pd_evaluated)}</div>
              <div>EAD: {fmtCurrency(whatIf.data.ead_current)} → {fmtCurrency(whatIf.data.ead_evaluated)}</div>
              <div>EP: {fmtCurrency(whatIf.data.ep_current)} → {fmtCurrency(whatIf.data.ep_evaluated)}</div>
              <div>EP uplift: {fmtCurrency(whatIf.data.ep_uplift)}</div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
