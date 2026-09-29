import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useCustomerHistory, useCustomerRiskProfile, useLatestRun, useRecommendations, useScoreCustomer } from "../api/hooks";
import { ActionBadge } from "../components/ActionBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
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
        className="w-48 rounded-md border border-zinc-300 px-3 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
        placeholder="Customer ID"
        value={input}
        onChange={(e) => setInput(e.target.value.replace(/\D/g, ""))}
        aria-label="Customer ID"
      />
      <button type="submit" className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)] focus-visible:ring-offset-1">
        Look up
      </button>
    </form>
  );
}

function SectionHeading({ children }: { children: string }) {
  return <h2 className="mb-3 text-sm font-semibold text-zinc-800">{children}</h2>;
}

export function CustomerDetailPage() {
  const params = useParams<{ customerId?: string }>();
  const customerId = params.customerId ? Number(params.customerId) : undefined;

  const risk = useCustomerRiskProfile(customerId);
  const history = useCustomerHistory(customerId);
  const latestRun = useLatestRun();
  const reasonLookup = useRecommendations(latestRun.data?.run_id, { customerId, limit: 1 });
  const reasonCodes = (reasonLookup.data?.items[0]?.reason_codes ?? "")
    .split("|")
    .map((s) => s.trim())
    .filter(Boolean);

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

  const utilization = risk.data && risk.data.current_limit > 0 ? risk.data.ead_current / risk.data.current_limit : undefined;

  return (
    <div className="space-y-8">
      <PageHeader
        title={`Customer ${customerId}`}
        subtitle={
          risk.data ? (
            <span className="flex flex-wrap items-center gap-2">
              <ActionBadge action={risk.data.action} />
              <span>
                {fmtCurrency(risk.data.current_limit)} current limit · PD {fmtPd(risk.data.pd_current)}
              </span>
            </span>
          ) : (
            "Risk profile, recommendation, and cross-run history"
          )
        }
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
            <SectionHeading>Credit profile</SectionHeading>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Kpi label="Current limit" value={fmtCurrency(risk.data.current_limit)} />
              <Kpi label="Current exposure (EAD)" value={fmtCurrency(risk.data.ead_current)} />
              <Kpi label="Utilization" value={utilization !== undefined ? `${(utilization * 100).toFixed(0)}%` : "—"} />
              <Kpi label="Probability of default" value={fmtPd(risk.data.pd_current)} tone={risk.data.pd_current > 0.2 ? "warning" : "neutral"} />
            </div>
          </section>

          <section>
            <SectionHeading>Decision</SectionHeading>
            <Card>
              <div className="flex flex-wrap items-center gap-6 p-4">
                <ActionBadge action={risk.data.action} />
                <div className="text-sm text-zinc-700">
                  <span className="text-zinc-500">Recommended limit: </span>
                  <span className="font-financial">{fmtCurrency(risk.data.current_limit)} → <strong>{fmtCurrency(risk.data.evaluated_limit)}</strong></span>
                  <span className="ml-2 font-financial text-xs text-zinc-400">
                    ({risk.data.evaluated_limit >= risk.data.current_limit ? "+" : ""}
                    {fmtCurrency(risk.data.evaluated_limit - risk.data.current_limit)})
                  </span>
                </div>
                <div className="font-financial text-sm text-zinc-700">
                  <span className="font-sans text-zinc-500">PD: </span>{fmtPd(risk.data.pd_current)} → {fmtPd(risk.data.pd_evaluated)}
                </div>
                <div className="font-financial text-sm font-medium text-[var(--color-positive)]">
                  <span className="font-sans font-normal text-zinc-500">EP uplift: </span>{fmtCurrency(risk.data.ep_uplift)}
                </div>
              </div>
            </Card>
          </section>

          <section>
            <SectionHeading>Risk drivers</SectionHeading>
            {reasonCodes.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {reasonCodes.map((code) => (
                  <span key={code} className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs text-zinc-700 shadow-sm">
                    {code}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">
                {reasonLookup.isLoading ? "Loading reason codes…" : "No SHAP-derived reason codes synced for this customer's latest run."}
              </p>
            )}
          </section>
        </>
      )}

      <section>
        <SectionHeading>History</SectionHeading>
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
          <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-3 py-2.5 font-medium">Run</th>
                  <th className="px-3 py-2.5 font-medium">Action</th>
                  <th className="px-3 py-2.5 font-medium">Limit</th>
                  <th className="px-3 py-2.5 font-medium">PD</th>
                  <th className="px-3 py-2.5 font-medium">EP uplift</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {history.data.map((h) => (
                  <tr key={h.run_id}>
                    <td className="px-3 py-2.5">
                      <Link className="text-xs text-zinc-500 hover:underline" to={`/runs/${h.run_id}`}>
                        {fmtDateTime(h.started_at)}
                      </Link>
                    </td>
                    <td className="px-3 py-2.5"><ActionBadge action={h.action} /></td>
                    <td className="px-3 py-2.5 font-financial text-zinc-600">
                      {fmtCurrency(h.current_limit)} → {fmtCurrency(h.recommended_limit)}
                    </td>
                    <td className="px-3 py-2.5 font-financial text-zinc-600">
                      {fmtPd(h.pd_current)} → {fmtPd(h.pd_recommended)}
                    </td>
                    <td className="px-3 py-2.5 font-financial text-zinc-600">{fmtCurrency(h.ep_uplift)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <SectionHeading>What-if analysis</SectionHeading>
        <p className="mb-3 text-xs text-zinc-500">
          Live, against the currently loaded models — not the database. Set a hypothetical limit to see exactly
          how the guardrails and economics respond; this is deliberate risk analysis, not a preview of what will
          be applied.
        </p>
        <div className="flex gap-2">
          <input
            className="w-56 rounded-md border border-zinc-300 px-3 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
            placeholder="Hypothetical new limit"
            value={newLimit}
            onChange={(e) => setNewLimit(e.target.value.replace(/\D/g, ""))}
            aria-label="Hypothetical new limit"
          />
          <button
            className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)] focus-visible:ring-offset-1"
            disabled={whatIf.isPending || !newLimit}
            onClick={() => whatIf.mutate({ new_limit: Number(newLimit) })}
          >
            {whatIf.isPending ? "Scoring…" : "Score this limit"}
          </button>
        </div>

        {whatIf.isError && <div className="mt-3"><ErrorState error={whatIf.error} fallback="Failed to score this scenario." /></div>}

        {whatIf.data && (
          <div className="mt-3 space-y-3 rounded-lg border border-[var(--color-brand-600)]/30 bg-[var(--color-brand-50)]/40 p-4 shadow-sm">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs uppercase tracking-wide text-zinc-400">Hypothetical input</span>
              <span className="font-financial text-sm text-zinc-700">{fmtCurrency(whatIf.data.evaluated_limit)}</span>
              <span className="mx-2 text-zinc-300">|</span>
              <span className="text-xs uppercase tracking-wide text-zinc-400">Model result</span>
              <ActionBadge action={whatIf.data.action} />
              {whatIf.data.guardrail_blocked && (
                <StatusBadge tone="warning">guardrail would block this increase</StatusBadge>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2 font-financial text-sm text-zinc-600 sm:grid-cols-4">
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
