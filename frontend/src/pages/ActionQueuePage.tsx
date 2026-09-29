import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { useLatestRun, useRecommendations } from "../api/hooks";
import { ActionBadge } from "../components/ActionBadge";
import { PageHeader } from "../components/ui/PageHeader";
import { SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { fmtCurrency, fmtPd } from "../lib/format";
import type { Recommendation } from "../api/types";

const columnHelper = createColumnHelper<Recommendation>();

const columns = [
  columnHelper.accessor("customer_id", {
    header: "Customer",
    cell: (c) => (
      <Link
        className="font-financial font-medium text-zinc-900 hover:text-[var(--color-brand-600)] hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)] rounded"
        to={`/customers/${c.getValue()}`}
      >
        {c.getValue()}
      </Link>
    ),
  }),
  columnHelper.accessor("action", { header: "Action", cell: (c) => <ActionBadge action={c.getValue()} /> }),
  columnHelper.accessor("current_limit", {
    header: "Current limit",
    cell: (c) => <span className="font-financial tabular-nums text-zinc-700">{fmtCurrency(c.getValue())}</span>,
  }),
  columnHelper.accessor("recommended_limit", {
    header: "Recommended limit",
    cell: (c) => <span className="font-financial tabular-nums font-medium text-zinc-900">{fmtCurrency(c.getValue())}</span>,
  }),
  columnHelper.display({
    id: "change",
    header: "Change",
    cell: (c) => {
      const delta = c.row.original.recommended_limit - c.row.original.current_limit;
      const tone = delta > 0 ? "text-[var(--color-positive)]" : delta < 0 ? "text-[var(--color-negative)]" : "text-zinc-400";
      return (
        <span className={`font-financial tabular-nums ${tone}`}>
          {delta === 0 ? "—" : `${delta > 0 ? "+" : ""}${fmtCurrency(delta)}`}
        </span>
      );
    },
  }),
  columnHelper.accessor("pd_current", {
    header: "PD",
    cell: (c) => <span className="font-financial tabular-nums text-zinc-600">{fmtPd(c.getValue())}</span>,
  }),
  columnHelper.accessor("ead_current", {
    header: "EAD",
    cell: (c) => <span className="font-financial tabular-nums text-zinc-600">{fmtCurrency(c.getValue())}</span>,
  }),
  columnHelper.accessor("ep_uplift", {
    header: "EP uplift",
    cell: (c) => (
      <span className={`font-financial tabular-nums font-medium ${c.getValue() >= 0 ? "text-[var(--color-positive)]" : "text-[var(--color-negative)]"}`}>
        {fmtCurrency(c.getValue())}
      </span>
    ),
  }),
  columnHelper.accessor("reason_codes", {
    header: "Reason codes",
    cell: (c) => (
      <span className="block max-w-[16rem] truncate text-xs text-zinc-500" title={c.getValue() ?? undefined}>
        {c.getValue() ?? "—"}
      </span>
    ),
  }),
];

const PAGE_SIZE = 25;

export function ActionQueuePage() {
  const { runId: routeRunId } = useParams<{ runId: string }>();
  const latest = useLatestRun();
  const runId = routeRunId ?? latest.data?.run_id;

  const [action, setAction] = useState<string>("");
  const [customerId, setCustomerId] = useState<string>("");
  const [page, setPage] = useState(0);

  const { data, isLoading, error } = useRecommendations(runId, {
    action: action || undefined,
    customerId: customerId ? Number(customerId) : undefined,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  });

  const table = useReactTable({
    data: data?.items ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;
  const hasFilters = !!action || !!customerId;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Action Queue"
        subtitle={runId ? <Link className="text-zinc-500 hover:underline" to={`/runs/${runId}`}>Run {runId}</Link> : "Awaiting a run"}
      />

      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-zinc-200 bg-white p-3 shadow-sm">
        <label className="text-sm">
          <div className="mb-1 text-xs font-medium text-zinc-500">Action</div>
          <select
            className="rounded-md border border-zinc-300 px-2 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
            value={action}
            onChange={(e) => {
              setAction(e.target.value);
              setPage(0);
            }}
          >
            <option value="">All</option>
            <option value="increase">Increase</option>
            <option value="decrease">Decrease</option>
            <option value="hold">Hold</option>
          </select>
        </label>
        <label className="text-sm">
          <div className="mb-1 text-xs font-medium text-zinc-500">Customer ID</div>
          <input
            className="rounded-md border border-zinc-300 px-2 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
            placeholder="e.g. 12"
            value={customerId}
            onChange={(e) => {
              setCustomerId(e.target.value.replace(/\D/g, ""));
              setPage(0);
            }}
          />
        </label>
        {data && <div className="ml-auto text-sm text-zinc-500">{data.total.toLocaleString()} matching customers</div>}
      </div>

      {isLoading && <SkeletonTable />}
      {error && <ErrorState error={error} fallback="Could not load recommendations for this run." />}

      {data && data.items.length === 0 && (
        <EmptyState
          title="No recommendations match these filters"
          hint={hasFilters ? "Try clearing the action or customer ID filter." : "This run has no recommendations synced."}
        />
      )}

      {data && data.items.length > 0 && (
        <>
          <div className="overflow-x-auto rounded-lg border border-zinc-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10 bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500">
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((h) => (
                      <th key={h.id} className="whitespace-nowrap px-3 py-2.5 font-medium">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id} className="hover:bg-zinc-50">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="whitespace-nowrap px-3 py-2.5 text-zinc-700">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between text-sm">
            <button
              className="rounded-md border border-zinc-300 px-3 py-1 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </button>
            <span className="text-zinc-500">
              Page {page + 1} of {Math.max(totalPages, 1)}
            </span>
            <button
              className="rounded-md border border-zinc-300 px-3 py-1 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)]"
              disabled={page + 1 >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
