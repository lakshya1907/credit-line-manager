import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { useRecommendations } from "../api/hooks";
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
        className="font-medium text-slate-900 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 rounded"
        to={`/customers/${c.getValue()}`}
      >
        {c.getValue()}
      </Link>
    ),
  }),
  columnHelper.accessor("action", { header: "Action", cell: (c) => <ActionBadge action={c.getValue()} /> }),
  columnHelper.accessor("current_limit", { header: "Current limit", cell: (c) => fmtCurrency(c.getValue()) }),
  columnHelper.accessor("recommended_limit", { header: "Recommended limit", cell: (c) => fmtCurrency(c.getValue()) }),
  columnHelper.accessor("pd_current", { header: "PD (current)", cell: (c) => fmtPd(c.getValue()) }),
  columnHelper.accessor("pd_recommended", { header: "PD (recommended)", cell: (c) => fmtPd(c.getValue()) }),
  columnHelper.accessor("ep_uplift", {
    header: "EP uplift",
    cell: (c) => (
      <span className={c.getValue() >= 0 ? "text-green-700" : "text-red-700"}>{fmtCurrency(c.getValue())}</span>
    ),
  }),
  columnHelper.accessor("reason_codes", {
    header: "Reason codes",
    cell: (c) => (
      <span className="block max-w-[16rem] truncate text-xs text-slate-500" title={c.getValue() ?? undefined}>
        {c.getValue() ?? "—"}
      </span>
    ),
  }),
];

const PAGE_SIZE = 25;

export function ActionQueuePage() {
  const { runId } = useParams<{ runId: string }>();
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
        subtitle={<Link className="text-slate-500 hover:underline" to={`/runs/${runId}`}>← Back to run {runId}</Link>}
      />

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <div className="mb-1 text-xs font-medium text-slate-500">Action</div>
          <select
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
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
          <div className="mb-1 text-xs font-medium text-slate-500">Customer ID</div>
          <input
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
            placeholder="e.g. 12"
            value={customerId}
            onChange={(e) => {
              setCustomerId(e.target.value.replace(/\D/g, ""));
              setPage(0);
            }}
          />
        </label>
        {data && <div className="text-sm text-slate-500">{data.total.toLocaleString()} matching customers</div>}
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
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((h) => (
                      <th key={h.id} className="whitespace-nowrap px-3 py-2">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody className="divide-y divide-slate-100">
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-50">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="whitespace-nowrap px-3 py-2 text-slate-700">
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
              className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </button>
            <span className="text-slate-500">
              Page {page + 1} of {Math.max(totalPages, 1)}
            </span>
            <button
              className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900"
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
