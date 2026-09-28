import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { useRecommendations } from "../api/hooks";
import { ActionBadge } from "../components/ActionBadge";
import { fmtCurrency } from "../lib/format";
import type { Recommendation } from "../api/types";

const columnHelper = createColumnHelper<Recommendation>();

const columns = [
  columnHelper.accessor("customer_id", { header: "Customer" }),
  columnHelper.accessor("action", { header: "Action", cell: (c) => <ActionBadge action={c.getValue()} /> }),
  columnHelper.accessor("current_limit", { header: "Current limit", cell: (c) => fmtCurrency(c.getValue()) }),
  columnHelper.accessor("recommended_limit", { header: "Recommended limit", cell: (c) => fmtCurrency(c.getValue()) }),
  columnHelper.accessor("pd_current", { header: "PD (current)", cell: (c) => c.getValue().toFixed(3) }),
  columnHelper.accessor("pd_recommended", { header: "PD (recommended)", cell: (c) => c.getValue().toFixed(3) }),
  columnHelper.accessor("ep_uplift", { header: "EP uplift", cell: (c) => fmtCurrency(c.getValue()) }),
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

  return (
    <div className="space-y-4">
      <div>
        <Link className="text-sm text-slate-500 hover:underline" to={`/runs/${runId}`}>
          ← Back to run {runId}
        </Link>
        <h1 className="text-lg font-semibold text-slate-900">Action Queue</h1>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <div className="mb-1 text-xs font-medium text-slate-500">Action</div>
          <select
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
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
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
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

      {isLoading && <p className="text-sm text-slate-500">Loading…</p>}
      {error && <p className="text-sm text-red-600">{(error as Error).message}</p>}

      {data && (
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
                {table.getRowModel().rows.length === 0 && (
                  <tr>
                    <td colSpan={columns.length} className="px-3 py-6 text-center text-slate-400">
                      No matching recommendations.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between text-sm">
            <button
              className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </button>
            <span className="text-slate-500">
              Page {page + 1} of {Math.max(totalPages, 1)}
            </span>
            <button
              className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40"
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
