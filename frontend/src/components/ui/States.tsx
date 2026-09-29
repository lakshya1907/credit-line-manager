import type { ReactNode } from "react";
import { ApiError } from "../../api/client";

/** A handful of skeleton block shapes, not a generic skeleton framework --
 * enough to cover this app's KPI rows/cards/tables without a dependency. */
export function SkeletonBlock({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-slate-200 ${className}`} />;
}

export function SkeletonKpiRow({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <SkeletonBlock className="h-3 w-20" />
          <SkeletonBlock className="mt-2 h-7 w-24" />
        </div>
      ))}
    </div>
  );
}

export function SkeletonTable({ rows = 6 }: { rows?: number }) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="space-y-3 p-4">
        {Array.from({ length: rows }).map((_, i) => (
          <SkeletonBlock key={i} className="h-4 w-full" />
        ))}
      </div>
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white px-6 py-10 text-center">
      <p className="text-sm font-medium text-slate-700">{title}</p>
      {hint && <p className="mt-1 text-sm text-slate-500">{hint}</p>}
    </div>
  );
}

/** Human-readable, never a raw stack trace -- ApiError carries the FastAPI
 * `detail` message; anything else falls back to a generic message rather
 * than printing exception internals. */
export function ErrorState({ error, fallback = "Something went wrong loading this data." }: { error: unknown; fallback?: string }) {
  const message = error instanceof ApiError ? error.message : fallback;
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      {message}
    </div>
  );
}
