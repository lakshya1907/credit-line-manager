import { NavLink, Outlet } from "react-router-dom";
import { useLatestRun, useReadiness } from "../api/hooks";
import { fmtDateTime } from "../lib/format";

const NAV_GROUPS: { label: string; items: { to: string; label: string }[] }[] = [
  {
    label: "Portfolio",
    items: [
      { to: "/", label: "Overview" },
      { to: "/action-queue", label: "Action Queue" },
      { to: "/customers", label: "Customers" },
    ],
  },
  {
    label: "Analytics",
    items: [
      { to: "/analytics", label: "Analytics" },
      { to: "/stress-testing", label: "Stress Testing" },
      { to: "/policy-comparison", label: "Policy Comparison" },
    ],
  },
  {
    label: "Operations",
    items: [{ to: "/runs", label: "Runs" }],
  },
];

function ReadinessDot() {
  const { data, isLoading } = useReadiness();
  const color = isLoading ? "bg-zinc-600" : data?.status === "ok" ? "bg-[var(--color-positive)]" : "bg-[var(--color-warning)]";
  const label = isLoading
    ? "Checking API status"
    : `API ${data?.status}, database ${data?.database}, models ${data?.models_loaded ? "loaded" : "not loaded"}`;
  return (
    <span className="flex items-center gap-1.5" role="status" aria-label={label} title={label}>
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} aria-hidden="true" />
      <span className="text-xs text-zinc-400">{isLoading ? "Checking…" : data?.status === "ok" ? "Operational" : "Degraded"}</span>
    </span>
  );
}

function RunContext() {
  const { data: run, isLoading } = useLatestRun();
  if (isLoading) return <div className="text-xs text-zinc-500">Loading run…</div>;
  if (!run) return <div className="text-xs text-zinc-500">No runs yet</div>;
  return (
    <div className="space-y-0.5">
      <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Current run</div>
      <div className="truncate font-financial text-xs text-zinc-300" title={run.run_id}>
        {run.run_id}
      </div>
      <div className="text-[11px] text-zinc-500">{fmtDateTime(run.started_at)}</div>
    </div>
  );
}

const ALL_NAV_ITEMS = NAV_GROUPS.flatMap((g) => g.items);

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-zinc-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
      >
        Skip to content
      </a>

      {/* Mobile top bar: full sidebar collapses to a horizontally scrollable
       * nav strip below md -- a fixed 240px rail has no room on a phone
       * viewport. */}
      <header className="flex items-center gap-3 border-b border-zinc-800 bg-zinc-950 px-4 py-3 text-zinc-100 md:hidden">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-[var(--color-brand-600)] text-xs font-bold text-white">
          C
        </span>
        <span className="shrink-0 text-sm font-semibold tracking-tight text-white">Credit Line Manager</span>
        <nav aria-label="Primary" className="ml-2 flex flex-1 gap-3 overflow-x-auto text-sm">
          {ALL_NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => `shrink-0 whitespace-nowrap rounded px-1 py-0.5 ${isActive ? "font-medium text-white" : "text-zinc-400"}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <aside className="hidden w-60 shrink-0 flex-col bg-zinc-950 text-zinc-100 md:flex">
        <div className="flex items-center gap-2 border-b border-zinc-800 px-5 py-4">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-[var(--color-brand-600)] text-xs font-bold text-white">
            C
          </span>
          <span className="text-sm font-semibold tracking-tight text-white">Credit Line Manager</span>
        </div>

        <nav aria-label="Primary" className="flex-1 space-y-6 overflow-y-auto px-3 py-5">
          {NAV_GROUPS.map((group) => (
            <div key={group.label}>
              <div className="mb-1.5 px-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                {group.label}
              </div>
              <div className="space-y-0.5">
                {group.items.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === "/"}
                    className={({ isActive }) =>
                      `block rounded-md px-2.5 py-1.5 text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)] ${
                        isActive
                          ? "bg-zinc-800/80 font-medium text-white"
                          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="space-y-3 border-t border-zinc-800 px-4 py-4">
          <RunContext />
          <ReadinessDot />
        </div>
      </aside>

      <main id="main-content" className="min-w-0 flex-1 overflow-x-hidden bg-zinc-50 px-4 py-5 md:px-8 md:py-7">
        <div className="mx-auto max-w-6xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
