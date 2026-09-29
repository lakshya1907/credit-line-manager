import { NavLink, Outlet } from "react-router-dom";
import { useReadiness } from "../api/hooks";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/runs", label: "Runs" },
  { to: "/customers", label: "Customer Lookup" },
];

function ReadinessDot() {
  const { data, isLoading } = useReadiness();
  const color = isLoading
    ? "bg-slate-300"
    : data?.status === "ok"
      ? "bg-green-500"
      : "bg-amber-500";
  const label = isLoading
    ? "Checking API status"
    : `API ${data?.status}, database ${data?.database}, models ${data?.models_loaded ? "loaded" : "not loaded"}`;
  return (
    <span className="flex items-center gap-1.5" role="status" aria-label={label} title={label}>
      <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} aria-hidden="true" />
      API status
    </span>
  );
}

export function Layout() {
  return (
    <div className="min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-slate-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
      >
        Skip to content
      </a>
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="text-sm font-semibold text-slate-900">Credit Line Manager</span>
          <nav aria-label="Primary" className="flex gap-4 text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded px-2 py-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-1 ${
                    isActive ? "bg-slate-100 font-medium text-slate-900" : "text-slate-500 hover:text-slate-800"
                  }`
                }
                end={item.to === "/"}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto text-xs text-slate-400">
            <ReadinessDot />
          </div>
        </div>
      </header>
      <main id="main-content" className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
