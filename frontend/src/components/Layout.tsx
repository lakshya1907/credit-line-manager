import { NavLink, Outlet } from "react-router-dom";
import { useReadiness } from "../api/hooks";

const navItems = [
  { to: "/", label: "Runs" },
  { to: "/customers", label: "Customer Lookup" },
];

function ReadinessDot() {
  const { data, isLoading } = useReadiness();
  const color = isLoading
    ? "bg-slate-300"
    : data?.status === "ok"
      ? "bg-green-500"
      : "bg-amber-500";
  const title = isLoading
    ? "Checking API..."
    : `API: ${data?.status} · DB: ${data?.database} · models: ${data?.models_loaded ? "loaded" : "not loaded"}`;
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} title={title} />;
}

export function Layout() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="text-sm font-semibold text-slate-900">Credit Line Manager</span>
          <nav className="flex gap-4 text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded px-2 py-1 ${isActive ? "bg-slate-100 font-medium text-slate-900" : "text-slate-500 hover:text-slate-800"}`
                }
                end={item.to === "/"}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
            <ReadinessDot />
            API status
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
