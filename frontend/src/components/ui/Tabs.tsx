import type { ReactNode } from "react";

export interface TabItem {
  id: string;
  label: string;
  badge?: ReactNode;
}

/** Plain tab buttons, not a routed/ARIA-tabs library -- native buttons
 * are keyboard/focus-accessible by default, which is all this needs. */
export function Tabs({
  items, activeId, onChange,
}: { items: TabItem[]; activeId: string; onChange: (id: string) => void }) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 border-b border-slate-200">
      {items.map((item) => {
        const active = item.id === activeId;
        return (
          <button
            key={item.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-1 ${
              active
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            {item.label}
            {item.badge}
          </button>
        );
      })}
    </div>
  );
}
