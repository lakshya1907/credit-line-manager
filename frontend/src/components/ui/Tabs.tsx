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
    <div role="tablist" className="flex flex-wrap gap-1 border-b border-zinc-200">
      {items.map((item) => {
        const active = item.id === activeId;
        return (
          <button
            key={item.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-brand-500)] focus-visible:ring-offset-1 ${
              active
                ? "border-[var(--color-brand-600)] text-zinc-900"
                : "border-transparent text-zinc-500 hover:text-zinc-800"
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
