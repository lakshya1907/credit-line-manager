import type { Action } from "../api/types";

const STYLES: Record<Action, string> = {
  increase: "bg-green-100 text-green-800",
  decrease: "bg-red-100 text-red-800",
  hold: "bg-slate-200 text-slate-700",
};

export function ActionBadge({ action }: { action: Action }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[action]}`}>
      {action}
    </span>
  );
}
