import type { ReactNode } from "react";

const TONE_STYLES = {
  success: "bg-green-100 text-green-800",
  danger: "bg-red-100 text-red-800",
  warning: "bg-amber-100 text-amber-800",
  neutral: "bg-slate-200 text-slate-700",
  info: "bg-blue-100 text-blue-800",
} as const;

export type Tone = keyof typeof TONE_STYLES;

/** Every tone pairs color with a text label -- never color alone, so this
 * stays legible without relying on color perception (see CLAUDE.md's
 * frontend accessibility notes). */
export function StatusBadge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${TONE_STYLES[tone]}`}>
      {children}
    </span>
  );
}
