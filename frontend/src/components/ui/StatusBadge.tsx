import type { ReactNode } from "react";

const TONE_STYLES = {
  success: "bg-[var(--color-positive-soft)] text-[var(--color-positive)]",
  danger: "bg-[var(--color-negative-soft)] text-[var(--color-negative)]",
  warning: "bg-[var(--color-warning-soft)] text-[var(--color-warning)]",
  neutral: "bg-zinc-200 text-zinc-700",
  info: "bg-[var(--color-brand-50)] text-[var(--color-brand-700)]",
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
