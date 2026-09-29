import { useEffect, useState, type ReactNode } from "react";

/**
 * Recharts' ResponsiveContainer measures its parent via ResizeObserver the
 * instant it mounts. For a data-driven chart, that's always right after
 * async data resolves -- and the surrounding layout (KPI cards, headers
 * rendering their own numbers for the first time) can still be settling
 * at that exact moment, with nothing to trigger a remeasure afterward.
 *
 * Found by actually clicking through the app, not by reading the code:
 * charts came up as empty axes (no bars) on both a fresh page load and
 * after SPA navigation, verified against correct underlying API data both
 * times via direct curl/SQL checks -- a rendering-timing bug, not a data
 * bug. A route-change-triggered `window.dispatchEvent(new
 * Event("resize"))` fixed the SPA-navigation case but not the fresh-load
 * case, because the real trigger is "this chart's own data just arrived
 * and its DOM subtree just mounted", not "the route changed".
 *
 * Fix: delay the chart's own mount by two animation frames after its
 * parent commits, so by the time ResponsiveContainer takes its first
 * measurement, the layout above it (which mounted in the same commit) has
 * already had a paint cycle to settle. Every chart in this app goes
 * through this wrapper rather than a bare ResponsiveContainer.
 *
 * Runs once per mount (not on every re-render -- re-running this on
 * every `children` change would flicker the chart back to its skeleton
 * on any parent re-render, e.g. an unrelated query refetch). Callers
 * that need a genuine remount when the underlying dataset changes (e.g.
 * Overview switching to a newly triggered run) should pass `key={...}`
 * at the call site -- React's normal remount-on-key-change already gives
 * a fresh `ready` cycle for free, no extra logic needed here.
 */
export function ChartContainer({ height, children }: { height: string; children: ReactNode }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const raf1 = requestAnimationFrame(() => {
      requestAnimationFrame(() => setReady(true));
    });
    return () => cancelAnimationFrame(raf1);
  }, []);

  return (
    <div className={height}>
      {ready ? children : <div className="h-full w-full animate-pulse rounded bg-slate-100" />}
    </div>
  );
}
