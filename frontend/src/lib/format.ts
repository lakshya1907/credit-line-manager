export function fmtCurrency(v: number): string {
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

export function fmtPercent(v: number, decimals = 1): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

/** PD is a 0-1 probability but small (usually <0.5) and decision-relevant
 * at the 3rd decimal -- percent formatting loses precision analysts need
 * ("14.9%" vs "15.0%" reads as equal-ish; "0.149" vs "0.150" doesn't). */
export function fmtPd(v: number): string {
  return v.toFixed(3);
}

export function fmtNumber(v: number): string {
  return v.toLocaleString();
}
