// Thin fetch wrapper. VITE_API_URL points straight at the FastAPI
// instance in production; in dev, relative /api/* paths go through
// vite.config.ts's proxy instead, so no CORS setup is needed locally.
const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | undefined>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined);
  if (entries.length === 0) return "";
  return "?" + new URLSearchParams(entries as [string, string][]).toString();
}

import type {
  CustomerHistoryEntry, CustomerSamplePoint, JobStatus, ModelRunDetail, ModelRunSummary,
  PaginatedRecommendations, PortfolioDistributions, ReadinessResponse, ScoreRequest, ScoreResponse,
} from "./types";

export const api = {
  readiness: () => request<ReadinessResponse>("/readiness"),

  listRuns: (limit = 20, offset = 0) =>
    request<ModelRunSummary[]>(`/runs${qs({ limit, offset })}`),

  getRun: (runId: string) => request<ModelRunDetail>(`/runs/${runId}`),

  getRecommendations: (
    runId: string,
    opts: { action?: string; customerId?: number; limit?: number; offset?: number } = {},
  ) =>
    request<PaginatedRecommendations>(
      `/runs/${runId}/recommendations${qs({
        action: opts.action,
        customer_id: opts.customerId,
        limit: opts.limit ?? 25,
        offset: opts.offset ?? 0,
      })}`,
    ),

  getDistributions: (runId: string) =>
    request<PortfolioDistributions>(`/runs/${runId}/distributions`),

  getCustomerSample: (runId: string, n = 1500) =>
    request<CustomerSamplePoint[]>(`/runs/${runId}/customer-sample${qs({ n })}`),

  triggerRun: () => request<JobStatus>("/runs", { method: "POST" }),

  getJob: (jobId: string) => request<JobStatus>(`/runs/jobs/${jobId}`),

  getCustomerHistory: (customerId: number) =>
    request<CustomerHistoryEntry[]>(`/customers/${customerId}/history`),

  scoreCustomer: (customerId: number, body: ScoreRequest) =>
    request<ScoreResponse>(`/customers/${customerId}/score`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
