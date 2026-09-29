import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type { ScoreRequest } from "./types";

export function useReadiness() {
  return useQuery({ queryKey: ["readiness"], queryFn: api.readiness, refetchInterval: 30_000 });
}

export function useRuns(limit = 20) {
  return useQuery({ queryKey: ["runs", limit], queryFn: () => api.listRuns(limit) });
}

/** The Overview page's data source: the most recently started run, in
 * full detail (KPIs + exposure + segments + fairness + stress). Two
 * requests (list then detail) rather than a new "latest" endpoint --
 * GET /runs already returns newest-first, so this is a thin composition,
 * not a missing API capability. */
export function useLatestRun() {
  const list = useRuns(1);
  const latestId = list.data?.[0]?.run_id;
  const detail = useRun(latestId);
  return {
    ...detail,
    isLoading: list.isLoading || (!!latestId && detail.isLoading),
    hasNoRuns: list.isSuccess && list.data.length === 0,
  };
}

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId as string),
    enabled: !!runId,
  });
}

export function useRecommendations(
  runId: string | undefined,
  opts: { action?: string; customerId?: number; limit?: number; offset?: number },
) {
  return useQuery({
    queryKey: ["recommendations", runId, opts],
    queryFn: () => api.getRecommendations(runId as string, opts),
    enabled: !!runId,
  });
}

export function useDistributions(runId: string | undefined) {
  return useQuery({
    queryKey: ["distributions", runId],
    queryFn: () => api.getDistributions(runId as string),
    enabled: !!runId,
  });
}

export function useCustomerSample(runId: string | undefined, n = 1500) {
  return useQuery({
    queryKey: ["customer-sample", runId, n],
    queryFn: () => api.getCustomerSample(runId as string, n),
    enabled: !!runId,
    staleTime: 60_000, // a random sample refreshing mid-session would visibly jitter scatter plots for no benefit
  });
}

export function useCustomerHistory(customerId: number | undefined) {
  return useQuery({
    queryKey: ["customer-history", customerId],
    queryFn: () => api.getCustomerHistory(customerId as number),
    enabled: !!customerId && customerId > 0,
    retry: false,
  });
}

export function useScoreCustomer(customerId: number | undefined) {
  return useMutation({
    mutationFn: (body: ScoreRequest) => api.scoreCustomer(customerId as number, body),
  });
}

/** The baseline (no new_limit) score, as a read-only query rather than a
 * mutation -- POST /customers/{id}/score has no side effects (nothing is
 * persisted; see its docstring), so treating the default best-candidate
 * scenario as cacheable query data is legitimate and lets the Customer
 * Detail page show a risk profile immediately on load, not only after an
 * explicit "Score" click. The what-if form (useScoreCustomer above)
 * covers the explicit-limit case. */
export function useCustomerRiskProfile(customerId: number | undefined) {
  return useQuery({
    queryKey: ["customer-risk-profile", customerId],
    queryFn: () => api.scoreCustomer(customerId as number, {}),
    enabled: !!customerId && customerId > 0,
    retry: false,
  });
}

/** Polls a triggered pipeline run's job until it completes/fails, then
 * invalidates the runs list so the new run shows up without a manual
 * refresh. */
export function useTriggerRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const job = await api.triggerRun();
      return job;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useJobPolling(jobId: string | undefined, enabled: boolean) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId as string),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") {
        queryClient.invalidateQueries({ queryKey: ["runs"] });
        return false;
      }
      return 1500;
    },
  });
}
