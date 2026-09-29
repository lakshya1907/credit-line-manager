// Mirrors src/api/schemas.py one-for-one. Keep in sync by hand for now;
// see CLAUDE.md's Frontend section for the note on generating these from
// FastAPI's OpenAPI schema instead, once the API surface stabilizes.

export type Action = "increase" | "decrease" | "hold";

export interface PortfolioRun {
  policy_name: string;
  el_budget: number;
  ead_budget: number;
  pd_increase_max: number;
  pd_decrease_min: number;
  used_el: number;
  used_ead: number;
  n_increase_applied: number;
  n_decrease: number;
  n_hold: number;
  total_ep_uplift: number;
}

export interface StressTestResult {
  pd_shock: string;
  n_increase: number;
  n_decrease: number;
  n_hold: number;
  total_ep_uplift: number;
  el_used: number;
  el_budget_pct: number;
}

export interface SegmentMetric {
  segment: string;
  segment_value: string;
  n_customers: number;
  increase_rate: number;
  decrease_rate: number;
  hold_rate: number;
  avg_pd_current: number;
  avg_ep_uplift: number;
  total_ep_uplift: number;
}

export interface BacktestResult {
  window_months: number;
  n_features: number;
  val_roc_auc: number;
  val_pr_auc: number;
}

export interface FairnessCheck {
  segment: string;
  min_max_approval_ratio: number;
  flag: string;
}

export interface DistributionBucket {
  bucket: string;
  count: number;
}

export interface PortfolioDistributions {
  risk_distribution: DistributionBucket[];
  utilization_distribution: DistributionBucket[];
  limit_change_distribution: DistributionBucket[];
}

export interface CustomerSamplePoint {
  customer_id: number;
  pd_current: number;
  utilization: number | null;
  current_limit: number;
  recommended_limit: number;
  ep_uplift: number;
  action: Action;
}

export interface ExposureSummary {
  total_current_limit: number;
  total_recommended_limit: number;
  total_current_ead: number;
  total_recommended_ead: number;
}

export interface ModelRunSummary {
  run_id: string;
  started_at: string;
  finished_at: string;
  wall_time_seconds: number;
  git_commit: string | null;
  pd_roc_auc: number;
  pd_pr_auc: number;
  ead_mae: number;
}

export interface ModelRunDetail extends ModelRunSummary {
  config: Record<string, unknown>;
  exposure_summary: ExposureSummary;
  portfolio_runs: PortfolioRun[];
  stress_test_results: StressTestResult[];
  segment_metrics: SegmentMetric[];
  backtest_results: BacktestResult[];
  fairness_checks: FairnessCheck[];
}

export interface Recommendation {
  customer_id: number;
  current_limit: number;
  recommended_limit: number;
  action: Action;
  pd_current: number;
  pd_recommended: number;
  ead_current: number;
  ead_recommended: number;
  ep_uplift: number;
  el_uplift_proxy: number;
  ead_uplift: number;
  reason_codes: string | null;
}

export interface PaginatedRecommendations {
  total: number;
  limit: number;
  offset: number;
  items: Recommendation[];
}

export interface CustomerHistoryEntry {
  run_id: string;
  started_at: string;
  action: Action;
  current_limit: number;
  recommended_limit: number;
  pd_current: number;
  pd_recommended: number;
  ep_uplift: number;
}

export interface ScoreRequest {
  new_limit?: number;
}

export interface ScoreResponse {
  customer_id: number;
  current_limit: number;
  evaluated_limit: number;
  action: Action;
  guardrail_blocked: boolean;
  pd_current: number;
  pd_evaluated: number;
  ead_current: number;
  ead_evaluated: number;
  ep_current: number;
  ep_evaluated: number;
  ep_uplift: number;
}

export interface JobStatus {
  job_id: string;
  status: "running" | "completed" | "failed";
  run_id: string | null;
  error: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface ReadinessResponse {
  status: "ok" | "degraded";
  database: "ok" | "unreachable";
  models_loaded: boolean;
}
