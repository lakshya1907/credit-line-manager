import { Link } from "react-router-dom";
import { useLatestRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { PolicyComparisonSection } from "../components/sections/PolicyComparisonSection";
import { fmtDateTime } from "../lib/format";

export function PolicyComparisonPage() {
  const { data: run, isLoading, error, hasNoRuns } = useLatestRun();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Policy Comparison"
        subtitle={run ? (
          <>
            Pre-computed guardrail/budget scenarios for <Link className="underline" to={`/runs/${run.run_id}`}>{run.run_id}</Link> ·{" "}
            {fmtDateTime(run.started_at)} — read-only; see Run Detail for other historical runs.
          </>
        ) : "Pre-computed guardrail/budget scenarios for the latest run"}
      />

      {isLoading && <SkeletonTable />}
      {error && <ErrorState error={error} fallback="Could not load policy comparison." />}
      {hasNoRuns && <EmptyState title="No model runs yet" hint="Trigger a run from the Runs page." />}
      {run && <PolicyComparisonSection run={run} />}
    </div>
  );
}
