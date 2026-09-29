import { Link } from "react-router-dom";
import { useLatestRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { SkeletonKpiRow, SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { StressTestSection } from "../components/sections/StressTestSection";
import { fmtDateTime } from "../lib/format";

export function StressTestingPage() {
  const { data: run, isLoading, error, hasNoRuns } = useLatestRun();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Stress Testing"
        subtitle={run ? (
          <>
            Shocked-PD re-simulation for <Link className="underline" to={`/runs/${run.run_id}`}>{run.run_id}</Link> ·{" "}
            {fmtDateTime(run.started_at)}
          </>
        ) : "Shocked-PD re-simulation for the latest run"}
      />

      {isLoading && (
        <div className="space-y-6">
          <SkeletonKpiRow />
          <SkeletonTable />
        </div>
      )}
      {error && <ErrorState error={error} fallback="Could not load stress test results." />}
      {hasNoRuns && <EmptyState title="No model runs yet" hint="Trigger a run from the Runs page." />}
      {run && <StressTestSection run={run} />}
    </div>
  );
}
