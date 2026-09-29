import { Link } from "react-router-dom";
import { useLatestRun } from "../api/hooks";
import { PageHeader } from "../components/ui/PageHeader";
import { SkeletonKpiRow, SkeletonTable, EmptyState, ErrorState } from "../components/ui/States";
import { AnalyticsSection } from "../components/sections/AnalyticsSection";
import { fmtDateTime } from "../lib/format";

export function AnalyticsPage() {
  const { data: run, isLoading, error, hasNoRuns } = useLatestRun();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analytics"
        subtitle={run ? (
          <>
            Segmentation and model quality for <Link className="underline" to={`/runs/${run.run_id}`}>{run.run_id}</Link> ·{" "}
            {fmtDateTime(run.started_at)}
          </>
        ) : "Segmentation and model quality for the latest run"}
      />

      {isLoading && (
        <div className="space-y-6">
          <SkeletonKpiRow />
          <SkeletonTable />
        </div>
      )}
      {error && <ErrorState error={error} fallback="Could not load analytics." />}
      {hasNoRuns && <EmptyState title="No model runs yet" hint="Trigger a run from the Runs page." />}
      {run && <AnalyticsSection run={run} />}
    </div>
  );
}
