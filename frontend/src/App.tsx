import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { RunsPage } from "./pages/RunsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { ActionQueuePage } from "./pages/ActionQueuePage";
import { CustomerDetailPage } from "./pages/CustomerDetailPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { StressTestingPage } from "./pages/StressTestingPage";
import { PolicyComparisonPage } from "./pages/PolicyComparisonPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/action-queue" element={<ActionQueuePage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/stress-testing" element={<StressTestingPage />} />
          <Route path="/policy-comparison" element={<PolicyComparisonPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/runs/:runId/recommendations" element={<ActionQueuePage />} />
          <Route path="/customers" element={<CustomerDetailPage />} />
          <Route path="/customers/:customerId" element={<CustomerDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
