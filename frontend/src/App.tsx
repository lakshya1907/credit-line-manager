import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RunsPage } from "./pages/RunsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { ActionQueuePage } from "./pages/ActionQueuePage";
import { CustomerDrilldownPage } from "./pages/CustomerDrilldownPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/runs/:runId/recommendations" element={<ActionQueuePage />} />
          <Route path="/customers" element={<CustomerDrilldownPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
