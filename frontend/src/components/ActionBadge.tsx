import type { Action } from "../api/types";
import { StatusBadge, type Tone } from "./ui/StatusBadge";

const ACTION_TONE: Record<Action, Tone> = {
  increase: "success",
  decrease: "danger",
  hold: "neutral",
};

export function ActionBadge({ action }: { action: Action }) {
  return <StatusBadge tone={ACTION_TONE[action]}>{action}</StatusBadge>;
}
