import type { EffectEstimate } from "./types";

export type EffectTone = "valid" | "warning" | "danger" | "unknown";

export interface EffectPresentation {
  tone: EffectTone;
  status: "decision-grade" | "caution" | "sensitivity-only" | "inconclusive" | "refuted" | "unknown";
  decisionGrade: boolean;
  details: string[];
}

export function formatEffectIntervention(effect: EffectEstimate): string {
  const intervention = effect.intervention;
  if (!intervention?.type) return "unspecified intervention";
  if (intervention.type === "additive_shift") {
    return `T := T + ${intervention.shiftDelta ?? "?"}`;
  }
  if (intervention.type === "contrast") {
    return `${String(intervention.controlValue ?? "?")} → ${String(intervention.treatmentValue ?? "?")}`;
  }
  return intervention.type;
}

export function presentEffect(effect: EffectEstimate): EffectPresentation {
  const details = [
    ...(effect.validity?.reasons ?? []),
    ...(effect.warnings ?? []),
    ...(effect.refutation_note ? [effect.refutation_note] : []),
  ];

  if (effect.refuted) {
    return { tone: "danger", status: "refuted", decisionGrade: false, details };
  }
  if (effect.estimateRole === "sensitivity_only" || effect.validity?.status === "sensitivity_only") {
    return { tone: "danger", status: "sensitivity-only", decisionGrade: false, details };
  }
  if (effect.validity?.status === "inconclusive") {
    return { tone: "danger", status: "inconclusive", decisionGrade: false, details };
  }
  if (effect.validity?.status === "caution") {
    if (effect.validity.decisionGrade !== true) {
      return { tone: "danger", status: "inconclusive", decisionGrade: false, details };
    }
    return {
      tone: "warning",
      status: "caution",
      decisionGrade: true,
      details,
    };
  }
  if (effect.validity?.status === "valid" && effect.validity.decisionGrade === true) {
    return { tone: "valid", status: "decision-grade", decisionGrade: true, details };
  }
  return { tone: "unknown", status: "unknown", decisionGrade: false, details };
}
