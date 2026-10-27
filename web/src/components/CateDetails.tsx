function stateRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stateNumber(value: unknown): string {
  if (value === null || value === undefined || value === "") return "N/A";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(4) : "N/A";
}

export function CateDetails({ cate }: { cate: Record<string, unknown> }) {
  const groups = Array.isArray(cate.groups)
    ? cate.groups.map(stateRecord).filter((value): value is Record<string, unknown> => value !== null)
    : [];
  const isAdditiveShift = cate.estimand === "conditional_total_additive_shift"
    || Array.isArray(cate.groups);

  if (!isAdditiveShift) {
    const quantiles = Array.isArray(cate.quantile_estimates)
      ? cate.quantile_estimates.map(stateRecord).filter((value): value is Record<string, unknown> => value !== null)
      : [];
    return (
      <details className="state-cate-disclosure">
        <summary>Legacy linear-interaction CATE by {String(cate.modifier ?? "unknown")}</summary>
        <div className="state-cate-meta">Formula: {String(cate.formula ?? "not available")}</div>
        <div className="state-cate-meta">
          Interaction coefficient: {stateNumber(cate.interaction_coef)} · p-value: {stateNumber(cate.interaction_p_value)}
        </div>
        {quantiles.length > 0 && (
          <table className="state-table state-cate-table">
            <thead>
              <tr><th>Modifier point</th><th>CATE</th></tr>
            </thead>
            <tbody>
              {quantiles.map((quantile, index) => (
                <tr key={String(quantile.label ?? index)}>
                  <td>{String(quantile.label ?? "unknown")} = {String(quantile.modifier_value ?? "?")}</td>
                  <td>{stateNumber(quantile.cate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </details>
    );
  }

  const identification = stateRecord(cate.identification);
  const reconciliation = stateRecord(cate.reconciliation);
  const conditioningSet = Array.isArray(identification?.conditioning_set)
    ? identification.conditioning_set.map(String).join(", ")
    : "unknown";

  return (
    <details className="state-cate-disclosure">
      <summary>CATE by {String(cate.modifier ?? "unknown")} · {String(stateRecord(cate.validity)?.status ?? "unknown")}</summary>
      <div className="state-cate-meta">
        Identification: {String(identification?.status ?? "unknown")} · conditioning set: {conditioningSet}
      </div>
      <div className="state-cate-meta">
        Population estimator: {String(cate.population_estimator_family ?? "unknown")} · role: {String(cate.population_estimate_role ?? "unknown")}
      </div>
      {reconciliation && (
        <div className="state-cate-meta">
          Reconciliation: {String(reconciliation.status ?? "unknown")} · ATE {stateNumber(reconciliation.population_ate)} · weighted CATE {stateNumber(reconciliation.weighted_cate)} · difference {stateNumber(reconciliation.ate_minus_weighted_cate)} · tolerance {stateNumber(reconciliation.effective_absolute_tolerance)}
        </div>
      )}
      <table className="state-table state-cate-table">
        <thead>
          <tr><th>Stratum</th><th>n</th><th>Effect</th><th>Estimator</th><th>Form</th><th>Role</th><th>Exploratory 95% interval</th><th>Support</th><th>Validity / reasons</th><th>Warnings</th></tr>
        </thead>
        <tbody>
          {groups.map((group) => {
            const support = stateRecord(group.support_diagnostics);
            const validity = stateRecord(group.validity);
            const lower = stateNumber(group.ci_lower);
            const upper = stateNumber(group.ci_upper);
            const hasInterval = lower !== "N/A" && upper !== "N/A";
            return (
              <tr key={String(group.id ?? group.label ?? "unknown")}>
                <td>{String(group.label ?? group.id ?? "unknown")}</td>
                <td>{String(group.n_rows ?? "?")}</td>
                <td>{stateNumber(group.effect)}</td>
                <td>{String(group.estimator_family ?? "unknown")}</td>
                <td>{String(group.functional_form ?? "unknown")}</td>
                <td>{String(group.estimate_role ?? "unknown")}</td>
                <td>{hasInterval ? `[${lower}, ${upper}]` : "not available"}</td>
                <td>{String(support?.status ?? "unknown")}</td>
                <td>
                  {String(validity?.status ?? "unknown")}
                  {Array.isArray(validity?.reasons) && validity.reasons.length > 0
                    ? `: ${validity.reasons.map(String).join("; ")}`
                    : ""}
                </td>
                <td>{Array.isArray(group.warnings) ? group.warnings.map(String).join("; ") : "none"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </details>
  );
}
