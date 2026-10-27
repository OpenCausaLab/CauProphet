"""Run Scout's ASP search and compare candidate DAGs with CI evidence."""

from __future__ import annotations

import json
import sys
from typing import Any


def _atoms(answer: Any) -> list[str]:
    return sorted(str(symbol) for symbol in answer.symbols(shown=True))


def _parse_atom(atom: str) -> tuple[str, list[str]] | None:
    if "(" not in atom:
        return atom, []
    name, rest = atom.split("(", 1)
    return name, rest.rstrip(")").split(",")


def _models(program: str, limit: int, diagnostics: bool) -> list[dict[str, Any]]:
    import clingo

    control = clingo.Control(["--opt-mode=ignore", "-n", str(limit)])
    control.add("base", [], program)
    control.ground([("base", [])])
    control.assign_external(clingo.Function("scout_diagnostics"), diagnostics)
    result: list[dict[str, Any]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            result.append({"atoms": _atoms(model), "cost": sum(model.cost)})
    return result


def _graph(atoms: list[str], nodes: list[str] | None = None):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes or [])
    for atom in atoms:
        parsed = _parse_atom(atom)
        if parsed and parsed[0] == "selected_edge" and len(parsed[1]) >= 2:
            graph.add_edge(parsed[1][0], parsed[1][1])
    return graph


def _hypotheses(atoms: list[str]) -> list[str]:
    result = []
    for atom in atoms:
        parsed = _parse_atom(atom)
        if parsed and parsed[0] == "selected_hypothesis" and parsed[1]:
            result.append(parsed[1][0])
    return sorted(set(result))


def _edges(atoms: list[str]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for atom in atoms:
        parsed = _parse_atom(atom)
        if parsed and parsed[0] == "selected_edge" and len(parsed[1]) >= 2:
            result.add((parsed[1][0], parsed[1][1]))
    return result


def _model_cost(model: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
    selected = set(_hypotheses(model["atoms"]))
    cost = 0.0
    for candidate in candidates:
        confidence = float(candidate.get("confidence", 0.5))
        edge_count = len(candidate.get("candidate_edges", []))
        support_count = len(candidate.get("support", []))
        if candidate.get("id") in selected:
            cost += 1.0 - confidence + edge_count * 0.1
        else:
            cost += confidence * 0.25 + support_count * 0.05
    cost += max(0.0, float(model.get("cost", 0.0))) * 0.001
    return cost


def _graph_consensus(models: list[dict[str, Any]]) -> tuple[list[list[str]], list[list[str]]]:
    if not models:
        return [], []
    directions = [_edges(model["atoms"]) for model in models]
    compelled = set.intersection(*directions)
    compelled = {
        (left, right)
        for left, right in compelled
        if (right, left) not in compelled
    }
    pair_directions: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for edge_set in directions:
        for left, right in edge_set:
            key = tuple(sorted((left, right)))
            pair_directions.setdefault(key, set()).add((left, right))
    reversible = {
        (left, right)
        for (left, right), values in pair_directions.items()
        if len(values) > 1
    }
    return [list(edge) for edge in sorted(compelled)], [list(edge) for edge in sorted(reversible)]


def _matches(model: dict[str, Any], evidence: list[dict[str, Any]], nodes: list[str]) -> tuple[list[str], list[str]]:
    import networkx as nx

    graph = _graph(model["atoms"], nodes)
    matched: list[str] = []
    conflicted: list[str] = []
    for fact in evidence:
        try:
            implied_independent = nx.is_d_separator(
                graph,
                {fact["from"]},
                {fact["to"]},
                set(fact.get("conditioned_on", [])),
            )
            if bool(implied_independent) == bool(fact["independent"]):
                matched.append(fact["id"])
            else:
                conflicted.append(fact["id"])
        except (nx.NetworkXError, nx.NodeNotFound):
            conflicted.append(fact["id"])
    return matched, conflicted


def _search(
    program: str,
    top_k: int,
    evidence: list[dict[str, Any]],
    nodes: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    # 12 optional hypotheses gives 4096 combinations; the cap makes the
    # completeness claim explicit rather than silently scaling exponentially.
    limit = 4096
    structural = _models(program, limit, diagnostics=False)
    diagnostics = _models(program, limit, diagnostics=True)
    evaluated = []
    for model in structural:
        matched, conflicted = _matches(model, evidence, nodes)
        evaluated.append({**model, "matched": matched, "conflicted": conflicted})

    active = list(evidence)
    relaxed: list[dict[str, Any]] = []
    while True:
        compatible = [
            model for model in evaluated
            if all(item["id"] in model["matched"] for item in active)
        ]
        if compatible or not active:
            break
        weakest = min(active, key=lambda item: (float(item.get("confidence", 0.5)), item["id"]))
        active = [item for item in active if item["id"] != weakest["id"]]
        relaxed.append({
            "id": weakest["id"],
            "confidence": float(weakest.get("confidence", 0.5)),
            "source": str(weakest.get("source", "")),
            "reason": "No structurally valid candidate graph satisfied all retained CI evidence.",
        })

    ranked_compatible = sorted(
        compatible,
        key=lambda model: (_model_cost(model, candidates), _hypotheses(model["atoms"])),
    )
    selected = ranked_compatible[:top_k]
    compelled_edges, reversible_pairs = _graph_consensus(compatible)
    rejected_models = []
    for model in evaluated:
        reasons = [
            atom
            for atom in model["atoms"]
            if atom.startswith("violation(")
        ]
        reasons.extend(
            f"ci_conflict({fact['id']})"
            for fact in active
            if fact["id"] in model["conflicted"]
        )
        if reasons:
            rejected_models.append({"hypotheses": _hypotheses(model["atoms"]), "reasons": reasons})
        if len(rejected_models) >= 64:
            break
    tests: list[dict[str, Any]] = []
    for fact in evidence:
        independent_in = []
        dependent_in = []
        for index, model in enumerate(compatible):
            if fact["id"] in model["matched"]:
                (independent_in if fact["independent"] else dependent_in).append(index + 1)
            else:
                (dependent_in if fact["independent"] else independent_in).append(index + 1)
        if independent_in and dependent_in:
            tests.append({
                "from": fact["from"],
                "to": fact["to"],
                "conditioned_on": fact.get("conditioned_on", []),
                "independent_in": independent_in,
                "dependent_in": dependent_in,
                "reason": "This CI query separates the remaining candidate graphs.",
            })

    return {
        "answer_sets": selected,
        "search": {
            "status": (
                "incomplete"
                if len(structural) >= limit
                else ("satisfiable" if selected and not relaxed else ("relaxed" if selected else "unsatisfiable"))
            ),
            "exhaustive": len(structural) < limit,
            "structurally_valid_count": len(structural),
            "compatible_count": len(compatible),
            "returned_count": len(selected),
            "compelled_edges": compelled_edges,
            "reversible_pairs": reversible_pairs,
            "retained_evidence": [item["id"] for item in active],
            "relaxed_evidence": relaxed,
            "rejected_models": rejected_models + [
                {"hypotheses": _hypotheses(model["atoms"]), "reasons": [atom for atom in model["atoms"] if atom.startswith("violation(")]}
                for model in diagnostics
                if any(atom.startswith("violation(") for atom in model["atoms"])
                and len(rejected_models) < 64
            ][: max(0, 64 - len(rejected_models))],
            "diagnostics_exhaustive": len(diagnostics) < limit,
            "discriminating_tests": tests,
            "assumptions": ["candidate structures are DAGs", "CI facts are interpreted through d-separation", "Meek closure is applied only when graph_semantics declares a valid CPDAG/PDAG", "causal sufficiency and Markov/faithfulness are provisional assumptions"],
            "warnings": ([] if len(structural) < limit else ["Search reached the 4096-model cap; exhaustive=false."]),
        },
    }


def solve(payload: str, top_k: int) -> dict[str, Any]:
    import clingo  # noqa: F401

    try:
        data = json.loads(payload)
        if isinstance(data, dict) and "program" in data:
            return _search(
                str(data["program"]),
                top_k,
                list(data.get("evidence", [])),
                list(data.get("nodes", [])),
                list(data.get("candidates", [])),
            )
    except json.JSONDecodeError:
        pass
    return {"answer_sets": _models(payload, top_k, diagnostics=False)}


def main() -> int:
    try:
        top_k = max(1, min(32, int(sys.argv[1]))) if len(sys.argv) > 1 else 8
        print(json.dumps(solve(sys.stdin.read(), top_k), ensure_ascii=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"answer_sets": [], "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
