---
name: causal-cognition
description: "Use long-term Cognition during substantial tasks: retrieve useful execution experience and causal knowledge, validate applicability, and save lessons or lift evidence-backed relations."
---

# Causal Cognition

Causal Cognition is long-term memory for useful execution experience, canonical
concepts, contextual causal relations, and episode-level claims. Its purpose is to test
whether a task-local Causal Belief and a global graph can improve one another
over time through **Index -> Explore -> Validate -> Lift**.

## Index

Start with `search_causal_cognition` using keywords chosen freely from the
task's domain, tools, API behavior, action dependencies, measurements, mechanisms,
current variables, or a suspected missing cause. Retrieve relevant experience when
it can improve the plan, avoid repeated mistakes, or guide a verification step.

- Labels and aliases name the same canonical concept.
- Keywords are topical search cues; a shared keyword does not establish concept
  identity.
- Mechanisms and contexts are also searchable.
- Experience titles, summaries, keywords, evidence, and scope are searchable.
  Check applicability before reusing a recorded lesson.
- Read the returned exact `concept_id`, aliases, keywords, active degree, and
  match reason before choosing where to explore.

## Explore

Use exact `concept_ids` returned by the index or a previous frontier. Each call
expands only one causal edge layer in `upstream`, `downstream`, or `both`
directions. Follow a returned frontier with another call when useful, so a
reasoning chain may cross as many layers as the task requires without asking
one call to flood the context.

Carry all previously inspected IDs in `known_concept_ids` to suppress loops and
repeated frontier nodes. Continue only when the next relation could change at
least one current decision: the task plan, local variable set, an edge direction,
an adjustment choice, or a concrete validation plan. Stop when the frontier no
longer has that potential or the returned budget is exhausted.

## Validate

Every retrieved relation or path is **candidate-only**. Before changing the
task-local Causal Belief, check its measurement meaning, context, lag,
mechanism, and current-task evidence. Do not copy a remembered edge merely
because it is active or highly ranked.

A stored relation is a contextual variant identified by directed endpoints,
lag, and context. Its `claims` entries are separate episode assertions, not
three confidence levels of one fact:

- `hypothesized` contributes only the strongest bounded proposal prior;
  repeating hypotheses cannot accumulate into strong evidence.
- `supported` and `contradicted` accumulate symmetrically with `low=0.5`,
  `medium=1`, and `high=2` log Bayes factors.
- An exactly balanced relation is unresolved and inactive. The displayed belief
  is a retrieval-ranking score, not a causal probability.

Use these summaries to decide what deserves present-task investigation, not as
a substitute for discovery, identification, data, or domain evidence.

## Written explanations

Search also returns current written mechanism notes. Check their evidence, scope,
conditions, and time scale before applying them. Use `instantiate_cognition_note`
to record the selected version and your natural-language application in the active
Belief; applying a note does not change the persistent explanation.

Use `write_cognition_note` to save or revise useful observations, reusable
mechanisms, strategies, or unresolved hypotheses in natural language. No graph
or existing Belief is required. Keep evidence and uncertainty clear. When new
observations change an explanation or its applicable conditions, revise the
related note by its ID and give a `revision_reason` instead of adding a duplicate
dated conclusion. Earlier versions remain available as provenance; normal
retrieval returns the latest available version. Keep observations, projections,
and hypotheses distinct, including the periods they describe.

## Experience memory

After a substantial task, review the actions, tool results, and Belief for one to
three useful lessons. Save a lesson with `update_causal_cognition` and its
`experience` object: `title`, `summary`, `keywords`, `context`, and `evidence`.
This records a searchable episode even when the task has no directed Belief edge.

Useful lessons include API semantics, prerequisites, side effects, ordering
constraints, successful procedures, failed attempts and recovery, and informative
observations. A single episode or documented API contract can qualify. Describe
the observed result, where the lesson applies, and any uncertainty in `summary`;
cite the relevant action/output or documentation in `evidence`. Abstract concrete
names, IDs, and credentials into reusable operation descriptions.

Experience guides future investigation and execution; causal relation claims
retain their evidence and grounding requirements. Search before writing to see
whether the lesson adds useful information. Leave memory unchanged when there is
no useful new lesson or evidence. Belief notes remain task-local working state.

## Lift

When the task yields reusable causal structure, call
`update_causal_cognition` with selected final directed relations, their
`hypothesized` / `supported` / `contradicted` verdicts, confidence, context,
lag, mechanism, and concise notes. Prefer an explicit `relations` list over
lifting every directed local edge. Ground variables to existing canonical
concept IDs whenever possible; ambiguous or weak groundings should be resolved
rather than used to create another concept.

For the causal part of memory, curate directed mechanisms that can recur across
tasks. A date-specific forecast can still reveal reusable drivers: keep dates
and current values in episode provenance rather than treating the entire task
as temporal-only. A final directed Belief edge may be lifted as a
`hypothesized` claim when its direction is mechanism-based but the current
outcome cannot support it causally. If the relation already exists, add this
episode claim or useful metadata instead of duplicating it.

When a useful lesson has unresolved causal direction or grounding, preserve its
scope and uncertainty through experience memory. Never invent or modify a relation
merely to produce a memory write.

Before Lift, use Index and frontier exploration to ground existing exact
`concept_id`s. Only create a concept when it denotes a stable measurement, not
an episode-specific observation: keep dates and release times in the episode
objective/`createdAt`, and use a reusable measurement name. Task-variable
`description` and `source` are optional for Lift and may remain empty. Preserve
available metadata and uncertainty without inventing a source to fill a field.
Relation endpoints accept normalized names and unique shortened names. For
semantic paraphrases or translations, supply `variable_aliases` mapping each
endpoint to its corresponding Belief variable. If a name has multiple matches,
use this mapping to select the intended variable.

Keep dates and current values in episode provenance. A time-specific observation
may be saved as experience when its scope makes it useful for later work.

Use only canonical relation lags: `unknown`, `immediate`, `days`, `weeks`,
`months`, `quarters`, or `years`. If context is needed, provide stable taxonomy
tags beginning with `domain:` or `scope:`. Do not put a date, year, month,
task/run ID, cutoff, or resolution marker in relation context. The episode
already retains the time-specific objective and creation time.

For a `supported` claim, provide a non-empty mechanism and evidence note, and
make sure the matching final directed Causal Belief edge has a non-empty
`method` and `rationale`. A `contradicted` claim also needs a mechanism and
evidence note. A lifted `supported` or `hypothesized` claim cannot be more
confident than its matching final Belief edge.

Use the same tool for focused memory correction:

- `concept_updates` adds or removes aliases and topical keywords;
- `merge_concepts` joins confirmed duplicates, while automatic exact
  label/alias merging handles conservative obvious cases;
- `retract_claims` removes one episode's mistaken claim without erasing other
  claims for the contextual relation;
- isolated concepts may be removed after their relations are resolved.

Do not merge concepts merely because their keywords overlap, and do not retract
a prior claim simply because a new task has a different context. The research
mechanism is the small loop between indexed graph exploration, current-task
validation, and evidence-aware Lift.
