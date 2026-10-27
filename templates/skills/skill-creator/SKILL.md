---
name: skill-creator
description: Create or refine a reusable CauProphet workspace skill when a repeated workflow needs stable, task-specific guidance. Do not use for one-off instructions that belong only to the current task.
---

# Workspace Skill Creator

Turn a demonstrated, repeatable workflow into a concise skill under
`skills/<skill-name>/SKILL.md`. A skill should preserve the decisions and
invariants that future runs would otherwise need to rediscover; it should not
archive an entire conversation or constrain unrelated work.

## Create or update

1. Inspect the relevant workflow, existing workspace conventions, and
   `skills/` before writing. Update the most specific existing skill instead of
   creating a near-duplicate.
2. Identify realistic trigger requests, the desired outcome, non-obvious
   decision criteria, and any stopping conditions. If the pattern has not
   repeated or stabilized, keep it as task notes rather than a skill.
3. Use a lowercase, hyphenated folder name. Every skill needs one entrypoint:

   ```text
   skills/<skill-name>/SKILL.md
   ```

4. Start `SKILL.md` with YAML frontmatter containing `name` and a concise,
   discriminating `description` that says both what the skill does and when it
   applies. Keep the body focused on guidance that changes future decisions.
5. Add `scripts/`, `references/`, or `assets/` only when a repeated concrete
   need justifies them. Any helper must live inside the skill and be usable by
   CauProphet without machine-specific internal paths, private system skills, or hidden
   installation scripts.
6. Preserve the current request's scope and authorization boundaries. A skill
   may describe when approval is needed; it does not grant permission to
   mutate external systems, publish, deploy, or perform destructive actions.

## Validate

Before reporting completion:

- read the saved file back and confirm the folder name, frontmatter `name`, and
  description agree;
- check that no scaffold placeholders, broken relative references, duplicated
  instructions, or unrelated universal rules remain;
- run any helper scripts on representative inputs if the skill contains them;
- perform an isolated forward test in a temporary directory. Give a fresh run
  only the saved skill, a realistic future request, and the minimum required
  artifacts; do not provide the intended answer or reuse test output in the
  workspace. Inspect whether it selects the skill appropriately, respects its
  scope, and produces the observable outcome. If an independent run is not
  available, do the same as a clean-room dry run using only those inputs and
  state that limitation.

Revise only when the validation exposes a concrete failure. Keep the final
skill as small as the reliable workflow permits.
