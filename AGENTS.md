# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes, and to keep autonomous
parallel agent loops safe and verifiable. Merge with project-specific instructions
as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Parallel Agent Loops

**Five agents max. Disjoint writes. Fresh-context critics. Bounded rounds.**

Parallelism multiplies throughput and it multiplies damage. Everything below exists
to keep the second from cancelling the first.

### 5.1 The unit of work

A task is dispatchable only when you can state all three:

- **Scope** — the exact files it may write, and no others.
- **Acceptance** — a command to run and the expected exit code / number.
- **Self-containment** — it fits in one fresh context without reading the whole repo.

Can't state them? The task isn't ready. Split it or spec it before spending an agent
on it.

### 5.2 Concurrency budget

- **Max 5 concurrent agents.** Builders and critics draw from the same budget.
- Fan out to the number of *genuinely independent* pieces, not to 5. Three
  independent tasks means three agents, not three plus two invented ones.
- Refill a freed slot only from tasks that are already dispatch-ready **and**
  write-disjoint from everything still running.
- Stages stay sequential; parallelism happens *within* a stage. Gold bar → taxonomy
  gap → generators → verification gates. Fanning out a stage whose inputs aren't
  final produces five agents building against a moving target.

### 5.3 Write disjointness (the hard rule)

Every concurrent agent owns a disjoint set of output paths.

- Partition by program / template module / record type. A given source file has
  **exactly one writer** at a time.
- **Never let two agents write the same shard.** Shard writes go to a temp file and
  `os.replace` on finalize — two writers race the rename and one silently wins.
- Repo-global derived artifacts (progress reports, coverage/status JSON, README
  composition tables) are **lead-only**. Agents report their numbers; the lead
  regenerates the artifact once, after the join.
- Shared inputs (seed config, gold bar, format spec) are **read-only for the wave**.
  Changing a shared contract is its own serialized task, run alone.
- Git belongs to the lead. Agents don't commit, rebase, or switch branches. A task
  that genuinely needs to move the tree gets its own worktree.

### 5.4 Determinism is what makes this safe

Parallel work is reproducible only because output is a pure function of its inputs —
here, a seed derived from `(program, template, variant)`. Nothing in a dispatched
task may key off wall-clock time, PID, worker index, or dispatch order. **If the
output depends on which agent produced it, the wave is not resumable and the result
is not verifiable.**

### 5.5 Builder / critic separation

- The critic runs in a **fresh context** and never sees the builder's reasoning —
  only the artifacts on disk.
- The critic inspects **real files, real samples, real numbers**. A critic grading a
  summary is measuring the summary.
- The critic returns pass/fail per axis, the evidence, and **the single largest
  remaining gap** — not twelve nits ranked equally.
- Builder and critic for the same piece are never the same agent.

### 5.6 Loop termination — decide before you start

Every loop declares its exit conditions up front:

1. **Win** — acceptance check passes and the critic clears every axis.
2. **Convergence** — a round moves no axis by a meaningful margin. Stop, report the
   residual gap.
3. **Round cap** — a hard maximum (3 is usually right). Hitting it is a *result to
   report*, not a failure to hide.
4. **Blocked** — the same failure survives two rounds. Stop and escalate with the
   evidence.

A loop without a stated cap is a spend-forever loop. Never write one.

### 5.7 The report contract

Every agent returns the same shape, so the lead can merge without re-reading
everything:

```
SCOPE:    files owned + actually written
RESULT:   what changed, one paragraph
EVIDENCE: command run → exit code → key numbers
GAP:      the single largest thing still wrong
```

A claim with no command and no exit code is unverified. Treat it as unverified,
including when it's your own.

### 5.8 Joining a wave, and recovering from it

After a wave returns, the lead — and only the lead:

1. Re-runs the **global** regression gate from the repo root. Per-agent checks are
   scoped and can all pass while the whole is broken.
2. Regenerates derived reports.
3. Reconciles conflicting claims. Two agents reporting the same metric differently
   means at least one is wrong — find out which before building on either.
4. Chooses the next wave from measured gaps, not from what's convenient to build.

On failure:

- One agent failing does not abort its wave. Collect it, join the rest, re-dispatch
  with the failure text in the brief.
- Re-dispatch must be safe to re-run. A task that half-applied its edits gets its
  scope reverted first — never patched forward blindly.
- Three failures on one task means the task is mis-specified. Fix the spec, not the
  agent.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and parallel waves join cleanly — no clobbered files, no unverified claims, no loop that ran past the point of measurable improvement.
