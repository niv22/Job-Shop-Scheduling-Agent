# Job Shop Scheduling Agent

An AI-powered scheduling assistant that lets a project manager interact with a job-shop scheduler in plain language. It uses Google OR-Tools CP-SAT to find optimal or near-optimal schedules, and a LangGraph ReAct agent with tools to run the scheduler, toggle machine availability, and set job priorities and deadlines.

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

Create a `.env` file with:

```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...        # e.g. https://cloud.langfuse.com
```

**Run the Streamlit UI:**

```bash
uv run streamlit run ui.py
```

**Run as CLI:**

```bash
uv run python agent.py
```

**Run the scheduler directly (no agent):**

```bash
uv run python scheduler.py
```

---

## How it works

- **`data/jobs.json`** — the source of truth for jobs, operations, machine statuses, priorities, and deadlines. The agent modifies this file in place when you ask it to change something.
- **`scheduler.py`** — builds and solves the CP-SAT model, writes a timestamped report to `outputs/`.
- **`agent.py`** — LangGraph ReAct agent with five tools: `run_scheduler`, `update_machine_status`, `handle_priority`, `handle_deadline`, `get_latest_scheduler_report`.
- **`ui.py`** — Streamlit chat UI with a sidebar showing live machine status and jobs data.

---

## Data

`data/jobs.json` contains a synthetic dataset of 10 jobs across 5 machines (IDs 0–4), randomly structured to exercise a range of scheduling scenarios — varying operation counts (3–5 ops per job), different machine orderings per job, and a mix of machine statuses (online/offline). Job 0 is pre-configured with a priority and a tight deadline to demonstrate those features out of the box. All other jobs have no priority or deadline set by default.


## CP-SAT model

### Variables

For every `(job, operation)` pair the model creates three variables:

| Variable | Domain | Purpose |
|----------|--------|---------|
| `start_{job}_{op}` | `[0, horizon]` | Time step at which the operation begins |
| `end_{job}_{op}` | `[0, horizon]` | Time step at which the operation finishes |
| `interval_{job}_{op}` | derived | Interval variable linking start, fixed duration, and end — used in no-overlap constraints |

`horizon` is the sum of all processing times across every schedulable operation, giving a tight (but always valid) upper bound on when any task can finish. When no priorities are present an additional scalar variable `makespan` is added as the minimisation target.

### Constraints

1. **No-overlap** — for each machine, all interval variables assigned to it are passed to `add_no_overlap`. The solver guarantees that no two operations share a machine at the same time.
2. **Precedence** — within each job, `start[op+1] >= end[op]` is enforced for every consecutive pair of operations, ensuring the strict ordering of a job's tasks.
3. **Deadlines (hard)** — when a job carries a deadline, `end[last_op] <= deadline` is added as a hard constraint. A single violated deadline makes the whole problem infeasible.

### Pre-solve feasibility checks

Before handing the model to the solver, two lightweight checks emit warnings (and feed the infeasibility analysis if needed):

- **Job-level check** — if a job's deadline is less than the sum of its own processing times it can never be met regardless of scheduling order.
- **Machine-level EDF check** — for each machine and each deadline value `d`, the total processing time of all jobs with `deadline <= d` that use that machine must not exceed `d`. This catches contention-induced infeasibility that the job-level check misses.

### Objective

| Mode | Objective |
|------|-----------|
| No priorities set | Minimize `makespan` — the finish time of the last operation across all jobs |
| Any job has a priority | Minimize `Σ priority(job) × end[last_op]` — weighted sum of completion times; higher-priority jobs are penalised more for finishing late |

The solver runs with a 30-second wall-clock limit and a single worker (`num_workers = 1`). It reports `OPTIMAL` when the objective value equals the best bound, and `FEASIBLE` otherwise (along with the optimality gap in the output report).

---

## Assumptions

- **Time is unitless.** Processing times and deadlines are integers in whatever unit the user defines (minutes, hours, etc.).
- **Operations are strictly ordered** within a job; a job's next operation cannot start before the previous one finishes.
- **Each machine handles one operation at a time** (no parallel machines).
- **Offline machines drop their jobs.** Any job that requires an offline machine is excluded from the current solve rather than queued.
- **Priorities are additive weights**, not hard ordering constraints. A higher priority number causes the solver to minimize that job's weighted completion time more aggressively.
- **Deadlines are hard constraints.** If a deadline makes the problem infeasible, the solver returns no solution. The agent will explain why.
- **The scheduler runs once per user turn.** A guard flag prevents re-running without explicit user confirmation.

---

## Agent design

**Graph structure** — Two nodes (`agent` → `tools`) connected by a conditional edge. After every LLM response, `tools_condition` checks whether the model emitted tool calls; if so it routes to `ToolNode`, which executes all calls and returns results. Control then returns to `agent`, repeating until the model produces a plain-text reply.

```
[user input]
     │
     ▼
  agent  ──(has tool calls?)──► tools
     ▲                              │
     └──────────────────────────────┘
     (plain text response → user)
```

**Tools** — Five LangChain `@tool` functions, each with a focused scope:

| Tool | What it does |
|------|-------------|
| `run_scheduler` | Invokes the CP-SAT solver and returns the formatted report |
| `update_machine_status` | Marks a machine `online` or `offline` in `jobs.json` |
| `handle_priority` | Sets or removes the numeric priority on a job |
| `handle_deadline` | Sets or removes the hard deadline on a job |
| `get_latest_scheduler_report` | Re-fetches the most recent report without re-solving |


**Scheduler re-run guard** — A module-level `_scheduler_called` flag is reset to `False` on each new human message and set to `True` the first time `run_scheduler` fires. If the tool is called a second time in the same turn it returns an early-exit string instead of solving again. This prevents the solver from running redundantly during multi-step tool chains.

**What-if exception** — The system prompt explicitly allows the agent to chain `update_*` + `run_scheduler` in a single turn when the user frames the request as a hypothetical ("what if machine 2 goes offline?"). In all other cases the agent asks for confirmation before re-solving.

**Observability** — Every session gets a UUID and is traced end-to-end via Langfuse, tagged `jssp` and `cli` (or `streamlit` from the UI). If the Langfuse keys are absent the callback silently no-ops.

---

## Infeasibility analysis

When the CP-SAT solver returns no solution, the agent triggers a diagnostic pass implemented in `scheduler_infeasibility_analysis.py` via the `InfeasibilityAnalysisMixin` class.

**Technique** — The diagnostic re-solves a copy of the model with no objective, guarding each deadline with an *assumption literal* (a boolean variable the solver can flip). After the solve, `sufficient_assumptions_for_infeasibility()` returns a minimal conflicting subset of those literals — i.e. the smallest group of deadlines that provably cannot all be met at the same time. (Method from google/or-tools#973.)

**Why deadlines, not precedence/no-overlap?** Precedence and machine no-overlap constraints can always be satisfied by serialising all tasks within the planning horizon — they can never alone cause infeasibility. Deadlines are therefore the only constraint family the diagnostic needs to investigate.

**Possible verdicts**

- `infeasible` — deadlines are the confirmed cause; the conflicting set is shown.
- `deadlines_ok` — deadlines are jointly satisfiable, meaning the original failure was likely a solver timeout rather than true infeasibility.
- `inconclusive` — the diagnostic itself timed out (30 s limit); increase `max_time_in_seconds` or `num_workers`.
- `no_deadlines` — infeasibility with no deadline constraints present is unexpected; indicates malformed input data (e.g. zero or negative processing times).

---

## Design decisions

**Dual objective** — When priorities are present the model minimizes weighted completion time (priority × completion). Without priorities it minimizes makespan. This keeps the default experience simple while supporting advanced use cases.

**Agent guards scheduler re-runs** — Calling the solver on every tool call would be expensive and confusing. The agent is instructed to ask before re-running after a configuration change, and a flag enforces this within a single turn.

**Jobs data mutated on disk** — Persisting changes to `data/jobs.json` means the UI sidebar always reflects the current state without extra synchronization logic. The tradeoff is that changes are permanent until manually reverted.

**Langfuse tracing** — Every session is tagged and traced via Langfuse for observability. The handler is instantiated at import time; if keys are missing the calls silently fail.

**Minimal system prompt** — The agent runs on a small LLM (Groq-hosted Llama), so the system prompt is kept short and directive: it names the tools, states the re-run guard rule, and gives the what-if exception. Verbose prose or redundant examples would consume context that the model needs for reasoning and tend to cause instruction-following failures in smaller models.

**Structured tool output** — Each tool returns a clearly formatted string (section headers, labelled fields) rather than raw JSON or unformatted text. This gives the LLM a predictable surface to cite in its reply and produces readable output in the Streamlit UI without additional post-processing.
