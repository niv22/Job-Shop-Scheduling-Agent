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


## Assumptions

- **Time is unitless.** Processing times and deadlines are integers in whatever unit the user defines (minutes, hours, etc.).
- **Operations are strictly ordered** within a job; a job's next operation cannot start before the previous one finishes.
- **Each machine handles one operation at a time** (no parallel machines).
- **Offline machines drop their jobs.** Any job that requires an offline machine is excluded from the current solve rather than queued.
- **Priorities are additive weights**, not hard ordering constraints. A higher priority number causes the solver to minimize that job's weighted completion time more aggressively.
- **Deadlines are hard constraints.** If a deadline makes the problem infeasible, the solver returns no solution. The agent will explain why.
- **The scheduler runs once per user turn.** A guard flag prevents re-running without explicit user confirmation.

---

## Design decisions

**Dual objective** — When priorities are present the model minimizes weighted completion time (priority × completion). Without priorities it minimizes makespan. This keeps the default experience simple while supporting advanced use cases.

**Agent guards scheduler re-runs** — Calling the solver on every tool call would be expensive and confusing. The agent is instructed to ask before re-running after a configuration change, and a flag enforces this within a single turn.

**Jobs data mutated on disk** — Persisting changes to `data/jobs.json` means the UI sidebar always reflects the current state without extra synchronization logic. The tradeoff is that changes are permanent until manually reverted.

**Langfuse tracing** — Every session is tagged and traced via Langfuse for observability. The handler is instantiated at import time; if keys are missing the calls silently fail.
