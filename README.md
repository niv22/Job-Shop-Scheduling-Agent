# Job Shop Scheduling Agent

## Contents

1. [Project Description](#1-project-description)
2. [Assumptions](#2-assumptions)
3. [Design Decisions](#3-design-decisions)
4. [On Scheduling](#4-on-scheduling)
   - [4.1 On Data](#41-on-data)
   - [4.2 The CP-SAT Solver](#42-the-cp-sat-solver)
   - [4.3 Infeasibility Analysis](#43-infeasibility-analysis)
   - [4.4 Code](#44-code)
5. [On the JSSP Agent](#5-on-the-jssp-agent)
   - [5.1 Agent Design](#51-agent-design)
   - [5.2 Agent Code](#52-agent-code)
6. [Set Up & Execution](#6-set-up--execution)
   - [6.1 Prerequisites](#61-prerequisites)
   - [6.2 Set Up](#62-set-up)

---

## 1. Project Description

This project is an AI-powered job-shop scheduling assistant. A project manager can interact with a CP-SAT scheduler entirely in plain language — asking it to run a schedule, take a machine offline, raise a job's priority, or set a deadline — without touching any configuration files directly. This has two main parts, the scheduler which solves the scheduling problem and the AI agent which allows the user to manage the scheduler.

**What you can do:**
- Run the scheduler and get a formatted report of the optimal (or best-found) schedule.
- Toggle machines online or offline and immediately see how the schedule changes.
- Assign numeric priorities to jobs so the solver penalises high-priority jobs for finishing late.
- Set deadlines on jobs
- Get a explanations on when the scheduling is not feasible

**Underlying technology:**
- **Google OR-Tools CP-SAT** — For performing the scheduling of the jobs.
- **LangGraph ReAct agent** — For AI agent graph orchestrating tool calls.
- **Groq-hosted Llama** — Free hosted LLM.
- **Streamlit** — browser-based chat UI.
- **Langfuse** — end-to-end observability and tracing for every agent session.

---

## 2. Assumptions

- **Time is unitless.** Processing times and deadlines are integers in whatever unit the user defines (minutes, hours, etc.). Hence the unit is not considered here.
- **Operations are strictly ordered** within a job; a job's next operation cannot start before the previous one finishes.
- **Each machine handles one operation at a time** (no parallel machines).
- **Offline machines drop their jobs.** Any job that has atleast one task that requires an offline machine is excluded from the current solve.
- **Priorities are additive weights**, not hard ordering constraints. A higher priority number causes the solver to minimise that job's weighted completion time more aggressively.
- **Deadlines are hard constraints.** If a deadline makes the problem infeasible, the solver returns no solution. The agent will explain why.

---

## 3. Design Decisions

**Dual objective** — When priorities are present the model minimises weighted completion time (priority × completion). Without priorities it minimises makespan. 

**Small LLM with Minimal system prompt and Agent harness** — The agent runs on a small LLM (Groq-hosted Llama), so the system prompt is kept short and directive, while deterministic actions like formatting, display of report with scheduler run are programmed.
*Alternate approach* - using more capable LLMs will improve the agent drastically while it comes with a cost implication.
However, the agent harness style of offloading predicatble tasks to non-LLM based functions improves over all efficiency and determinism of the system.

**Structured tool output** — Each tool returns a clearly formatted string (section headers, labelled fields) rather than raw JSON or unformatted text. This gives the LLM a predictable surface to cite in its reply and produces readable output in the Streamlit UI without additional post-processing.

**Infeasibility analysis grounded in determinstic evaluation** - Since the agent runs a small LLM, the reasoning capabilities are limited. Hence solver infeasibility is first examined thoroughly via programming and then the report is sent to the agent for further insights. Refer #43-infeasibility-analysis for how this is done. Also, having this agent harness-style approach to infeasibility analysis provides more reliable outcomes.
*Alternate approach* - If using more capable LLMs, the LLM may be able to evaluate based on detailed logs and input data without needing this additional analysis. It may also suggest better fixes for handiling the infeasibility scenario.

**Langfuse tracing** — Every session is tagged and traced via Langfuse for observability. The handler is instantiated at import time; if keys are missing the calls silently fail.

**Agent guards scheduler re-runs** — A guard flag prevents re-running without explicit user confirmation. This is for handling the limitation of the small Language model's reasoning and tool call ability.

**Jobs data mutated on disk** — Persisting changes to `data/jobs.json` means the UI sidebar always reflects the current state without extra synchronisation logic. The tradeoff is that changes are permanent until manually reverted.

---

## 4. On Scheduling

### 4.1 On Data
The scheduler reads from `data/jobs.json`, which is the single source of truth for all scheduling inputs. 
The bundled dataset is a synthetic set of 10 jobs across 5 machines (IDs 0–4), randomly structured to exercise a range of scheduling scenarios — varying operation counts (3–5 ops per job), different machine orderings per job, and a mix of machine statuses. 
The agent modifies `data/jobs.json` in place when you ask it to change machine status, priorities, or deadlines.
The jobs.json contains the following fields:
```
{
      "id": 3, # job identifier
      "name": "Job3", # job name, currently on the lines of Job0, Job1 etc
      "deadline": 10, # optional, mentioned in time units
      "priority" : 2, # optional, higher the number -> more priority
      "operations": [  # list of tasks to be performed on different machines
        {
          "machine_id": 0, # the machine id where the tasks neeeds to be run
          "processing_time": 2 # in time units
        },
        {
          "machine_id": 2,
          "processing_time": 3
        },
        {
          "machine_id": 3,
          "processing_time": 4
        },
      ]
    }
```

the data.json also contains machine statuses
```
  "machines": [
    {
      "id": 0,
      "status": "online" 
    },
    {
      "id": 1,
      "status": "offline" # if a job has a task for this machine, that job will be dropped for scheduling
    },
  ],
```

---

### 4.2 The CP-SAT Solver

The scheduler is built on [Google OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model)
#### Variables

For every `(job, operation)` pair the model creates three variables:

| Variable | Domain | Purpose |
|----------|--------|---------|
| `start_{job}_{op}` | `[0, horizon]` | Time step at which the operation begins |
| `end_{job}_{op}` | `[0, horizon]` | Time step at which the operation finishes |
| `interval_{job}_{op}` | derived | Interval variable linking start, fixed duration, and end — used in no-overlap constraints |

`horizon` is the sum of all processing times across every schedulable operation, giving a tight (but always valid) upper bound on when any task can finish. When no priorities are present an additional scalar variable `makespan` is added as the minimisation target.

#### Constraints

1. **No-overlap** — for each machine, all interval variables assigned to it are passed to `add_no_overlap`. The solver guarantees that no two operations share a machine at the same time.
2. **Precedence** — within each job, `start[op+1] >= end[op]` is enforced for every consecutive pair of operations, ensuring the strict ordering of a job's tasks.
3. **Deadlines (hard)** — when a job carries a deadline, `end[last_op] <= deadline` is added as a hard constraint. A single violated deadline makes the whole problem infeasible.

#### Objective

| Mode | Objective |
|------|-----------|
| No priorities set | Minimise `makespan` — the finish time of the last operation across all jobs |
| Any job has a priority | Minimise `Σ priority(job) × end[last_op]` — weighted sum of completion times; higher-priority jobs are penalised more for finishing late |

The solver runs with a 30-second wall-clock limit and a single worker (`num_workers = 1`). It reports `OPTIMAL` when the objective value equals the best bound, and `FEASIBLE` otherwise (along with the optimality gap in the output report).

#### Pre-solve Feasibility Checks

Before handing the model to the solver, two lightweight checks emit warnings (and feed the infeasibility analysis if needed):

- **Job-level check** — if a job's deadline is less than the sum of its own processing times it can never be met regardless of scheduling order.
- **Machine-level Earliest Deadline First check** — for each machine and each deadline value `d`, the total processing time of all jobs with `deadline <= d` that use that machine must not exceed `d`. This catches infeasibility that the previous job-level check is likely to miss.

---

### 4.3 Infeasibility Analysis

When the CP-SAT solver returns no solution, the agent triggers a diagnostic pass implemented in `scheduler_infeasibility_analysis.py` via the `InfeasibilityAnalysisMixin` class.
The idea for approaching the infeasibility analysis this way was got from this thread - https://github.com/google/or-tools/issues/973


**Technique** — The diagnostic re-solves a copy of the model with no objective, guarding each deadline with an *assumption literal* (a boolean variable the solver can flip). After the solve, `sufficient_assumptions_for_infeasibility()` returns a minimal conflicting subset of those literals — i.e. the smallest group of deadlines that provably cannot all be met simultaneously.

**Why deadlines, not precedence/no-overlap?** Precedence and machine no-overlap constraints can always be satisfied by serialising all tasks within the planning horizon — they can never alone cause infeasibility. Deadlines are therefore the only constraint family the diagnostic needs to investigate.

**Possible verdicts:**

| Verdict | Meaning |
|---------|---------|
| `infeasible` | Deadlines are the confirmed cause; the conflicting set is shown |
| `deadlines_ok` | Deadlines are jointly satisfiable — the original failure was likely a solver timeout, not true infeasibility |
| `inconclusive` | The diagnostic itself timed out (30 s limit); increase `max_time_in_seconds` or `num_workers` |
| `no_deadlines` | Infeasibility with no deadline constraints present is unexpected; indicates malformed input data (e.g. zero or negative processing times) |

---

### 4.4 Code

The scheduling logic lives in two files:

- **`scheduler.py`** — builds the CP-SAT model, applies constraints and the objective, invokes the solver, and writes a timestamped human-readable report to `outputs/`.
- **`scheduler_infeasibility_analysis.py`** — contains `InfeasibilityAnalysisMixin`, mixed into the scheduler class to add the assumption-literal diagnostic when the solver returns no solution.

**Tests** live in [tests/test_scheduler.py](tests/test_scheduler.py) and run against small, hand-crafted scenarios defined in [tests/test_jobs.json](tests/test_jobs.json). 

| Test | What it checks |
|------|---------------|
| `test_no_machine_overlap` | No two operations assigned to the same machine overlap in time. All three jobs visit machine 0, creating forced contention. |
| `test_no_preemption` | Every operation's scheduled duration equals its declared `processing_time` — the solver never splits or shortens a task. |
| `test_operation_precedence` | Within each job, every operation starts only after the preceding one finishes. Jobs visit machines in different orders so the check crosses machine boundaries. |
| `test_makespan_minimisation` | The solver finds the provably optimal makespan. A 2-job, 2-machine instance is used where the optimal (makespan = 5) can be confirmed by hand. |
| `test_deadline_respected` | Jobs with a deadline have their last operation finish at or before that deadline. |
| `test_machine_capacity` | Each machine processes at most one job at a time under maximum contention. Four jobs all routed through a single machine; the only valid makespan equals total work (10), confirming full serialisation and pairwise non-overlap. |

Test fixture data is stored as named keys in `tests/test_jobs.json`.

---

## 5. On the JSSP Agent

### 5.1 Agent Design

The agent is a **LangGraph ReAct agent** — it loops between thinking and acting until it produces a plain-text reply.

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


| Tool | What it does |
|------|-------------|
| `run_scheduler` | Invokes the CP-SAT solver and returns the formatted report |
| `update_machine_status` | Marks a machine `online` or `offline` in `jobs.json` |
| `handle_priority` | Sets or removes the numeric priority on a job |
| `handle_deadline` | Sets or removes the hard deadline on a job |
| `get_latest_scheduler_report` | Re-fetches the most recent report without re-solving |

**Scheduler re-run guard** — A module-level `_scheduler_called` flag is reset to `False` on each new human message and set to `True` the first time `run_scheduler` fires. If the tool is called a second time in the same turn it returns an early-exit string instead of solving again, preventing redundant solves during multi-step tool chains.

**Observability** — Every session gets a UUID and is traced end-to-end via Langfuse, tagged `jssp` and `cli` (or `streamlit` from the UI). If the Langfuse keys are absent the callback silently no-ops.

---

### 5.2 Agent Code

- **`agent.py`** — defines the five tools, builds the LangGraph graph, sets up the Langfuse callback, and runs the agent loop. Can be run directly as a CLI (`uv run python agent.py`).
- **`ui.py`** — Streamlit chat UI. Wraps the same agent graph with a browser interface and adds a sidebar that reads `data/jobs.json` on every refresh to show live machine statuses and job details.

---

## 6. Set Up & Execution

### 6.1 Prerequisites

- **Python 3.12+**
- **uv** — fast Python package manager. Makes it very easy to setup the project and dependencies.
Install via the [official guide](https://docs.astral.sh/uv/getting-started/installation/).
- **docker**
- **Langfuse** can be self-hosted via docker or cloud-based via API keys can be used. 
Refer https://langfuse.com/docs/observability/get-started



### 6.2 Set Up

**Clone the repository:**

```bash
git clone https://github.com/niv22/Job-Shop-Scheduling-Agent/
cd JSSP
```

**Install dependencies:**

```bash
uv sync
```

**Create a `.env` file** refer the .env.example

---

**Run the Streamlit UI:**

```bash
uv run streamlit run ui.py
```

**Run the agent as a CLI:**

```bash
uv run python agent.py
```

**Run the scheduler directly (no agent):**

```bash
uv run python scheduler.py
```

**Run the test suite:**

```bash
uv run pytest tests/
```
