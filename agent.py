"""LangGraph agent for interactive job-shop scheduling."""
import copy
import json
import logging
import uuid
from typing import Annotated
import os

from dotenv import load_dotenv
load_dotenv()

from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from scheduler import JSSP

logger = logging.getLogger(__name__)

with open("data/jobs.json") as f:
    _ORIGINAL: dict = json.load(f)

_jobs_data: dict = copy.deepcopy(_ORIGINAL)
_last_run_id: str = ""
_scheduler_called: bool = False


def _save_jobs_data() -> None:
    with open("data/jobs.json", "w") as f:
        json.dump(_jobs_data, f, indent=2)

def _open_jobs_data() -> None:
    with open("data/jobs.json") as f:
        jobs_data = json.load(f)
    return jobs_data


def _find_job(job_id: int) -> dict | None:
    return next((j for j in _jobs_data["jobs"] if j["id"] == job_id), None)


def _find_machine(machine_id: int) -> dict | None:
    return next((m for m in _jobs_data["machines"] if m["id"] == machine_id), None)


def _utilization_bar(pct: float, width: int = 10) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _report_to_markdown(report: str) -> str:
    lines = report.splitlines()
    out: list[str] = []
    section: str | None = None
    sub_section: str | None = None
    schedule_lines: list[str] = []
    key_metrics: dict[str, str] = {}
    util_rows: list[tuple] = []
    completion_rows: list[dict] = []

    def flush_schedule() -> None:
        if schedule_lines:
            out.append("```")
            out.extend(schedule_lines)
            out.append("```\n")
            schedule_lines.clear()

    def flush_metrics() -> None:
        if key_metrics:
            status = key_metrics.get("Status", "")
            icon = "✅" if status == "OPTIMAL" else "⚠️"
            out.append("| Makespan | Optimality Gap | Status |")
            out.append("|:---:|:---:|:---:|")
            out.append(
                f"| **{key_metrics.get('Makespan', '—')}** "
                f"| {key_metrics.get('Optimality gap', '—')} "
                f"| {icon} {status} |"
            )
            out.append("")
            key_metrics.clear()

        if util_rows:
            out.append("**Machine Utilization**\n")
            out.append("| Machine | Busy / Total | Utilization |")
            out.append("|---------|:---:|---|")
            for name, busy_total, pct_str in util_rows:
                pct = float(pct_str.strip("%"))
                bar = _utilization_bar(pct)
                if busy_total:
                    out.append(f"| {name} | {busy_total} | {bar} {pct_str} |")
                else:
                    out.append(f"| **{name}** | — | {bar} {pct_str} |")
            out.append("")
            util_rows.clear()

        if completion_rows:
            has_priority = any("priority" in r for r in completion_rows)
            has_deadline = any("deadline" in r for r in completion_rows)
            cols = ["Job", "Completion", "Processing", "Wait"]
            if has_priority:
                cols.append("Priority")
            if has_deadline:
                cols.append("Deadline")
            out.append("**Job Completion Times**\n")
            out.append("| " + " | ".join(cols) + " |")
            out.append("|" + "---|" * len(cols))
            for row in completion_rows:
                cells = [
                    row["name"],
                    row.get("completion", "—"),
                    row.get("processing", "—"),
                    row.get("wait", "—"),
                ]
                if has_priority:
                    cells.append(row.get("priority", "—"))
                if has_deadline:
                    if "deadline" in row:
                        val = row["deadline"]
                        if "MISSED" in val:
                            cells.append(f"❌ {val}")
                        elif "MET" in val:
                            cells.append(f"✅ {val}")
                        else:
                            cells.append(val)
                    else:
                        cells.append("—")
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
            completion_rows.clear()

    for line in lines:
        stripped = line.strip()

        if line.startswith("Run ID") or line.startswith("Date"):
            key, _, val = line.partition(" : ")
            out.append(f"*{key.strip()}: {val.strip()}*  ")
            continue

        if line.startswith("=== ") and line.endswith(" ==="):
            flush_schedule()
            flush_metrics()
            section = line[4:-4]
            sub_section = None
            out.append(f"\n### {section}\n")
            continue

        if not stripped:
            continue

        if section == "Schedule Summary":
            schedule_lines.append(line)
        elif section == "Schedule Metrics":
            if stripped == "Machine utilization:":
                sub_section = "utilization"
            elif stripped == "Job completion times:":
                sub_section = "completion"
            elif sub_section is None and " : " in line:
                key, _, val = line.partition(" : ")
                key_metrics[key.strip()] = val.strip()
            elif sub_section == "utilization":
                if "Overall" in stripped:
                    _, _, pct = stripped.partition(": ")
                    util_rows.append(("Overall", None, pct.strip()))
                elif ":" in stripped:
                    name, _, rest = stripped.partition(": ")
                    parts = rest.split()
                    busy_total = parts[0] if parts else "—"
                    pct_str = parts[1].strip("()") if len(parts) > 1 else "—"
                    util_rows.append((name.strip(), busy_total, pct_str))
            elif sub_section == "completion":
                name, _, attrs_str = stripped.partition(": ")
                attrs: dict[str, str] = {}
                for attr in attrs_str.split(", "):
                    if "=" in attr:
                        k, _, v = attr.partition("=")
                        attrs[k.strip()] = v.strip()
                attrs["name"] = name
                completion_rows.append(attrs)

    flush_schedule()
    flush_metrics()
    return "\n".join(out)


@tool
def run_scheduler() -> str:
    """Run the job-shop scheduler with the current jobs data and return the schedule."""
    global _last_run_id, _scheduler_called
    if _scheduler_called:
        return "Scheduler already ran this turn. Ask the user before running it again."
    _scheduler_called = True
    jssp = JSSP(_jobs_data)
    result = jssp.run()
    if jssp.run_id:
        _last_run_id = jssp.run_id
        path = f"outputs/{jssp.run_id}.txt"
        if os.path.exists(path):
            with open(path) as f:
                return _report_to_markdown(f.read())
        status = "dispalyed report"
    else:
        status = "could not fetch scheduling report"
    return status


@tool
def update_machine_status(machine_id: int, status: str) -> str:
    """Set a machine online or offline. status must be 'online' or 'offline'."""
    if status not in ("online", "offline"):
        return f"Invalid status '{status}'. Use 'online' or 'offline'."
    machine = _find_machine(machine_id)
    if machine is None:
        return f"Machine {machine_id} not found."
    machine["status"] = status
    _save_jobs_data()
    return f"Machine {machine_id} is now {status}."


@tool
def handle_priority(job_id: int, priority: int | None = None) -> str:
    """Set or remove the priority of a job. Pass priority=None to remove it."""
    job = _find_job(job_id)
    if job is None:
        return f"Job {job_id} not found."
    if priority is None:
        job.pop("priority", None)
        _save_jobs_data()
        return f"Priority removed from {job['name']}."
    job["priority"] = priority
    _save_jobs_data()
    return f"{job['name']} priority set to {priority}."


@tool
def handle_deadline(job_id: int, deadline: int | None = None) -> str:
    """Add, modify, or remove a deadline for a job. Pass deadline=None to remove it."""
    job = _find_job(job_id)
    if job is None:
        return f"Job {job_id} not found."
    if deadline is None:
        job.pop("deadline", None)
        _save_jobs_data()
        return f"Deadline removed from {job['name']}."
    job["deadline"] = deadline
    _save_jobs_data()
    return f"{job['name']} deadline set to {deadline}."


@tool
def get_latest_scheduler_report() -> str:
    """fetch the latest scheduler report"""
    path = f"outputs/{_last_run_id}.txt"
    if os.path.exists(path):
        with open(path) as f:
            return _report_to_markdown(f.read())
    else:
        return "No report found"


# --- Graph ---

_tools = [run_scheduler, update_machine_status, handle_priority, handle_deadline, get_latest_scheduler_report]
_llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(_tools)

_SYSTEM = SystemMessage(content=(
    "You are a scheduling assistant for a project manager. "
    "Your job is to understand the user's intent and perform the related task from your capabilities.\n\n"
    "Available actions:\n"
    "- Run the scheduler to get the current schedule\n"
    "- Mark machines online or offline (e.g. when one goes down for maintenance)\n"
    "- Set or adjust job priorities (higher number = higher priority)\n"
    "- Set or remove deadlines on jobs\n"
    "Instructions"
    "- Do not run tools sequentially on your own without getting user's input"
    "- After any configuration change, ask the user if the schedule needs to be calculated"
    "- Explain why scheduling failed when no solution is found\n\n"
    "How to respond:\n"
    "- After running the scheduler, always call out: which jobs will miss their deadline (if any), "
    "and which machines are the bottlenecks.\n"
    "- When the user asks a what-if question (e.g. 'what if Machine 2 goes offline?'), "
    "apply the change, run the scheduler, and summarise the impact — don't just describe what you did.\n"
    "- When the scheduler returns no solution, immediately pull the latest report, examine the logs and report "
    "the root cause in plain language (impossible deadlines, machine overload, etc.) and suggest a fix.\n"
    "- Speak in business terms: completion times, deadline risk, resource conflicts. "
    "- Be concise. Lead with the answer, follow with supporting detail only if it matters."
))


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_llm(state: State) -> State:
    global _scheduler_called
    if state["messages"][-1].type == "human":
        _scheduler_called = False
    return {"messages": [_llm.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(_tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()
langfuse_handler = LangfuseCallbackHandler()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("JSSP Agent ready. Type 'quit' to exit.\n")

    state: State = {"messages": [_SYSTEM]}
    session_id = str(uuid.uuid4())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        state["messages"].append(HumanMessage(content=user_input))

        config = {
            "recursion_limit": 10,
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": ["jssp", "cli"],
            },
        }
        for step in app.stream(state, config=config, stream_mode="values"):
            last = step["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  [calling] {tc['name']}()")
            elif last.type == "tool":
                print(f"  [result] {last.content[:300]}")
            state = step

        print(f"\nAgent: {state['messages'][-1].content}\n")
