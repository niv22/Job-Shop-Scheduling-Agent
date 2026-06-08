import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ortools.sat.python import cp_model
from scheduler import JSSP


def _load(scenario):
    with open("tests/test_jobs.json") as f:
        return json.load(f)[scenario]


def _solve(jobs_data):
    """Return (obj, solver, machine_to_tasks) for constraint-level tests."""
    obj = JSSP(jobs_data)
    solver, status = obj.solve()
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), "Solver returned no solution"
    return obj, solver, obj.get_machine_to_tasks(solver)


def test_no_machine_overlap():
    """H1: No two tasks on the same machine overlap in time.

    3 jobs, 3 machines — all three jobs visit machine 0, creating forced contention.
    Job0: M0(3) -> M1(2) -> M2(1)
    Job1: M2(4) -> M0(2) -> M1(1)
    Job2: M1(3) -> M2(2) -> M0(2)
    """
    _, _, machine_to_tasks = _solve(_load("test_no_machine_overlap"))

    for machine, tasks in machine_to_tasks.items():
        intervals = [(t["start"], t["start"] + t["duration"], t["label"]) for t in tasks]
        for i, (s1, e1, name1) in enumerate(intervals):
            for s2, e2, name2 in intervals[i + 1:]:
                assert e1 <= s2 or e2 <= s1, (
                    f"Machine {machine}: {name1} [{s1},{e1}) overlaps {name2} [{s2},{e2})"
                )


def test_no_preemption():
    """H2: Every task's scheduled duration matches its declared processing_time.

    4 jobs, 2 machines with varying durations to exercise the constraint broadly.
    Job0: M0(1) -> M1(4)
    Job1: M1(2) -> M0(3)
    Job2: M0(5) -> M1(1)
    Job3: M1(3) -> M0(2)
    """
    jobs_data = _load("test_no_preemption")
    _, _, machine_to_tasks = _solve(jobs_data)

    declared = {
        f"{job['name']} Op{task_id}": op["processing_time"]
        for job in jobs_data["jobs"]
        for task_id, op in enumerate(job["operations"])
    }
    for tasks in machine_to_tasks.values():
        for t in tasks:
            expected = declared[t["label"]]
            assert t["duration"] == expected, (
                f"{t['label']}: scheduled duration {t['duration']} != declared {expected}"
            )


def test_makespan_minimisation():
    """S1: Solver finds the provably optimal makespan.

    2 jobs, 2 machines — small enough to verify the optimal by hand.
    Job0: M0(2) -> M1(3)
    Job1: M1(2) -> M0(1)

    Optimal schedule (makespan = 5):
      M0: Job0Op0 [0,2]  Job1Op1 [2,3]
      M1: Job1Op0 [0,2]  Job0Op1 [2,5]
    """
    _, _, machine_to_tasks = _solve(_load("test_makespan_minimisation"))

    makespan = max(
        t["start"] + t["duration"]
        for tasks in machine_to_tasks.values()
        for t in tasks
    )
    assert makespan == 5, f"Expected optimal makespan 5, got {makespan}"


def test_operation_precedence():
    """H3: Within each job, every operation starts only after the preceding one ends.

    3 jobs, 3 machines — each job visits all three machines in a different order,
    so the precedence chain must hold across machine-boundary crossings.
    Job0: M0(2) -> M1(3) -> M2(1)
    Job1: M1(1) -> M2(2) -> M0(3)
    Job2: M2(2) -> M0(1) -> M1(2)
    """
    jobs_data = _load("test_operation_precedence")
    _, _, machine_to_tasks = _solve(jobs_data)

    # Reconstruct per-job op ordering from task labels ("JobX OpN")
    job_ops: dict[str, list[tuple[int, int, int]]] = {}
    for tasks in machine_to_tasks.values():
        for t in tasks:
            job_name, op_part = t["label"].rsplit(" Op", 1)
            job_ops.setdefault(job_name, []).append(
                (int(op_part), t["start"], t["duration"])
            )

    for job_name, ops in job_ops.items():
        ops.sort()  # sort by op index
        for i in range(len(ops) - 1):
            op_idx, s, d = ops[i]
            next_idx, s_next, _ = ops[i + 1]
            assert s + d <= s_next, (
                f"{job_name}: Op{op_idx} ends at {s + d} but Op{next_idx} starts at {s_next} "
                f"(precedence violated)"
            )


def test_deadline_respected():
    """D1: A job's last operation must finish at or before its declared deadline.

    2 jobs, 2 machines, both with deadline=10.
    Job0: M0(2) -> M1(3)   min completion = 5
    Job1: M1(1) -> M0(2)   min completion = 3
    Both deadlines are comfortably achievable; the solver must honour them.
    """
    jobs_data = _load("test_deadline_respected")
    _, _, machine_to_tasks = _solve(jobs_data)

    deadline_by_job = {j["name"]: j["deadline"] for j in jobs_data["jobs"] if "deadline" in j}

    # Find the last-op finish time per job
    last_finish: dict[str, int] = {}
    for tasks in machine_to_tasks.values():
        for t in tasks:
            job_name = t["label"].rsplit(" Op", 1)[0]
            finish = t["start"] + t["duration"]
            if job_name not in last_finish or finish > last_finish[job_name]:
                last_finish[job_name] = finish

    for job_name, deadline in deadline_by_job.items():
        finish = last_finish[job_name]
        assert finish <= deadline, (
            f"{job_name}: completed at {finish} which exceeds deadline {deadline}"
        )


def test_machine_capacity():
    """H1: Each machine processes at most one job at a time (no overlap).

    4 jobs all routed through a single machine — the strictest possible capacity
    test because every pair of tasks competes for the same resource.
    Job0: M0(2)  
    Job1: M0(3)  
    Job2: M0(1)  
    Job3: M0(4)
    Total work = 10, so the only valid makespan is 10 (full serialization).
    Pairwise non-overlap is verified for every combination of tasks.
    """
    _, _, machine_to_tasks = _solve(_load("test_machine_capacity"))

    tasks = machine_to_tasks[0]
    total_work = sum(t["duration"] for t in tasks)
    makespan = max(t["start"] + t["duration"] for t in tasks)

    # True serialization: no idle gaps would make makespan exactly equal total work
    assert makespan == total_work, (
        f"Expected makespan {total_work} (full serialization), got {makespan}"
    )

    # Pairwise non-overlap on machine 0
    intervals = [(t["start"], t["start"] + t["duration"], t["label"]) for t in tasks]
    for i, (s1, e1, name1) in enumerate(intervals):
        for s2, e2, name2 in intervals[i + 1:]:
            assert e1 <= s2 or e2 <= s1, (
                f"Machine 0: {name1} [{s1},{e1}) overlaps {name2} [{s2},{e2})"
            )

