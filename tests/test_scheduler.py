import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import JSSP



def _load(scenario):
    with open("tests/test_jobs.json") as f:
        return json.load(f)[scenario]


def test_no_machine_overlap():
    """H1: No two tasks on the same machine overlap in time.

    3 jobs, 3 machines — all three jobs visit machine 0, creating forced contention.
    Job0: M0(3) -> M1(2) -> M2(1)
    Job1: M2(4) -> M0(2) -> M1(1)
    Job2: M1(3) -> M2(2) -> M0(2)
    """
    obj = JSSP(_load("test_no_machine_overlap"))
    machine_to_tasks = obj.run()
    del obj  # Free memory from the potentially large CP model
    assert machine_to_tasks, "Solver returned no solution"

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
    machine_to_tasks = JSSP(jobs_data).run()
    assert machine_to_tasks, "Solver returned no solution"

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
    machine_to_tasks = JSSP(_load("test_makespan_minimisation")).run()
    assert machine_to_tasks, "Solver returned no solution"

    makespan = max(
        t["start"] + t["duration"]
        for tasks in machine_to_tasks.values()
        for t in tasks
    )
    assert makespan == 5, f"Expected optimal makespan 5, got {makespan}"
