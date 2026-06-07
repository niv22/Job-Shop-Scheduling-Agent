"""Minimal jobshop example."""
import collections
import logging
import os
from datetime import datetime
from ortools.sat.python import cp_model

from scheduler_infeasibility_analysis import InfeasibilityAnalysisMixin

logger = logging.getLogger(__name__)

class JSSP(InfeasibilityAnalysisMixin):
    def __init__(self, jobs_data):
        self.task_type = collections.namedtuple("task_type", "start end interval")
        self.assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )
        self.has_priorities = any("priority" in j for j in jobs_data.get("jobs", []))
        self.has_deadlines = any("deadline" in j for j in jobs_data.get("jobs", []))
        self.run_time = datetime.now()
        self.run_id = self.run_time.strftime("%Y%m%d_%H%M%S")
        offline_machines = {
            m["id"] for m in jobs_data.get("machines", []) if m.get("status") != "online"
        }
        if offline_machines:
            logger.warning("Offline machines detected: %s", sorted(offline_machines))

        all_jobs = jobs_data["jobs"]
        dropped = [j for j in all_jobs if any(op["machine_id"] in offline_machines for op in j["operations"])]
        if dropped:
            logger.warning(
                "Dropping %d job(s) due to offline machines: %s",
                len(dropped),
                [j["name"] for j in dropped],
            )
        self.jobs = [j for j in all_jobs if j not in dropped]
        if not self.jobs:
            raise ValueError(
                "No schedulable jobs remain — all jobs were dropped due to offline machines or the job list is empty."
            )
        self.machines_count = 1 + max(
            op["machine_id"] for job in self.jobs for op in job["operations"]
        )
        self.all_machines = range(self.machines_count)
        self.horizon = sum(op["processing_time"] for job in self.jobs for op in job["operations"])
        self.model = cp_model.CpModel()
        self.all_tasks = {}
        self._check_deadline_feasibility()
        logger.debug(
            "Initialized JSSP: %d jobs, %d machines, horizon=%d, priority_mode=%s, deadline_mode=%s",
            len(self.jobs), self.machines_count, self.horizon, self.has_priorities, self.has_deadlines,
        )

    def _check_deadline_feasibility(self):
        deadline_jobs = [j for j in self.jobs if "deadline" in j]
        if not deadline_jobs:
            return

        # Job-level check: deadline < own minimum completion time
        for job in deadline_jobs:
            min_completion = sum(op["processing_time"] for op in job["operations"])
            if job["deadline"] < min_completion:
                logger.warning(
                    "Job '%s' deadline=%d is impossible (min completion=%d) — will be INFEASIBLE",
                    job["name"], job["deadline"], min_completion,
                )

        # Machine-level EDF check: for each machine and each deadline d, total
        # processing time on that machine across all jobs with deadline <= d must fit within d.
        # Individually feasible deadlines can still conflict via shared machine contention.
        deadlines = sorted({j["deadline"] for j in deadline_jobs})
        for machine in self.all_machines:
            for d in deadlines:
                load = sum(
                    op["processing_time"]
                    for job in deadline_jobs
                    if job["deadline"] <= d
                    for op in job["operations"]
                    if op["machine_id"] == machine
                )
                if load > d:
                    culprits = [j["name"] for j in deadline_jobs if j["deadline"] <= d
                                and any(op["machine_id"] == machine for op in j["operations"])]
                    logger.warning(
                        "Machine %d overloaded by deadline=%d: %d units of work needed but only %d available — "
                        "conflicting jobs: %s — will be INFEASIBLE",
                        machine, d, load, d, culprits,
                    )

    def get_machine_to_tasks(self, solver):
        machine_to_tasks = collections.defaultdict(list)
        for job in self.jobs:
            job_id = job["id"]
            for task_id, op in enumerate(job["operations"]):
                machine = op["machine_id"]
                machine_to_tasks[machine].append({
                    "label": f"{job['name']} Op{task_id}",
                    "start": solver.Value(self.all_tasks[job_id, task_id].start),
                    "duration": op["processing_time"],
                })
        return machine_to_tasks

    def format_summary(self, machine_to_tasks):
        if machine_to_tasks == {}:
            return f"Scheduling was not feasible. Check output report {self.run_id}.txt"
        lines = ["=== Schedule Summary ==="]
        for machine in self.all_machines:
            tasks = sorted(machine_to_tasks[machine], key=lambda x: x["start"])
            seq = "  →  ".join(f"{t['label']}({t['duration']})" for t in tasks)
            lines.append(f"  Machine {machine}:  {seq}")
        lines.append("")
        return "\n".join(lines)

    def format_metrics(self, solver):
        total_processing = sum(op["processing_time"] for job in self.jobs for op in job["operations"])
        makespan = max(
            solver.Value(self.all_tasks[job["id"], len(job["operations"]) - 1].end)
            for job in self.jobs
        )
        lines = []

        lines.append("=== Schedule Metrics ===")
        lines.append(f"Makespan       : {makespan}")
        obj_val = solver.objective_value
        lines.append(f"Optimality gap : {100 * (obj_val - solver.best_objective_bound) / obj_val:.1f}%")
        lines.append(f"Status         : {'OPTIMAL' if solver.objective_value == solver.best_objective_bound else 'FEASIBLE'}")
        lines.append("")

        lines.append("Machine utilization:")
        for machine in self.all_machines:
            busy = sum(
                op["processing_time"]
                for job in self.jobs
                for op in job["operations"]
                if op["machine_id"] == machine
            )
            lines.append(f"  Machine {machine}: {busy}/{makespan} ({100 * busy / makespan:.1f}%)")
        overall = total_processing / (makespan * self.machines_count)
        lines.append(f"  Overall      : {100 * overall:.1f}%")
        lines.append("")

        lines.append("Job completion times:")
        for job in self.jobs:
            job_id = job["id"]
            last_task = len(job["operations"]) - 1
            completion = solver.Value(self.all_tasks[job_id, last_task].end)
            total_proc = sum(op["processing_time"] for op in job["operations"])
            wait = completion - total_proc
            priority_str = f", priority={job['priority']}" if "priority" in job else ""
            deadline_str = ""
            if "deadline" in job:
                dl = job["deadline"]
                status = "MET" if completion <= dl else "MISSED"
                deadline_str = f", deadline={dl} ({status})"
            lines.append(f"  {job['name']}: completion={completion}, processing={total_proc}, wait={wait}{priority_str}{deadline_str}")
        lines.append("")

        return "\n".join(lines)

    def solve(self):
        logger.info("Building model: %d jobs, %d machines", len(self.jobs), self.machines_count)
        machine_to_intervals = collections.defaultdict(list)

        for job in self.jobs:
            job_id = job["id"]
            for task_id, op in enumerate(job["operations"]):
                machine = op["machine_id"]
                duration = op["processing_time"]
                suffix = f"_{job_id}_{task_id}"
                start_var = self.model.new_int_var(0, self.horizon, "start" + suffix)
                end_var = self.model.new_int_var(0, self.horizon, "end" + suffix)
                interval_var = self.model.new_interval_var(
                    start_var, duration, end_var, "interval" + suffix
                )
                self.all_tasks[job_id, task_id] = self.task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        for machine in self.all_machines:
            self.model.add_no_overlap(machine_to_intervals[machine])

        for job in self.jobs:
            job_id = job["id"]
            ops = job["operations"]
            for task_id in range(len(ops) - 1):
                self.model.add(
                    self.all_tasks[job_id, task_id + 1].start >= self.all_tasks[job_id, task_id].end
                )

        for job in self.jobs:
            if "deadline" in job:
                job_id = job["id"]
                last_task = len(job["operations"]) - 1
                self.model.add(self.all_tasks[job_id, last_task].end <= job["deadline"])

        if self.has_priorities:
            self.model.minimize(sum(
                job.get("priority", 1) * self.all_tasks[job["id"], len(job["operations"]) - 1].end
                for job in self.jobs
            ))
        else:
            obj_var = self.model.new_int_var(0, self.horizon, "makespan")
            self.model.add_max_equality(
                obj_var,
                [self.all_tasks[job["id"], len(job["operations"]) - 1].end for job in self.jobs],
            )
            self.model.minimize(obj_var)

        logger.info("Solving... (objective=%s)", "weighted_completion" if self.has_priorities else "makespan")
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = 30
        status = solver.solve(self.model)
        status_name = solver.status_name(status)
        logger.info("Solver finished: status=%s, objective=%s", status_name, int(solver.objective_value) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else "N/A")
        return solver, status
    
    def save_output_report(self, output_str):
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/{self.run_id}.txt"
        with open(output_path, "w") as f:
            f.write(output_str)
        logger.info("Output report saved to %s", output_path)

    def run(self):
        solver, status = self.solve()

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info("Solution found: objective=%s", int(solver.objective_value))
            header = f"Run ID : {self.run_id}\nDate   : {self.run_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            machine_to_tasks = self.get_machine_to_tasks(solver)
            output = header + self.format_summary(machine_to_tasks) + "\n" + self.format_metrics(solver)
        else:
            logger.warning("No solution found (status=%s)", solver.status_name(status))
            # Infeasible (or otherwise unsolved): run the analysis and write a markdown report.
            output = self.diagnose_infeasibility()

        self.save_output_report(output)

        return output

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    jobs_data = json.load(open("data/jobs.json"))
    jssp = JSSP(jobs_data)
    output = jssp.run()