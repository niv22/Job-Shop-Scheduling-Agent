import collections
import logging
from datetime import datetime
from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


class InfeasibilityAnalysisMixin:
    def diagnose_infeasibility(self):
        """Re-solve a copy of the model with each deadline guarded by an
        assumption literal and NO objective, then read back the minimal set of
        deadlines the solver proved cannot hold together. Returns a markdown
        report string.

        Technique from google/or-tools#973:
          - Background model (precedence + no_overlap) is always enforced. On
            its own it is always satisfiable (serialise everything inside the
            horizon), so it can never be the cause of infeasibility.
          - Each deadline (a linear constraint, which supports enforcement
            literals — unlike no_overlap) is guarded by one assumption literal.
          - Solving WITHOUT an objective lets sufficient_assumptions_for_infeasibility()
            return a small conflicting subset. With an objective set it would
            instead return ALL assumptions, so we deliberately omit it here.
        """
        model = cp_model.CpModel()
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        # ---- Background model: always enforced, never the culprit ----
        for job in self.jobs:
            job_id = job["id"]
            for task_id, op in enumerate(job["operations"]):
                suffix = f"_{job_id}_{task_id}"
                start_var = model.new_int_var(0, self.horizon, "start" + suffix)
                end_var = model.new_int_var(0, self.horizon, "end" + suffix)
                interval_var = model.new_interval_var(
                    start_var, op["processing_time"], end_var, "interval" + suffix
                )
                all_tasks[job_id, task_id] = self.task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[op["machine_id"]].append(interval_var)

        for machine in self.all_machines:
            model.add_no_overlap(machine_to_intervals[machine])

        for job in self.jobs:
            job_id = job["id"]
            for task_id in range(len(job["operations"]) - 1):
                model.add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

        # ---- Active model: one assumption literal per deadline ----
        lit_to_job = {}
        assumptions = []
        for job in self.jobs:
            if "deadline" not in job:
                continue
            job_id = job["id"]
            last_task = len(job["operations"]) - 1
            lit = model.new_bool_var(f"deadline_ok_{job_id}")
            model.add(all_tasks[job_id, last_task].end <= job["deadline"]).only_enforce_if(lit)
            assumptions.append(lit)
            lit_to_job[lit.index] = job

        if not assumptions:
            return self._render_infeasibility_report(core=[], status="no_deadlines")

        model.add_assumptions(assumptions)

        # NO objective on purpose (see docstring).
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = 30
        status = solver.solve(model)
        logger.info("Diagnostic solve finished: status=%s", solver.status_name(status))

        if status == cp_model.INFEASIBLE:
            core = [lit_to_job[i] for i in solver.sufficient_assumptions_for_infeasibility()]
            return self._render_infeasibility_report(core=core, status="infeasible")
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._render_infeasibility_report(core=[], status="deadlines_ok")
        return self._render_infeasibility_report(
            core=[], status="inconclusive", solver_status=solver.status_name(status)
        )

    def _render_infeasibility_report(self, core, status, solver_status=None):
        nl = "\n"
        deadline_jobs = [j for j in self.jobs if "deadline" in j]
        out = []

        out.append("# Infeasibility Analysis Report")
        out.append("")
        out.append(f"- **Run ID:** {self.run_id}")
        out.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        out.append(f"- **Jobs analysed:** {len(self.jobs)}")
        out.append(f"- **Machines:** {self.machines_count}")
        out.append("")

        out.append("## Verdict")
        out.append("")
        if status == "no_deadlines":
            out.append(
                "The model is infeasible but **no deadline constraints are present**. "
                "Precedence and machine no-overlap constraints alone cannot make this model "
                "infeasible (every task can always be run back-to-back within the horizon), so "
                "this is unexpected. Check the input data — e.g. negative or zero processing "
                "times, or malformed operations."
            )
            return nl.join(out) + nl
        if status == "deadlines_ok":
            out.append(
                "The deadlines are **jointly satisfiable**. The original failure was therefore "
                "not caused by conflicting deadlines — the solver most likely **timed out** "
                "(status UNKNOWN within the 30s limit) rather than proving infeasibility. "
                "Consider increasing `max_time_in_seconds` or `num_workers`."
            )
            return nl.join(out) + nl
        if status == "inconclusive":
            out.append(
                f"The diagnostic solve did not finish (status `{solver_status}`) within the time "
                "limit, so the conflicting deadlines could not be isolated. Try raising the "
                "diagnostic time limit and re-running."
            )
            return nl.join(out) + nl

        # status == "infeasible"
        out.append("The schedule is **INFEASIBLE because of deadline constraints**.")
        out.append("")
        out.append(
            "> **Why deadlines?** Without deadlines, the precedence and machine no-overlap "
            "constraints can always be satisfied by running every task back-to-back within the "
            f"horizon ({self.horizon} time units). Deadlines are therefore the only thing that "
            "can make this model infeasible, and the solver confirmed it below."
        )
        out.append("")

        # --- Conflicting set ---
        core_names = {j["name"] for j in core}
        out.append("## Conflicting deadlines (minimal infeasible set)")
        out.append("")
        out.append(
            "The solver proved the following deadlines **cannot all be met at the same time**. "
            "Relaxing (extending) the deadline on *any one* of these jobs is usually enough to "
            "break the conflict:"
        )
        out.append("")
        if core:
            for j in core:
                mc = sum(op["processing_time"] for op in j["operations"])
                out.append(f"- **{j['name']}** — deadline = {j['deadline']}, own minimum completion = {mc}")
        else:
            out.append("- _(the solver reported infeasibility but returned no specific core; "
                       "see the tables below for the likely cause)_")
        out.append("")
        out.append(
            "> Note: this is a *sufficient* explanation produced heuristically — it is a genuine "
            "conflicting group, but not guaranteed to be the single smallest possible one."
        )
        out.append("")

        # --- Per-job deadline feasibility ---
        out.append("## Per-job deadline feasibility")
        out.append("")
        out.append(
            "`Min completion` is the sum of a job's own operation times — the earliest it could "
            "finish even with every machine free. A deadline below that value is impossible on "
            "its own, regardless of the other jobs."
        )
        out.append("")
        out.append("| Job | Deadline | Min completion | Individually feasible | In conflict set |")
        out.append("|-----|---------:|---------------:|:---------------------:|:---------------:|")
        for j in sorted(deadline_jobs, key=lambda x: x["deadline"]):
            mc = sum(op["processing_time"] for op in j["operations"])
            feasible = "yes" if j["deadline"] >= mc else "**NO**"
            inset = "✓" if j["name"] in core_names else ""
            out.append(f"| {j['name']} | {j['deadline']} | {mc} | {feasible} | {inset} |")
        out.append("")

        out.append("")
        deadlines = sorted({j["deadline"] for j in deadline_jobs})
        overloads = []
        for machine in self.all_machines:
            for d in deadlines:
                load = sum(
                    op["processing_time"]
                    for job in deadline_jobs if job["deadline"] <= d
                    for op in job["operations"] if op["machine_id"] == machine
                )
                if load > d:
                    culprits = [
                        job["name"] for job in deadline_jobs
                        if job["deadline"] <= d
                        and any(op["machine_id"] == machine for op in job["operations"])
                    ]
                    overloads.append((machine, d, load, culprits))

        if overloads:
            out.append("| Machine | Deadline d | Work due by d | Available | Over by | Jobs involved |")
            out.append("|--------:|-----------:|--------------:|----------:|--------:|---------------|")
            for machine, d, load, culprits in overloads:
                out.append(f"| {machine} | {d} | {load} | {d} | {load - d} | {', '.join(culprits)} |")
            out.append("")
            out.append("Each overloaded row above is a hard bottleneck that no schedule can resolve.")
        else:
            out.append(
                "No single machine is overloaded by the earliest-deadline-first test. The conflict "
                "therefore comes from the **combination** of precedence ordering and machine "
                "sharing across the jobs in the conflict set above, rather than one machine being "
                "individually overbooked."
            )
        out.append("")

        # --- Recommendations ---
        out.append("## Recommended fixes")
        out.append("")
        recs = []
        for j in deadline_jobs:
            mc = sum(op["processing_time"] for op in j["operations"])
            if j["deadline"] < mc:
                recs.append(
                    f"Job **{j['name']}** is impossible on its own — its deadline ({j['deadline']}) "
                    f"is below its own minimum completion time ({mc}). Raise its deadline to at "
                    f"least **{mc}**."
                )
        for machine, d, load, culprits in overloads:
            recs.append(
                f"Machine {machine} needs {load} units done by {d} but only has {d}. Push the "
                f"deadlines of {', '.join(culprits)} out by at least **{load - d}** units, move some "
                "of their operations to another machine, or reduce their processing times."
            )
        if not recs:
            recs.append(
                "Extend the deadline of any one job in the conflict set, or rebalance its "
                "operations onto less-contended machines."
            )
        for r in recs:
            out.append(f"- {r}")
        out.append("")

        return nl.join(out) + nl
