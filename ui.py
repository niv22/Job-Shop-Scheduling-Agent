"""Streamlit chatbot UI for the JSSP agent."""
import json
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import _SYSTEM, app, langfuse_handler


def _load_jobs_data() -> dict:
    with open("data/jobs.json") as f:
        return json.load(f)

NUM_MACHINES = 5

st.set_page_config(
    page_title="JSSP Scheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Global background ── */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], section.main, .main .block-container {
        background-color: #000 !important;
    }
    [data-testid="stChatMessageContent"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
    /* ── Chat input pinned to bottom ── */
    [data-testid="stBottom"], [data-testid="stChatInput"] {
        background-color: #000 !important;
    }

    /* ── Machine cards ── */
    .m-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid;
        font-size: 0.9rem;
    }
    .m-online  { background: #0f2a1e; border-color: #22c55e; color: #86efac; }
    .m-offline { background: #2a0f0f; border-color: #ef4444; color: #fca5a5; }
    .m-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-on  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-off { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* ── Job rows ── */
    .job-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .job-row strong { color: #f1f5f9; }
    .badge {
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 4px;
    }
    .badge-priority { background: #d97706; color: #fff; }
    .badge-deadline { background: #2563eb; color: #fff; }
    .ops-count { color: #64748b; font-size: 0.8rem; }

    /* ── Main header ── */
    .page-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
    }
    .page-header h1 { color: #f1f5f9; margin: 0 0 4px; font-size: 1.6rem; }
    .page-header p  { color: #64748b; margin: 0; font-size: 0.92rem; }

    /* ── Welcome card ── */
    .welcome-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
        font-size: 0.93rem;
        line-height: 1.6;
        color: #cbd5e1;
    }

    /* ── Status panel ── */
    .status-panel {
        background-color: #0f172a;
        border-radius: 10px;
        padding: 16px;
        color: #cbd5e1;
        position: sticky;
        top: 1rem;
    }
    .sp-title { color: #f1f5f9; font-size: 1.05rem; font-weight: 700; margin: 0 0 4px; }
    .sp-hr { border: none; border-top: 1px solid #1e293b; margin: 10px 0; }
    .sp-section { color: #f1f5f9; font-size: 0.82rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.05em; margin: 10px 0 6px; }
    .sp-metrics { display: flex; gap: 10px; margin-bottom: 8px; }
    .sp-metric  { flex: 1; background: #1e293b; border-radius: 6px; padding: 8px 10px; }
    .sp-metric .lbl { display: block; color: #94a3b8; font-size: 0.72rem; }
    .sp-metric .val { display: block; color: #f1f5f9; font-size: 1.25rem; font-weight: 700; }
    .sp-metric-single { background: #1e293b; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
    .sp-metric-single .lbl { color: #94a3b8; font-size: 0.72rem; }
    .sp-metric-single .val { color: #f1f5f9; font-size: 1.25rem; font-weight: 700; margin-left: 6px; }

    /* ── Jobs detail panel ── */
    .jobs-panel {
        background-color: #0f172a;
        border-radius: 10px;
        padding: 16px;
        color: #cbd5e1;
        margin-top: 12px;
    }
    .jd-job {
        background: #1e293b;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .jd-job-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .jd-name { color: #f1f5f9; font-size: 0.9rem; font-weight: 700; }
    .jd-ops-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }
    .jd-op {
        background: #0f172a;
        border-radius: 4px;
        padding: 3px 7px;
        font-size: 0.72rem;
        color: #94a3b8;
        white-space: nowrap;
    }
    .jd-op span { color: #e2e8f0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "langfuse_session_id" not in st.session_state:
    st.session_state.langfuse_session_id = str(uuid.uuid4())

# ── Page header (full width) ────────────────────────────────────────────────

st.markdown(
    '<div class="page-header">'
    '<h1>Job Shop Scheduling Assistant</h1>'
    '<p>AI-powered scheduler — manage jobs, machines, priorities, and deadlines through conversation</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Two-column layout ────────────────────────────────────────────────────────

chat_col, status_col = st.columns([3, 1])

with status_col:
    status_placeholder = st.empty()
    jobs_placeholder = st.empty()


def _build_status_html() -> str:
    data = _load_jobs_data()
    offline_ids = {m["id"] for m in data["machines"] if m.get("status") != "online"}
    online_count = NUM_MACHINES - len(offline_ids)
    jobs = data["jobs"]
    total_ops = sum(len(j["operations"]) for j in jobs)
    deadline_count = sum(1 for j in jobs if "deadline" in j)

    machine_cards = ""
    for i in range(NUM_MACHINES):
        is_off = i in offline_ids
        cls = "m-offline" if is_off else "m-online"
        dot = "dot-off" if is_off else "dot-on"
        label = "Offline" if is_off else "Online"
        machine_cards += (
            f'<div class="m-card {cls}">'
            f'<span class="m-dot {dot}"></span>'
            f'<strong>Machine {i}</strong>&nbsp;—&nbsp;{label}'
            f'</div>'
        )

    job_rows = ""
    for job in jobs:
        badges = ""
        if "priority" in job:
            badges += f'<span class="badge badge-priority">P{job["priority"]}</span>'
        if "deadline" in job:
            badges += f'<span class="badge badge-deadline">D:{job["deadline"]}</span>'
        ops = len(job["operations"])
        job_rows += (
            f'<div class="job-row">'
            f'<span><strong>{job["name"]}</strong>{badges}</span>'
            f'<span class="ops-count">{ops} ops</span>'
            f'</div>'
        )

    deadline_row = (
        f'<div class="sp-metric-single">'
        f'<span class="lbl">With Deadline</span>'
        f'<span class="val">{deadline_count}</span>'
        f'</div>'
    ) if deadline_count else ""

    return (
        '<div class="status-panel">'
        '<div class="sp-title">🏭 Status</div>'
        '<hr class="sp-hr">'
        '<div class="sp-section">Machines</div>'
        '<div class="sp-metrics">'
        f'<div class="sp-metric"><span class="lbl">Online</span><span class="val">{online_count}</span></div>'
        f'<div class="sp-metric"><span class="lbl">Offline</span><span class="val">{len(offline_ids)}</span></div>'
        '</div>'
        f'{machine_cards}'
        '<hr class="sp-hr">'
        '<div class="sp-section">Jobs</div>'
        '<div class="sp-metrics">'
        f'<div class="sp-metric"><span class="lbl">Total</span><span class="val">{len(jobs)}</span></div>'
        f'<div class="sp-metric"><span class="lbl">Operations</span><span class="val">{total_ops}</span></div>'
        '</div>'
        f'{deadline_row}'
        f'{job_rows}'
        '</div>'
    )


def _build_jobs_html() -> str:
    jobs = _load_jobs_data()["jobs"]
    cards = ""
    for job in jobs:
        badges = ""
        if "priority" in job:
            badges += f'<span class="badge badge-priority">P{job["priority"]}</span>'
        if "deadline" in job:
            badges += f'<span class="badge badge-deadline">D:{job["deadline"]}</span>'
        ops_html = "".join(
            f'<div class="jd-op">M{op["machine_id"]} <span>{op["processing_time"]}t</span></div>'
            for op in job["operations"]
        )
        cards += (
            f'<div class="jd-job">'
            f'<div class="jd-job-header">'
            f'<span class="jd-name">{job["name"]}</span>'
            f'<span>{badges}</span>'
            f'</div>'
            f'<div class="jd-ops-grid">{ops_html}</div>'
            f'</div>'
        )
    return (
        '<div class="jobs-panel">'
        '<div class="sp-title">📋 Jobs Data</div>'
        '<hr class="sp-hr">'
        f'{cards}'
        '</div>'
    )


def render_status() -> None:
    status_placeholder.markdown(_build_status_html(), unsafe_allow_html=True)
    jobs_placeholder.markdown(_build_jobs_html(), unsafe_allow_html=True)


render_status()

# ── Chat area ────────────────────────────────────────────────────────────────

with chat_col:
    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-card">'
            'Hello! I can help you optimise your job shop schedule. Here are some things to try:<br><br>'
            '&nbsp;&nbsp;• <strong>Run the scheduler</strong> — generate an optimised schedule for all jobs<br>'
            '&nbsp;&nbsp;• <strong>Machine management</strong> — e.g. <em>"mark machine 2 as offline"</em><br>'
            '&nbsp;&nbsp;• <strong>Job priorities</strong> — e.g. <em>"set priority of JobA to 5"</em><br>'
            '&nbsp;&nbsp;• <strong>Deadlines</strong> — e.g. <em>"set a deadline of 15 on Job3"</em><br>'
            '&nbsp;&nbsp;• <strong>What-if analysis</strong> — e.g. <em>"what happens if machine 1 goes down?"</em>'
            '</div>',
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt := st.chat_input("Ask me to run the scheduler, update machines, set priorities…"):
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                config = {
                    "configurable": {"thread_id": "thread"},
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": st.session_state.langfuse_session_id,
                        "langfuse_tags": ["jssp", "ui"],
                    },
                }
                input_messages = (
                    [_SYSTEM, user_message] if len(st.session_state.messages) == 1 else [user_message]
                )
                for chunk, _ in app.stream(
                    {"messages": input_messages}, config=config, stream_mode="messages"
                ):
                    if isinstance(chunk, AIMessage) and chunk.content:
                        full_response += chunk.content
                        placeholder.markdown(full_response)
                    elif isinstance(chunk, ToolMessage):
                        if chunk.name == "run_scheduler":
                            full_response += f"\n\n{chunk.content}\n\n"
                        else:
                            tool_label = chunk.name.replace("_", " ").title()
                            full_response += f"\n\n> **{tool_label}**: {chunk.content}"
                        placeholder.markdown(full_response)

                placeholder.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))
            except Exception as e:
                st.error(f"Error: {e}")

        render_status()
