import json
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from agent import app, _SYSTEM, langfuse_handler

st.set_page_config(
    page_title="JSSP Scheduling Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS ----------
st.markdown("""
<style>
/* ── Global dark background ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], section.main, .main .block-container {
    background-color: #000 !important;
}
[data-testid="stChatMessageContent"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
}
[data-testid="stBottom"], [data-testid="stChatInput"] {
    background-color: #000 !important;
}
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}

/* ── Header ── */
.header-band {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.header-band h1 { color: #f1f5f9; margin: 0 0 4px; font-size: 1.6rem; }
.header-band p  { color: #64748b; margin: 0; font-size: 0.92rem; }

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
.m-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
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
.job-block { margin-bottom: 8px; }
.ops-chips { display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 0 6px; }
.op-chip {
    background: #1e293b;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.72rem;
    color: #94a3b8;
    white-space: nowrap;
}
.op-chip strong { color: #e2e8f0; }

/* ── Sidebar section title ── */
.sidebar-title {
    color: #f1f5f9;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 10px 0 6px;
}
.sp-hr { border: none; border-top: 1px solid #1e293b; margin: 10px 0; }
.sp-metrics { display: flex; gap: 10px; margin-bottom: 8px; }
.sp-metric {
    flex: 1;
    background: #1e293b;
    border-radius: 6px;
    padding: 8px 10px;
}
.sp-metric .lbl { display: block; color: #94a3b8; font-size: 0.72rem; }
.sp-metric .val { display: block; color: #f1f5f9; font-size: 1.25rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    '<div class="header-band">'
    '<h1>Job Shop Scheduling Agent</h1>'
    '<p>AI-powered assistant for managing machines, configuring job priorities &amp; deadlines, and running optimised production schedules.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [_SYSTEM]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- Sidebar ----------
def load_jobs() -> dict:
    with open("data/jobs.json") as f:
        return json.load(f)


def _render_sidebar() -> None:
    data = load_jobs()
    machines = data["machines"]
    jobs = data["jobs"]

    online_count = sum(1 for m in machines if m.get("status") == "online")
    offline_count = len(machines) - online_count

    machine_cards = "".join(
        f'<div class="m-card {"m-online" if m["status"] == "online" else "m-offline"}">'
        f'<span class="m-dot {"dot-on" if m["status"] == "online" else "dot-off"}"></span>'
        f'<strong>Machine {m["id"]}</strong>&nbsp;—&nbsp;{"Online" if m["status"] == "online" else "Offline"}'
        f'</div>'
        for m in machines
    )

    job_rows = ""
    for job in jobs:
        badges = ""
        if "priority" in job:
            badges += f'<span class="badge badge-priority">P{job["priority"]}</span>'
        if "deadline" in job:
            badges += f'<span class="badge badge-deadline">D:{job["deadline"]}</span>'
        ops_chips = "".join(
            f'<span class="op-chip">M{op["machine_id"]} <strong>{op["processing_time"]}t</strong></span>'
            for op in job["operations"]
        )
        job_rows += (
            f'<div class="job-block">'
            f'<div class="job-row"><span><strong>{job["name"]}</strong>{badges}</span>'
            f'<span class="ops-count">{len(job["operations"])} ops</span></div>'
            f'<div class="ops-chips">{ops_chips}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="sidebar-title">Machines</div>'
        f'<div class="sp-metrics">'
        f'<div class="sp-metric"><span class="lbl">Online</span><span class="val">{online_count}</span></div>'
        f'<div class="sp-metric"><span class="lbl">Offline</span><span class="val">{offline_count}</span></div>'
        f'</div>'
        f'{machine_cards}'
        f'<hr class="sp-hr">'
        f'<div class="sidebar-title">Jobs</div>'
        f'{job_rows}',
        unsafe_allow_html=True,
    )


with st.sidebar:
    _render_sidebar()


# ---------- Chat ----------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask the scheduling agent…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append(HumanMessage(content=prompt))

    config = {
        "recursion_limit": 10,
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": st.session_state.session_id,
            "langfuse_tags": ["jssp", "streamlit"],
        },
    }

    state = {"messages": st.session_state.agent_messages}
    with st.chat_message("assistant"):
        with st.spinner("Working…"):
            for step in app.stream(state, config=config, stream_mode="values"):
                state = step
        reply = state["messages"][-1].content
        st.markdown(reply)

    st.session_state.agent_messages = state["messages"]
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()
