import streamlit as st

from src.constants import RUN_METRIC_COLUMNS


def initialize_llm_state():
    if "llm_agent_run_log" not in st.session_state:
        st.session_state["llm_agent_run_log"] = []


def trim_llm_run_log(max_entries):
    st.session_state["llm_agent_run_log"] = st.session_state["llm_agent_run_log"][:max_entries]


def add_llm_run_log_entry(entry, max_entries):
    st.session_state["llm_agent_run_log"].insert(0, entry)
    trim_llm_run_log(max_entries)


def latest_result_has_current_schema(latest_result):
    run_results = latest_result.get("run_results")
    if run_results is None:
        return False

    required_run_columns = set(RUN_METRIC_COLUMNS + ["policy", "llm_model"])
    return required_run_columns.issubset(run_results.columns)


def attach_model_label(run_results, model):
    run_results = run_results.copy()
    run_results.insert(0, "llm_model", model)
    return run_results


def attach_policy_label(run_results, policy):
    run_results = run_results.copy()
    run_results.insert(0, "policy", policy)
    return run_results


def normalize_representative_agents(raw_agents, model, policy):
    rows = []
    for agent in raw_agents:
        if isinstance(agent, dict):
            row = dict(agent)
        else:
            row = {"description": str(agent)}
        row["policy"] = policy
        row["llm_model"] = model
        rows.append(row)
    return rows
