import streamlit as st


def initialize_llm_state():
    if "llm_agent_run_log" not in st.session_state:
        st.session_state["llm_agent_run_log"] = []


def trim_llm_run_log(max_entries):
    st.session_state["llm_agent_run_log"] = st.session_state["llm_agent_run_log"][:max_entries]


def add_llm_run_log_entry(entry, max_entries):
    st.session_state["llm_agent_run_log"].insert(0, entry)
    trim_llm_run_log(max_entries)


def simulation_settings_signature(settings):
    return {
        "population_size": int(settings["population_size"]),
        "no_policy_outcome_rate": round(float(settings["no_policy_outcome_rate"]), 6),
        "symmetric_error_rate": round(float(settings["symmetric_error_rate"]), 6),
        "policy_intensity_tier": str(settings.get("policy_intensity_tier", "")),
        "llm_simulation_runs": int(settings["llm_simulation_runs"]),
        "llm_representative_agents": int(settings["llm_representative_agents"]),
        "llm_agent_models": tuple(settings.get("llm_agent_models", [])),
    }


def latest_result_has_current_schema(latest_result, settings=None):
    if not isinstance(latest_result, dict):
        return False

    if latest_result.get("schema_version") != 5:
        return False

    if not isinstance(latest_result.get("comparison_table"), dict):
        return False

    if not isinstance(latest_result.get("policy_results"), dict):
        return False

    if settings is not None:
        return latest_result.get("settings_signature") == simulation_settings_signature(settings)

    return True


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
