"""DeepRL-RecSys Dashboard – Main Streamlit application.

Bug fixes applied (2026-03-24):
- ope_view: treat severity 'ok' as green (was checking 'pass', which never matched)
- training_view: delegates to load_train_log which now handles .json and .jsonl
- playground_view: default context now uses correct schema (user_item_affinity / user_id)
- playground_view: shows stub warning for DQN / PPO agents
- scan_artifacts import now includes AGENT_STUB_NAMES
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Important: st.set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="DeepRL-RecSys Dashboard",
    page_icon="🤖",
    layout="wide",
)

from deeprl_recsys.ui.utils import (
    AGENT_STUB_NAMES,
    check_reports_extra,
    compute_ess,
    load_ope_report,
    load_train_log,
    scan_artifacts,
)

BASE_ARTIFACTS_DIR = Path("artifacts/models")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _no_artifacts_error() -> None:
    st.error(
        "No se encontró ningún modelo entrenado. "
        "Ejecuta `python scripts/run_full_benchmark.py --agent sac` "
        "para generar los artefactos, o usa la muestra de OBD en `data/sample/`."
    )


def _get_artifacts() -> pd.DataFrame:
    return scan_artifacts(BASE_ARTIFACTS_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────────────

def home_view() -> None:
    st.title("🏠 Resumen de Experimentos")
    st.write("Explora los modelos y artefactos generados por los benchmarks.")

    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🔄 Recargar", use_container_width=True):
            scan_artifacts.clear()
            st.rerun()

    with st.spinner("Escaneando artefactos..."):
        df_artifacts = _get_artifacts()

    if df_artifacts.empty:
        _no_artifacts_error()
        return

    st.success(f"✅ Se encontraron **{len(df_artifacts)}** artefactos en `artifacts/models/`.")
    st.dataframe(
        df_artifacts[["artifact_id", "agent_name", "schema_version", "created_at", "description"]],
        use_container_width=True,
        hide_index=True,
    )


def ope_view() -> None:
    st.title("📊 Análisis OPE")
    st.write("Evaluación Off-Policy de los agentes entrenados.")

    df_artifacts = _get_artifacts()
    if df_artifacts.empty:
        _no_artifacts_error()
        return

    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_id = st.selectbox("Seleccionar Artefacto", artifact_opts)
    selected_row = df_artifacts[df_artifacts["artifact_id"] == selected_id].iloc[0]
    selected_path = selected_row["path"]
    agent_name = selected_row.get("agent_name", "unknown").lower()

    # Stub notice
    if agent_name in AGENT_STUB_NAMES:
        st.info(
            f"ℹ️ **{agent_name.upper()} es un agente stub funcional.** "
            "Sus métricas OPE son idénticas a las de otros stubs porque el agente no ha completado "
            "el entrenamiento real. Esto es comportamiento esperado, no un error."
        )

    report_data = load_ope_report(selected_path)
    if not report_data:
        st.warning(f"No se encontró `ope_report.json` en `{selected_path}`.")
        return

    # ── Verdict Traffic Light ──────────────────────────────────────────────
    verdict = report_data.get("verdict", {})
    sev = verdict.get("severity", "unknown").lower()
    reliable = verdict.get("reliable", False)

    if sev == "ok" or reliable:
        st.success("🟢 Fiabilidad: OK — Los estimadores son confiables.")
    elif sev == "warning":
        st.warning("🟡 Fiabilidad: WARNING — Revisa las advertencias a continuación.")
    else:
        st.error(f"🔴 Fiabilidad: {sev.upper()}")

    for w in verdict.get("warnings", []):
        st.write(f"  ⚠️ {w}")

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = verdict.get("stats", {})
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N Samples", f"{int(stats.get('n_samples', 0)):,}")
        c2.metric("ESS", f"{stats.get('ess', 0):.1f}")
        c3.metric("Clipping Rate", f"{stats.get('clipping_rate', 0):.1%}")
        c4.metric("Max Weight", f"{stats.get('max_weight', 0):.2f}")

    # ── Estimadores ───────────────────────────────────────────────────────
    st.markdown("### Estimadores IPS / DR / MIPS")
    estimates = report_data.get("estimates", {})
    if estimates:
        df_est = pd.DataFrame(
            list(estimates.items()), columns=["Estimador", "Valor"]
        )
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.dataframe(df_est, hide_index=True, use_container_width=True)
        with col_r:
            fig = px.bar(
                df_est,
                x="Estimador",
                y="Valor",
                color="Estimador",
                title="Comparativa de Estimadores",
                text_auto=".5f",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El reporte OPE no contiene estimadores.")

    # ── ESS from raw weights (computed) ───────────────────────────────────
    weights = report_data.get("importance_weights", [])
    if weights:
        ess_val = compute_ess(weights)
        st.metric(label="ESS (calculado desde importance_weights)", value=f"{ess_val:.2f}")

    # ── PDF Export ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Exportar a PDF"):
        if check_reports_extra():
            try:
                from deeprl_recsys.evaluation.reports import run_generate_report_pdf

                pdf_path = run_generate_report_pdf(selected_path)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Descargar PDF", f, file_name=f"{selected_id}_report.pdf"
                    )
            except Exception as e:
                st.error(f"Falló la generación del PDF: {e}")
        else:
            st.warning("Faltan dependencias para exportar a PDF.")
            st.info("Ejecuta `pip install -e .[reports]` para habilitar esta función.")


def training_view() -> None:
    st.title("📈 Curvas de Entrenamiento")
    st.write("Monitoreo de métricas durante el entrenamiento (reward por step).")

    df_artifacts = _get_artifacts()
    if df_artifacts.empty:
        _no_artifacts_error()
        return

    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_id = st.selectbox("Seleccionar Artefacto", artifact_opts, key="train_sel")
    selected_row = df_artifacts[df_artifacts["artifact_id"] == selected_id].iloc[0]
    selected_path = selected_row["path"]
    agent_name = selected_row.get("agent_name", "unknown").lower()

    col1, col2 = st.columns([8, 2])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)")
        if st.button("Actualizar ahora"):
            load_train_log.clear()
            st.rerun()

    if auto_refresh:
        st_autorefresh(interval=5000, key="train_autorefresh")
        load_train_log.clear()

    with st.spinner("Cargando log de entrenamiento..."):
        df_log = load_train_log(selected_path)

    if df_log.empty:
        st.info(
            "No se encontró `train_log.json` / `train_log.jsonl` para este artefacto, "
            "o el archivo está vacío. Esto puede ocurrir en artefactos legacy o en el "
            "artefacto `latest` que apunta a otra carpeta."
        )
        return

    # Smooth reward with rolling average for readability
    if "step" in df_log.columns and "reward" in df_log.columns:
        df_plot = df_log[["step", "reward"]].copy()
        window = max(1, len(df_plot) // 50)  # ~2% sliding window
        df_plot["reward_smooth"] = df_plot["reward"].rolling(window, min_periods=1).mean()

        st.markdown(f"### Reward over Steps — **{selected_id}** ({agent_name.upper()})")
        if agent_name in AGENT_STUB_NAMES:
            st.caption(
                "ℹ️ Este agente es un *stub* — la curva de reward corresponde a la recompensa "
                "promedio del entorno, no a una política aprendida."
            )

        fig_reward = px.line(
            df_plot,
            x="step",
            y=["reward", "reward_smooth"],
            labels={"value": "Reward", "step": "Step", "variable": "Serie"},
            title=f"Reward vs Step — {selected_id}",
            color_discrete_map={"reward": "lightblue", "reward_smooth": "royalblue"},
        )
        fig_reward.update_traces(opacity=0.4, selector=dict(name="reward"))
        st.plotly_chart(fig_reward, use_container_width=True)

        last_reward = df_plot["reward_smooth"].iloc[-1]
        mean_reward = df_plot["reward_smooth"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Último Reward (smooth)", f"{last_reward:.4f}")
        c2.metric("Reward Promedio", f"{mean_reward:.4f}")
        c3.metric("Pasos totales", f"{len(df_plot):,}")

    elif "step" in df_log.columns and "loss" in df_log.columns:
        st.markdown("### Loss Curve")
        fig_loss = px.line(df_log, x="step", y="loss", title="Loss over Steps")
        st.plotly_chart(fig_loss, use_container_width=True)
        st.metric("Último Loss", f"{df_log['loss'].iloc[-1]:.4f}")

    else:
        st.write("Datos de log disponibles (columnas no estándar):")
        st.dataframe(df_log.tail(20), use_container_width=True)


def playground_view() -> None:
    st.title("🎮 Recommendation Playground")
    st.write("Prueba predicciones interactivas y compara agentes lado a lado.")

    df_artifacts = _get_artifacts()
    if df_artifacts.empty:
        _no_artifacts_error()
        return

    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_ids = st.multiselect(
        "Seleccionar Agentes para Comparar",
        artifact_opts,
        default=[artifact_opts[0]] if artifact_opts else [],
        key="play_sel",
    )

    if not selected_ids:
        st.warning("Selecciona al menos un artefacto para continuar.")
        return

    # Stub info banner
    selected_agents = [
        df_artifacts[df_artifacts["artifact_id"] == aid].iloc[0].get("agent_name", "").lower()
        for aid in selected_ids
    ]
    stub_selected = [a.upper() for a in selected_agents if a in AGENT_STUB_NAMES]
    if stub_selected:
        st.info(
            f"ℹ️ **{', '.join(stub_selected)}** {'son agentes stub funcionales' if len(stub_selected) > 1 else 'es un agente stub funcional'}. "
            "Sus recomendaciones son deterministas y no dependen del contexto del usuario — "
            "esto es comportamiento esperado (política cercana a aleatoria, no entrenada). "
            "Compara con **SAC** para ver diferenciación real por contexto."
        )

    # ── Context configuration ──────────────────────────────────────────────
    st.markdown("### 🛠️ Configuración de Inferencia")

    first_row = df_artifacts[df_artifacts["artifact_id"] == selected_ids[0]].iloc[0]
    schema_ver = first_row.get("schema_version", "bandit_v1")

    # Default context based on schema — always include user_item_affinity for bandit_v1
    if schema_ver == "sequential_v1":
        default_ctx: dict | list = [{"item": 101, "reward": 1}, {"item": 205, "reward": 0}]
    else:
        # bandit_v1 (and unknown) — ServingRuntime expects user_item_affinity + user_id
        default_ctx = {"user_item_affinity": 0.75, "user_id": 42}

    context_str = st.text_area(
        "Contexto de Usuario (JSON)",
        value=json.dumps(default_ctx, indent=2),
        height=120,
        help=(
            "Para agentes bandit_v1, incluye `user_item_affinity` (float) y `user_id` (int). "
            "Valores de afinidad más altos tienden a producir scores más altos en SAC."
        ),
    )

    try:
        context_data = json.loads(context_str)
        is_valid_json = True
    except json.JSONDecodeError as e:
        st.error(f"JSON Inválido: {e}")
        is_valid_json = False

    candidates = st.multiselect(
        "Items Candidatos (IDs)",
        options=list(range(80)),
        default=[10, 25, 33, 41, 55],
        max_selections=20,
        help="Selecciona hasta 20 items. Para agentes entrenados con dataset sintético usa IDs < 50.",
    )

    if not candidates:
        st.warning("Selecciona al menos un item candidato.")
        return

    if st.button("🚀 Generar Recomendaciones Lado a Lado", disabled=not is_valid_json):
        from deeprl_recsys.ui.utils import load_serving_runtime

        cols = st.columns(len(selected_ids))

        for i, art_id in enumerate(selected_ids):
            with cols[i]:
                row = df_artifacts[df_artifacts["artifact_id"] == art_id].iloc[0]
                art_path = row["path"]
                agent = row.get("agent_name", "?").upper()

                st.subheader(f"🤖 {art_id}")
                st.caption(f"Agent type: **{agent}**")

                if row.get("agent_name", "").lower() in AGENT_STUB_NAMES:
                    st.caption("⚠️ Stub: probabilidades deterministas, independientes del contexto.")

                with st.spinner(f"Inferencia {art_id}..."):
                    try:
                        runtime = load_serving_runtime(art_path)
                        preds = runtime.predict(context_data, candidates, k=len(candidates))

                        df_res = pd.DataFrame(
                            {
                                "Item ID": [p["item_id"] for p in preds],
                                "Score": [round(p["score"], 4) for p in preds],
                            }
                        )

                        st.dataframe(
                            df_res,
                            column_config={
                                "Item ID": st.column_config.NumberColumn("Item ID", format="%d"),
                                "Score": st.column_config.ProgressColumn(
                                    "Score",
                                    format="%.4f",
                                    min_value=0.0,
                                    max_value=1.0,
                                ),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )

                        total = sum(p["score"] for p in preds)
                        st.caption(
                            f"Candidates: {len(candidates)} | "
                            f"Sum of scores: {total:.4f} | "
                            f"Schema: {row.get('schema_version', 'N/A')}"
                        )

                    except Exception as e:
                        st.error(f"Error loading {art_id}: {e}")
                        st.exception(e)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.sidebar.title("🤖 DeepRL-RecSys")
    st.sidebar.markdown(
        "**Dashboard analítico** para visualizar benchmarks de agentes RL "
        "con evaluación Off-Policy (OPE)."
    )
    st.sidebar.markdown("---")

    view_selection = st.sidebar.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Análisis OPE", "📈 Entrenamiento", "🎮 Playground"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Agentes disponibles:** SAC ✅ | DQN ⚠️ stub | PPO ⚠️ stub\n\n"
        "[Ver reportes](reports/) · [GitHub](https://github.com/aalopez76/DeepRL-RecSys-Platform)"
    )

    if view_selection == "🏠 Inicio":
        home_view()
    elif view_selection == "📊 Análisis OPE":
        ope_view()
    elif view_selection == "📈 Entrenamiento":
        training_view()
    elif view_selection == "🎮 Playground":
        playground_view()


if __name__ == "__main__":
    main()
