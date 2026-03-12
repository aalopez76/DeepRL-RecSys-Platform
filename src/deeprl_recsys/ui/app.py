from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# Important: st.set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="DeepRL-RecSys Dashboard",
    page_icon="🤖",
    layout="wide"
)

from deeprl_recsys.ui.utils import (
    scan_artifacts,
    load_ope_report,
    load_train_log,
    check_reports_extra,
)

BASE_ARTIFACTS_DIR = Path("artifacts/models")

def home_view():
    st.title("🏠 Resumen de Experimentos")
    st.write("Explora los modelos y artefactos generados.")
    
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🔄 Recargar", use_container_width=True):
            scan_artifacts.clear()
            st.rerun()
            
    with st.spinner("Escaneando artefactos..."):
        df_artifacts = scan_artifacts(BASE_ARTIFACTS_DIR)
        
    if df_artifacts.empty:
        st.info("No se encontraron artefactos. Ejecuta los pipelines de entrenamiento primero.")
        return
        
    st.dataframe(
        df_artifacts,
        use_container_width=True,
        hide_index=True,
    )

def ope_view():
    st.title("📊 Análisis OPE")
    st.write("Evaluación Off-Policy de los agentes.")
    
    df_artifacts = scan_artifacts(BASE_ARTIFACTS_DIR)
    if df_artifacts.empty:
        st.warning("No hay artefactos disponibles para analizar.")
        return
        
    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_id = st.selectbox("Seleccionar Artefacto", artifact_opts)
    selected_path = df_artifacts[df_artifacts["artifact_id"] == selected_id].iloc[0]["path"]
    
    report_data = load_ope_report(selected_path)
    if not report_data:
        st.warning("No se encontró `ope_report.json` para este artefacto.")
        return
        
    # Verdict Traffic Light
    verdict = report_data.get("verdict", {})
    sev = verdict.get("severity", "unknown").lower()
    if sev == "pass":
        st.success("🟢 Fiabilidad: PASS")
    elif sev == "warning":
        st.warning("🟡 Fiabilidad: WARNING")
    else:
        st.error(f"🔴 Fiabilidad: {sev.upper()}")
        
    for w in verdict.get("warnings", []):
        st.write(f"- {w}")
        
    st.markdown("### Estimadores")
    estimates = report_data.get("estimates", {})
    if estimates:
        df_est = pd.DataFrame(list(estimates.items()), columns=["Estimator", "Value"])
        st.dataframe(df_est, hide_index=True)
        
        fig = px.bar(df_est, x="Estimator", y="Value", title="Comparativa de Estimadores")
        st.plotly_chart(fig, use_container_width=True)
        
    # PDF Export Toggle
    st.markdown("---")
    if st.button("Exportar a PDF"):
        if check_reports_extra():
            # Mock PDF generation call for now, since generate_report requires deeper core logic
            # In a real scenario we would call core.evaluation.reports.generate_report
            try:
                from deeprl_recsys.evaluation.reports import run_generate_report_pdf
                pdf_path = run_generate_report_pdf(selected_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("Descargar PDF", f, file_name=f"{selected_id}_report.pdf")
            except Exception as e:
                st.error(f"Falló la generación del PDF: {e}")
                st.warning("Sugerencia: Revisa los logs o intenta ejecutar nuevamente.")
        else:
            st.warning("Faltan dependencias para exportar a PDF.")
            st.info("Sugerencia: Ejecuta `pip install -e .[reports]` para habilitar esta función.")
    
def training_view():
    st.title("📈 Curvas de Entrenamiento")
    st.write("Monitoreo de métricas durante el entrenamiento.")

    df_artifacts = scan_artifacts(BASE_ARTIFACTS_DIR)
    if df_artifacts.empty:
        st.warning("No hay artefactos disponibles.")
        return
        
    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_id = st.selectbox("Seleccionar Artefacto", artifact_opts, key="train_sel")
    selected_path = df_artifacts[df_artifacts["artifact_id"] == selected_id].iloc[0]["path"]

    col1, col2 = st.columns([8, 2])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)")
        if st.button("Actualizar ahora"):
            load_train_log.clear()
            st.rerun()

    if auto_refresh:
        st_autorefresh(interval=5000, key="train_autorefresh")
        # Clears the specific cache for this view
        load_train_log.clear()

    df_log = load_train_log(selected_path)
    if df_log.empty:
        st.warning("No se encontró `train_log.jsonl` o está vacío.")
        return

    # Assuming 'step', 'loss', and 'reward' are columns in the log format
    if "step" in df_log.columns:
        if "loss" in df_log.columns:
            st.markdown("### Loss Curve")
            fig_loss = px.line(df_log, x="step", y="loss", title="Loss over Steps")
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Display last metric
            last_loss = df_log.iloc[-1]["loss"]
            st.metric(label="Último Loss", value=f"{last_loss:.4f}")
            
        if "reward" in df_log.columns:
            st.markdown("### Reward Curve")
            fig_reward = px.line(df_log, x="step", y="reward", title="Reward over Steps")
            st.plotly_chart(fig_reward, use_container_width=True)
            
            last_reward = df_log.iloc[-1]["reward"]
            st.metric(label="Último Reward", value=f"{last_reward:.4f}")
    else:
        st.write("Datos de log no estándar:", df_log.tail())


def playground_view():
    st.title("🎮 Recommendation Playground")
    st.write("Prueba predicciones interactivas.")

    df_artifacts = scan_artifacts(BASE_ARTIFACTS_DIR)
    if df_artifacts.empty:
        st.warning("No hay artefactos disponibles.")
        return
        
    artifact_opts = df_artifacts["artifact_id"].tolist()
    selected_id = st.selectbox("Seleccionar Artefacto", artifact_opts, key="play_sel")
    selected_row = df_artifacts[df_artifacts["artifact_id"] == selected_id].iloc[0]
    schema_ver = selected_row["schema_version"]

    st.markdown(f"**Schema Version:** `{schema_ver}`")

    # Generate default JSON based on schema
    if schema_ver == "bandit_v1":
        default_ctx = {"user_features": {"age": 30, "gender": "F"}, "history": [101, 205]}
    elif schema_ver == "sequential_v1":
        default_ctx = [{"item": 101, "reward": 1}, {"item": 205, "reward": 0}]
    else:
        default_ctx = {"features": {}}

    context_str = st.text_area("Contexto de Usuario (JSON)", value=json.dumps(default_ctx, indent=2), height=150)
    
    # Validar JSON
    try:
        context_data = json.loads(context_str)
        is_valid_json = True
    except json.JSONDecodeError as e:
        st.error(f"JSON Inválido: {e}")
        is_valid_json = False

    if st.button("Recomendar", disabled=not is_valid_json):
        with st.spinner("Generando recomendación..."):
            # Mock de recomendación por ahora, en la vida real se usa ServingRuntime
            import time
            time.sleep(0.5)
            
            # Simulated dummy response
            dummy_results = pd.DataFrame({
                "Item ID": [10, 25, 33, 41, 55],
                "Score": [0.95, 0.88, 0.74, 0.65, 0.50]
            })
            
            st.success("Recomendación lista")
            st.table(dummy_results)

def main():
    st.sidebar.title("DeepRL-RecSys")
    st.sidebar.markdown("---")
    
    view_selection = st.sidebar.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Análisis OPE", "📈 Entrenamiento", "🎮 Playground"]
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
