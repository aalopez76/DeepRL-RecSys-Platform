import nbformat as nbf
import os
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # 1. Title and setup
    cells.append(nbf.v4.new_markdown_cell("# Análisis Avanzado Off-Policy Evaluation (OPE)\n\n"
        "Este notebook proporciona una experiencia interactiva para visualizar los resultados del "
        "entrenamiento del agente SAC y evaluar su rendimiento utilizando técnicas avanzadas de OPE: "
        "Inverse Propensity Scoring (**IPS**), Doubly Robust (**DR**) y Marginal IPS (**MIPS**).\n\n"
        "**Objetivos**:\n"
        "1. Analizar la reducción de varianza lograda por MIPS en comparación con IPS/DR.\n"
        "2. Simular interactivamente la sensibilidad al contexto del agente SAC."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Instalación de dependencias (necesario si se ejecuta en Colab u otro entorno aislado)\n"
        "# !pip install -q ipywidgets plotly pandas matplotlib nbconvert"
    ))

    # 2. Imports and Data Loading
    cells.append(nbf.v4.new_code_cell(
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from pathlib import Path\n"
        "import plotly.graph_objects as go\n"
        "import ipywidgets as widgets\n"
        "from IPython.display import display, HTML, clear_output\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Ensure we can import deeprl_recsys models (if running locally)\n"
        "import sys\n"
        "sys.path.append('../src')\n"
        "try:\n"
        "    from deeprl_recsys.agents.sac import SACAgent\n"
        "except ImportError:\n"
        "    warnings.warn('SACAgent not available directly. Simulated interactions might fail.')"
    ))

    cells.append(nbf.v4.new_code_cell(
        'SCENARIOS = [\n'
        '    {"id": "benchmark_synthetic", "name": "Control Sintético"},\n'
        '    {"id": "benchmark_random", "name": "OBD Random (Bajo Sesgo)"},\n'
        '    {"id": "benchmark_bts", "name": "OBD BTS (Alto Sesgo)"}\n'
        ']\n\n'
        'ARTIFACTS_DIR = Path("../artifacts/models")\n\n'
        'results = []\n'
        'missing_data = False\n'
        'for s in SCENARIOS:\n'
        '    report_path = ARTIFACTS_DIR / s["id"] / "ope_report.json"\n'
        '    if report_path.exists():\n'
        '        with open(report_path, "r") as f:\n'
        '            data = json.load(f)\n'
        '        estimates = data.get("estimates", {})\n'
        '        diagnostics = data.get("verdict", {}).get("diagnostics", {})\n'
        '        results.append({\n'
        '            "Escenario": s["name"],\n'
        '            "IPS": estimates.get("ips", 0.0),\n'
        '            "DR": estimates.get("dr", 0.0),\n'
        '            "MIPS": estimates.get("mips", 0.0),\n'
        '            "ESS (Effective Sample Size)": diagnostics.get("ess", 0.0)\n'
        '        })\n'
        '    else:\n'
        '        missing_data = True\n'
        '        print(f"⚠️ Reporte {s[\'id\']} no encontrado en {report_path}")\n\n'
        'if missing_data:\n'
        '    display(HTML("<h3 style=\'color:red\'>⚠️ Faltan datos de los reportes. Asegúrate de ejecutar <code>python scripts/run_full_benchmark.py</code> antes de correr el notebook.</h3>"))\n\n'
        'df_results = pd.DataFrame(results)\n'
        'display(df_results.style.format({"IPS": "{:.4f}", "DR": "{:.4f}", "MIPS": "{:.4f}", "ESS (Effective Sample Size)": "{:,.1f}"}).background_gradient(cmap="viridis"))\n'
    ))

    # 3. Theory Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 🧠 Teoría Off-Policy Evaluation\n\n"
        "- **IPS (Inverse Propensity Scoring)**: Pondera las recompensas observadas usando el ratio $\\frac{\\pi(a|s)}{\\pi_0(a|s)}$. Puede sufrir de alta varianza si el denominador es pequeño.\n"
        "- **DR (Doubly Robust)**: Combina IPS con un modelo de predicción de recompensas para reducir la varianza matemática mientras se mantiene insesgado.\n"
        "- **MIPS (Marginal IPS)**: Relaja el supuesto de condicionalidad del estado $\\frac{\\pi(a|s)}{P_0(a)}$ para estabilizar variaciones extremas generadas por interacciones deterministas o contextos de muy alta dimensionalidad. Frecuentemente produce un ESS superior.\n\n"
        "**💡 Degradación de ESS en BTS**: En el escenario BTS, notará un ESS mucho más bajo que en las otras simulaciones. Esto ocurre porque la política *logging* favorece ciertas acciones fuertemente condicionadas (sesgo de popularidad/presentación), lo que infla los weights de IPS en acciones que el agente SAC prioriza pero el sistema original ignoró."
    ))

    # 4. Convergence Curve
    cells.append(nbf.v4.new_markdown_cell("## 📈 Gráfico de Convergencia del SAC\nVisualizamos el rendimiento del modelo en entrenamiento a lo largo de pasos secuenciales."))

    cells.append(nbf.v4.new_code_cell(
        '# Visualizar Curva de Recompensa (Random) \n'
        'train_log_path = ARTIFACTS_DIR / "benchmark_random" / "train_log.json"\n'
        'if train_log_path.exists():\n'
        '    with open(train_log_path, "r") as f:\n'
        '        train_data = json.load(f)\n'
        '    metrics = train_data.get("metrics", [])\n'
        '    df_train = pd.DataFrame(metrics)\n'
        '    \n'
        '    fig = go.Figure()\n'
        '    fig.add_trace(go.Scatter(x=df_train["step"], y=df_train["reward"], mode="lines", name="Training Reward", line=dict(color="blue")))\n'
        '    # Add rolling mean\n'
        '    df_train["rolling"] = df_train["reward"].rolling(window=100, min_periods=1).mean()\n'
        '    fig.add_trace(go.Scatter(x=df_train["step"], y=df_train["rolling"], mode="lines", name="Rolling Mean (100)", line=dict(color="red", width=3)))\n'
        '    \n'
        '    fig.update_layout(title="Convergencia del Agente SAC (Envíronment Random)", xaxis_title="Pasos de Entrenamiento", yaxis_title="Recompensa Promedio", template="plotly_white")\n'
        '    fig.show()\n'
        'else:\n'
        '    print("⚠️ No hay log de entrenamiento.")'
    ))

    # 5. Interactive Simulator Explanation
    cells.append(nbf.v4.new_markdown_cell(
        "## 🕹️ Simulador de Sensibilidad del Contexto\n\n"
        "Una política predictiva 'production-ready' asimila los cambios del contexto. "
        "Si la variable crítica `user_item_affinity` varía, el agente SAC (mediante sus capas de concatenación y proyección) "
        "debería re-ordenar el ranking de acciones recomendadas. Esto se evidencia si la correlación de *Spearman* "
        "con respecto a un escenario base es **estrictamente menor a 1.0**. \n\n"
        "Controla el slider para observar cómo el top-5 cambia dinámicamente según la afinidad del usuario."
    ))
    
    # 6. Interactive Simulator Code
    cells.append(nbf.v4.new_code_cell(
        'from scipy.stats import spearmanr\n\n'
        'output_area = widgets.Output()\n\n'
        'base_ctx = {"user_item_affinity": 0.0, "user_id": 1234, "time_active": 300}\n'
        'num_items = 100\n'
        'candidates = list(range(num_items))\n\n'
        '# Cargar Agente\n'
        'agent = None\n'
        'model_path = ARTIFACTS_DIR / "benchmark_random" / "model.pt"\n'
        'if model_path.exists():\n'
        '    agent = SACAgent(num_items=10000, context_dim=1)\n'
        '    agent.load(str(model_path))\n\n'
        'def evaluate_context(affinity_val):\n'
        '    if agent is None:\n'
        '        with output_area:\n'
        '            clear_output(wait=True)\n'
        '            print("❌ Modelo SAC no cargado. Ejecuta el run_full_benchmark.py primero.")\n'
        '            return\n'
        '            \n'
        '    # Inferencia Original\n'
        '    base_ctx["user_item_affinity"] = 0.0\n'
        '    probs_orig = agent.get_action_probabilities(base_ctx, candidates)\n'
        '    scores_orig = np.array([probs_orig[c] for c in candidates])\n'
        '    rank_orig = np.argsort(-scores_orig)[:5].tolist()\n'
        '    \n'
        '    # Inferencia Actualizada\n'
        '    current_ctx = base_ctx.copy()\n'
        '    current_ctx["user_item_affinity"] = affinity_val\n'
        '    probs_curr = agent.get_action_probabilities(current_ctx, candidates)\n'
        '    scores_curr = np.array([probs_curr[c] for c in candidates])\n'
        '    rank_curr = np.argsort(-scores_curr)[:5].tolist()\n'
        '    \n'
        '    # Spearman sobre todos los items \n'
        '    coef, _ = spearmanr(scores_orig, scores_curr)\n'
        '    \n'
        '    # Render UI\n'
        '    with output_area:\n'
        '        clear_output(wait=True)\n'
        '        display(HTML(f"<h3>Correlación de Spearman vs Baseline (Afinidad 0): <strong style=\'color:{((coef < 1.0) and \\"green\\" or \\"red\\")}\'>{coef:.4f}</strong></h3>"))\n'
        '        display(HTML("<i>Una correlación menor a 1.0 indica sensibilidad, donde 1.0 indica rigidez total.</i><br/>"))\n'
        '        \n'
        '        # Tablas\n'
        '        df_top5 = pd.DataFrame({\n'
        '            "Top-K": [1, 2, 3, 4, 5],\n'
        '            "Baseline Item IDs": rank_orig,\n'
        '            "Baseline Probs": [probs_orig[i] for i in rank_orig],\n'
        '            "Perturbed Item IDs": rank_curr,\n'
        '            "Perturbed Probs": [probs_curr[i] for i in rank_curr],\n'
        '        })\n'
        '        \n'
        '        # Identificar si cambió el orden (highlight)\n'
        '        def highlight_diff(row):\n'
        '            color = "background-color: lightgreen" if row["Baseline Item IDs"] != row["Perturbed Item IDs"] else ""\n'
        '            return [color] * 5\n'
        '            \n'
        '        display(df_top5.style.apply(highlight_diff, axis=1).format({"Baseline Probs": "{:.4%}", "Perturbed Probs": "{:.4%}"}))\n\n'
        '\n'
        'slider = widgets.FloatSlider(\n'
        '    value=0.0,\n'
        '    min=-3.0,\n'
        '    max=3.0,\n'
        '    step=0.1,\n'
        '    description="Afinidad:",\n'
        '    continuous_update=False,\n'
        '    readout_format=".2f"\n'
        ')\n\n'
        'widgets.interactive_output(evaluate_context, {"affinity_val": slider})\n'
        'display(widgets.VBox([slider, output_area]))'
    ))

    # 7. Convert to HTML Instruction
    cells.append(nbf.v4.new_markdown_cell(
        "---\n\n"
        "### Exportación del Reporte\n"
        "Para guardar este cuaderno junto con sus salidas y resultados interactivos como informe HTML estático, "
        "puede ejecutar el siguiente comando:"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "!jupyter nbconvert --to html advanced_ope_analysis.ipynb"
    ))

    nb["cells"] = cells

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True, parents=True)
    with open(examples_dir / "advanced_ope_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
if __name__ == "__main__":
    create_notebook()
    print("Notebook generated.")
