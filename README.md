# DeepRL-RecSys-Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Tests](https://img.shields.io/badge/tests-123%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-%3E85%25-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deeprl-recsys.streamlit.app)

**DeepRL-RecSys-Platform** es un framework de grado industrial para construir, evaluar y desplegar sistemas de recomendación basados en Deep Reinforcement Learning (DRL). Combina agentes avanzados (DQN, PPO, SAC), evaluación offline robusta (OPE) y una arquitectura lista para producción.

## ✨ Características clave

- **Agentes RL funcionales**: DQN, PPO y SAC implementados con PyTorch y listos para usar.
- **Evaluación Off-Policy (OPE)**: Estimadores IPS, DR y MIPS con diagnósticos de fiabilidad (ESS, clipping).
- **Inferencia escalable**: Servicio FastAPI con trazabilidad de peticiones, autenticación opcional y soporte para batching.
- **Dashboard interactivo**: Visualiza experimentos, curvas de entrenamiento y prueba recomendaciones en tiempo real con Streamlit.
- **Modular y extensible**: Arquitectura limpia con `core` ligero, lazy-loading y extras opcionales (`[torch]`, `[llm]`, `[ui]`).

## 📦 Instalación

El proyecto usa [Poetry](https://python-poetry.org/) para gestión de dependencias.

1. Clona el repositorio:
   ```bash
   git clone https://github.com/aalopez76/DeepRL-RecSys-Platform.git
   cd DeepRL-RecSys-Platform
   ```

2. Instala el paquete con extras opcionales (por ejemplo, para el dashboard):
   ```bash
   poetry install --extras ui
   ```

3. Activa el entorno virtual:
   ```bash
   poetry shell
   ```

## 🚀 Ejemplo Rápido (Notebook)

¿Quieres ver la plataforma en acción sin configurar nada? Hemos preparado un Notebook interactivo que muestra el flujo completo (End-to-End): generación de datos sintéticos, entrenamiento SAC, evaluación OPE y despliegue del motor de Inferencia.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aalopez76/DeepRL-RecSys-Platform/blob/master/examples/demo_train_eval_serve.ipynb)

👉 **[Ver Notebook Básico (Entrenamiento/Servicio): `examples/demo_train_eval_serve.ipynb`](./examples/demo_train_eval_serve.ipynb)**

👉 **[Ver Notebook Avanzado (Interactividad OPE & Sensibilidad): `examples/advanced_ope_analysis.ipynb`](./examples/advanced_ope_analysis.ipynb)**

## 📊 Integración E2E: Open Bandit Dataset (OBD)

La plataforma soporta nativamente la ingesta y preparación de **Open Bandit Dataset** para Benchmarks con datos reales distribuidos:

1. **Transformación Vectorizada Parquet:** Extrae, mapea, comprime y formatea el contexto a nivel JSON.
   ```bash
   python scripts/prepare_obd.py --policy random --campaign all
   ```
2. **Entrenamiento Continuo SAC:** Dispara tu baseline DQN/SAC con el buffer generado sin estropear memoria.
   ```bash
   deeprl-recsys train --config configs/experiments/exp_obd_sac_real.yaml
   ```
3. **Evaluación Contra-factual (OPE):** Retén los índices de propensión (IPS, Doubly Robust, etc.) sobre recompensas con clip de $\epsilon$.
   ```bash
   deeprl-recsys evaluate --config configs/experiments/exp_obd_sac_real.yaml
   ```

4. **Orquestación Completa (Single-Agent):** Ejecuta todo el benchmark (Synthetic, OBD Random, OBD BTS) y regenera los reportes automáticamente en un solo paso.
   ```bash
   python scripts/run_full_benchmark.py --agent dqn
   ```

5. **Multi-Agent Benchmark (🏆):** Ejecuta iterativamente todos los escenarios para `SAC`, `DQN` y `PPO`. Limpia la memoria GPU/CPU entre corridas y genera el documento comparativo `Agents_Comparison.md`.
   ```bash
   python scripts/run_all_agents_benchmark.py
   ```

## 🌐 Dashboard Online (Streamlit Cloud)

Hemos desplegado un **Dashboard Interactivo y Analítico** para visualizar fácilmente los reportes OPE, los simuladores de sensibilidad de ranking y las tablas maestras.
- Puedes visitarlo en línea: **[DeepRL-RecSys Streamlit App](https://deeprl-recsys.streamlit.app)**
- Soporta inferencia y test de "Recommendation Playground" usando modelos locales preentrenados cargados on-the-fly (`artifacts/models/`). Si no se incluye el id del modelo, buscará iterativamente el modelo de ejemplo o te advertirá afablemente que corras un Benchmark primero!

## ⚡ Uso por CLI

**Entrenar un agente (baseline Random)**
```bash
deeprl-recsys train --config configs/experiments/exp1_dqn_movielens.yaml
```

**Evaluar con OPE**
```bash
deeprl-recsys evaluate --config configs/experiments/exp1_dqn_movielens.yaml
```

**Lanzar el dashboard interactivo local**
```bash
python -m streamlit run streamlit_app.py
```

**Servir el modelo (FastAPI)**
```bash
deeprl-recsys serve --artifact ./artifacts/models/your_run_id
```

## 🧪 Calidad del código

- Más de 123 pruebas unitarias y de integración.
- Cobertura de código >85% en módulos críticos.
- Pre-commit hooks con Ruff, Black y MyPy.
- CI/CD con GitHub Actions.

## 🗂️ Estructura del proyecto

```text
DeepRL-RecSys-Platform/
├── src/deeprl_recsys/          # Paquete principal (SDK)
│   ├── core/                   # Contratos, configuración, artefactos
│   ├── agents/                  # DQN, PPO, SAC, baselines
│   ├── training/                 # Bucle de entrenamiento y callbacks
│   ├── evaluation/               # Métricas y OPE
│   ├── serving/                  # FastAPI, runtime, middleware
│   ├── ui/                       # Dashboard Streamlit
│   └── cli.py                    # Interfaz de línea de comandos
├── configs/                      # YAMLs para experimentos
├── pipelines/                     # Scripts de orquestación
├── tests/                         # Pruebas unitarias e integración
├── artifacts/                     # Modelos, logs, reportes (gitignored)
└── pyproject.toml                 # Configuración de Poetry y extras
```

## 📄 Licencia

Distribuido bajo la licencia MIT. Ver LICENSE para más información.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request siguiendo las plantillas.
