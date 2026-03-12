# DeepRL-RecSys-Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Tests](https://img.shields.io/badge/tests-123%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-%3E85%25-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## 🚀 Uso rápido

**Entrenar un agente (baseline Random)**
```bash
deeprl-recsys train --config configs/experiments/exp1_dqn_movielens.yaml
```

**Evaluar con OPE**
```bash
deeprl-recsys evaluate --config configs/experiments/exp1_dqn_movielens.yaml
```

**Lanzar el dashboard interactivo**
```bash
deeprl-recsys ui
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
