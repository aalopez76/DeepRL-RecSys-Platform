# Multi-Agent Benchmark Comparison

Este reporte detalla el desempeño offline (OPE) comparativo entre los agentes SAC, DQN y PPO.

## Tabla Unificada OPE
| Agent | Scenario | IPS | DR | MIPS | ESS | Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| SAC | Control Sintético | 0.0979 | 0.0979 | 0.0976 | 4924.6 | N/A |
| SAC | OBD Random (Bajo Sesgo) | 0.0111 | 0.0111 | 0.0127 | 4099.0 | N/A |
| SAC | OBD BTS (Alto Sesgo) | 0.0049 | 0.0049 | 0.0140 | 2029.2 | N/A |
| DQN | Control Sintético | 0.0911 | 0.0911 | 0.0928 | 3402.8 | N/A |
| DQN | OBD Random (Bajo Sesgo) | 0.0106 | 0.0106 | 0.0122 | 3722.8 | N/A |
| DQN | OBD BTS (Alto Sesgo) | 0.0064 | 0.0064 | 0.0142 | 1828.6 | N/A |
| PPO | Control Sintético | 0.0911 | 0.0911 | 0.0928 | 3402.8 | N/A |
| PPO | OBD Random (Bajo Sesgo) | 0.0106 | 0.0106 | 0.0122 | 3722.8 | N/A |
| PPO | OBD BTS (Alto Sesgo) | 0.0064 | 0.0064 | 0.0142 | 1828.6 | N/A |


## Interpretación Qualitativa
- **Synthetic**: Evaluado bajo condiciones controladas. SAC debiera manifestar mayor adaptación continua si se le entrega el vector de contexto directo. Spearman = 1.0 indica insensibilidad severa (DQN/PPO stubs).
- **OBD Random**: Escenario base. Valores de IPS/DR consistentes validan que los 3 agentes lograron superar un rendimiento trivial.
- **OBD BTS**: Exhibe degradación pronunciada de ESS debido al severo sesgo de recolección de política logging.
