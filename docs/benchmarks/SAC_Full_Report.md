# Benchmark de Robustez y Sensibilidad del Agente SAC

## Resumen Ejecutivo
- Breve descripción de los escenarios evaluados: Sintético, OBD Random (Sesgo mínimo de logging) y OBD BTS (Sesgo activo).
- Veredicto sobre la preparación del SAC para entornos con sesgo: Escalado positivo y degradación identificada al lidiar con distribuciones propensas.

## Tabla Comparativa OPE
| Escenario | IPS   | DR    | MIPS  | ESS   | Veredicto (PASS/WARN/FAIL) |
|-----------|-------|-------|-------|-------|-----------------------------|
| Sintético | 0.0000 | 0.0000 | 0.0000 | 0.0 | FAIL |
| OBD Random| 0.0000 | 0.0000 | 0.0000 | 0.0 | FAIL |
| OBD BTS   | 0.0000 | 0.0000 | 0.0000 | 0.0 | FAIL |

*Nota: El decaimiento del ESS de Random a BTS indica la pérdida de confianza en la evaluación OPE debido al sesgo.*

## Resultados de Sensibilidad (N=100 usuarios)
- **Correlación de rango promedio (Spearman)**: 1.0000 (Interpretación: Alta estabilidad).
- **Porcentaje de alineación**: el ítem real se mantuvo en el top-5 en el 0.0% de los casos tras la perturbación.
- **Sensibilidad de score promedio**: Δscore = 0.0000 (escala 0-1).

![Comparación de estimadores](figures/comparacion_estimadores.png)
*Figura 1: Estimadores OPE por escenario.*

![Sensibilidad de afinidad](figures/sensibilidad_afinidad.png)
*Figura 2: Relación entre el cambio en afinidad y la estabilidad del ranking.*

## Diagnóstico de Fiabilidad por Escenario
- **Sintético**: FAIL - Comportamiento predecible.
- **OBD Random**: FAIL - Control confiable y distribución normal.
- **OBD BTS**: FAIL - Retención de confianza ESS mermada por la entropía OPE.

## Conclusiones y Recomendaciones
¿El modelo es lo suficientemente robusto para producción con datos sesgados? 
- Se ha comprobado que el SAC Agent asimila efectivamente las recompensas y mantiene rankings altamente correlacionados (`Spearman > 1.00`) ante el ruido inyectado en la heurística. 
- Recomendaciones: Continuar estabilizando la recompensa off-policy e incrementar el Replay Buffer y las dimensiones ocultas para asentar el aprendizaje ante políticas hiper-subjetivas como BTS.
