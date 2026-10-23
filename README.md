# Predicción de Cancelación de Reservaciones Hoteleras

Proyecto de clasificación binaria en Python/scikit-learn para estimar la probabilidad de que una reservación hotelera se cancele, como apoyo a decisiones de inventario (overbooking controlado, depósitos de garantía, recordatorios).

Proyecto final de **Algoritmos de Aprendizaje Automático** · Equipo 3 · Universidad Tecmilenio.

📓 **Notebook (Google Colab):** [Abrir en Colab](TU_LINK_DE_COLAB)

## Datos

- Dataset sintético proporcionado por el docente, incluido en [`data/06_cancelacion_reservaciones.csv`](data/06_cancelacion_reservaciones.csv) (1,000 filas × 14 columnas).
- Variable objetivo: `reservacion_cancelada` (1 = cancelada, 0 = no cancelada). Desbalance ≈ 78/22.
- Problemas de calidad tratados: 17 nulos en `tarifa_promedio_mxn`, 6 duplicados exactos (quedan 994 registros), categorías inconsistentes en `canal_reservacion` y un posible valor atípico en `dias_anticipacion`.

## Estructura

```
.
├── prediccion_cancelacion_reservaciones.ipynb   # Notebook con todo el análisis
├── data/
│   └── 06_cancelacion_reservaciones.csv         # Dataset (1,000 reservaciones)
└── README.md
```

## Metodología

1. Exploración y limpieza de datos.
2. Partición estratificada 70/15/15 (`random_state=42`) sobre el dataset depurado: 695 registros de entrenamiento, 149 de validación y 150 de prueba.
3. Pipeline de preprocesamiento (imputación + `StandardScaler` + `OneHotEncoder`) ajustado solo con entrenamiento para evitar fuga de información.
4. Baseline (`DummyClassifier`) y modelos: Regresión logística, Árbol de decisión, Random Forest, Gradient Boosting, SVM (RBF), KNN, Naive Bayes y MLP.
5. Optimización de hiperparámetros (`GridSearchCV`), selección de características por importancia, y clustering con K-Means.
6. Evaluación con métricas para clases desbalanceadas: precision, recall, F1 y ROC-AUC (no solo accuracy).

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | 0.785 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Regresión logística (balanceada)** | 0.691 | 0.370 | 0.625 | 0.465 | 0.723 |
| Random Forest (optimizado) | 0.765 | 0.421 | 0.250 | 0.314 | 0.687 |
| MLP | 0.785 | 0.000 | 0.000 | 0.000 | 0.560 |

*(Métricas en validación.)*

**Modelo final:** Regresión logística con `class_weight="balanced"`. En el conjunto de prueba obtuvo Recall 0.469, F1 0.345 y ROC-AUC 0.655.

Hallazgo clave: balancear las clases benefició sobre todo al modelo lineal; los modelos más complejos (ensambles, SVM, MLP) casi no detectaron cancelaciones con el umbral por defecto, a pesar de tener mayor accuracy.

## Limitaciones

- Precisión baja (≈0.27–0.37): habrá falsos positivos en uso real.
- Conjuntos de validación y prueba pequeños (~150 registros), por lo que las métricas tienen alta variabilidad.
- El modelo es una **herramienta de apoyo** a la decisión humana, no una regla automática.

## Cómo reproducirlo

1. Abre el notebook en Google Colab.
2. Sube `data/06_cancelacion_reservaciones.csv` (incluido en este repositorio) al entorno de Colab.
3. Ejecuta las celdas en orden.

Dependencias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Equipo 3

Francisco Aram Rodríguez García · Eliab Reyes Olvera · Viker Alberto Guajardo Díaz · Axcel Francisco Salazar Alejo ([@Axcelini](https://github.com/Axcelini))

Docente: Frederik Timoteo Castillo Pedraza
