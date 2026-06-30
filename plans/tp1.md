# TP1 - Perceptrón Simple

## Archivo: `TP1.py`

### Módulos existentes (sin cambios)
- Funciones de activación: step, linear, sigmoid, tanh, relu (con sus derivadas)
- Operaciones vectorizadas: `compute_activation_potential`, `compute_output`, `compute_output_error`, `compute_output_delta`, `compute_hidden_delta`, `compute_weight_correction`
- Clase `ModularPerceptron`: `__init__`, `_add_bias`, `forward`, `train_sample`
- Funciones auxiliares: `train_test_split_data`, `evaluate_accuracy`

### Nuevo: Visualización con convexos cerrados (Parte 1 - datos 2D)
```python
def plot_convex_hulls_and_hyperplane(X, y, title=""):
    """
    - Scatter plot coloreado por clase (-1 rojo, 1 azul)
    - Convex hull de cada clase (polígono semitransparente)
    - Distancia mínima entre convexos (línea punteada)
    - Hiperplano perpendicular al segmento de mínima distancia en el punto medio
    """
```
Dependencia: `scipy.spatial.ConvexHull`

### Nuevo: Visualización de regresión (Parte 2)
```python
def plot_regression_results(y_true, y_pred, title=""):
    """Scatter y_true vs y_pred + recta ideal y=x"""

def plot_error_history(errors, title=""):
    """Error cuadrático total vs épocas"""
```

### Main
- **Parte 1a (AND):** entrenar perceptrón escalón → `plot_convex_hulls_and_hyperplane(X, y, "AND - Linealmente separable")`
- **Parte 1b (XOR):** entrenar (frena por épocas máximas, no por error=0) → `plot_convex_hulls_and_hyperplane(X, y, "XOR - NO separable")`
- **Parte 2a (lineal):** cargar datos TP1-ej2, split 80/20, entrenar → plot error vs épocas + plot y_true vs y_pred
- **Parte 2b (no lineal):** idem con tanh (salidas normalizadas a [-0.9, 0.9]) → comparar gráficos
- **Parte 2c (generalización):** probar splits [90/10, 80/20, 70/30] con distintos shuffles → gráfico error en test vs tamaño de entrenamiento

### Dependencias
`numpy`, `matplotlib`, `scipy.spatial.ConvexHull`
