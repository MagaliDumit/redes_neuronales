# TP2 - Perceptrón Multicapa

## Archivo: `TP2.py`

### Estructura propuesta

Reutilizar los módulos de `TP1.py` (activaciones, operaciones vectorizadas) y extender con una clase `ModularMultiLayerPerceptron`.

### Nueva clase: `ModularMultiLayerPerceptron`
```python
class ModularMultiLayerPerceptron:
    def __init__(self, layer_dims, activations, eta=0.01):
        """
        layer_dims: lista con [input_dim, hidden_dim, ..., output_dim]
        activations: lista de strings con activación por capa (ej. ['tanh', 'tanh', 'linear'])
        """
        - Inicializar pesos y biases para cada capa
        - Almacenar función de activación y derivada para cada capa
    ```

    Métodos:
    - `forward(V_input)`: propagación hacia adelante capa por capa, almacenando h y V de cada capa
    - `backward(target)`: retropropagación, calcula deltas y acumula gradientes
    - `train_sample(V_input, target, acumular=False)`: forward + backward + actualizar pesos (incremental o batch)
    - `train_batch(X, y)`: entrenamiento por lotes (acumular gradientes, actualizar una vez)

### Algoritmo de retropropagación
Para cada capa m (desde salida M hacia atrás):
- Capa de salida: `δ_M = g'(h_M) * (ζ - O)`
- Capa oculta m: `δ_m = g'(h_m) * (W_{m+1}^T @ δ_{m+1})`
- Actualización: `ΔW_m = η * δ_m (outer) V_{m-1}`

### Main

#### Ejercicio 1: XOR con multicapa
- Arquitectura: 2 → 2 (tanh) → 1 (tanh)
- Entradas: `[[-1,1],[1,-1],[-1,-1],[1,1]]`, targets: `[1,1,-1,-1]`
- Entrenar hasta convergencia o épocas máximas
- Mostrar que AHORA SÍ converge (a diferencia del perceptrón simple)
- Visualización: scatter plot de puntos + frontera de decisión

#### Ejercicio 2: Clasificar par/impar desde imágenes 5×7
- Cargar `TP2-ej3-mapa-de-pixeles-digitos-decimales.txt` (10 dígitos, cada uno como 7×5 píxeles binarios)
- Etiquetas: 1 si el dígito es par, -1 si es impar
- Arquitectura: 35 (input) → hidden (ej. 10, tanh) → 1 (tanh)
- Probar splits de entrenamiento/testeo (ej. entrenar con 6 dígitos, testear con 4)
- Evaluar capacidad de generalización

#### Ejercicio 3: 10 salidas one-hot + ruido
- Arquitectura: 35 → hidden (ej. 15, tanh) → 10 (tanh o softmax)
- Targets: vector one-hot (1 en la posición del dígito, 0 en las otras)
- Probar con ruido: intercambiar bits con probabilidad 0.02 en las imágenes de entrada
- Evaluar accuracy antes y después de agregar ruido

### Visualizaciones
- Matplotlib para curvas de error por época
- Para ejercicio 2: matriz de confusión
- Para ejercicio 3: comparación de accuracy con/sin ruido (gráfico de barras)

### Dependencias
`numpy`, `matplotlib`
