# TP2 — Perceptrón Multicapa: Descripción Detallada

## 1. Introducción

Este trabajo práctico extiende el perceptrón simple del TP1 a un **Perceptrón Multicapa (MLP)** con una o más capas ocultas, utilizando el algoritmo de **retropropagación (backpropagation)**. Se abordan tres problemas:

1. **XOR** — demostrar que el MLP resuelve el problema que el perceptrón simple no pudo.
2. **Clasificación par/impar** — a partir de imágenes binarias de dígitos 5×7.
3. **Reconocimiento de dígitos** — con 10 salidas one-hot y evaluación de robustez ante ruido.

El objetivo es comprender cómo las capas ocultas y la retropropagación permiten resolver problemas no lineales, y cómo se aplica un MLP a un problema real de reconocimiento de patrones.

---

## 2. Fundamentos Teóricos

### 2.1 Perceptrón Multicapa (MLP)

El MLP es una red neuronal con una o más **capas ocultas** entre la entrada y la salida. Cada capa está compuesta por neuronas con una función de activación no lineal. La presencia de capas ocultas permite al MLP aproximar funciones no lineales, superando la limitación del perceptrón simple.

Arquitectura general:
```
Capa de entrada → Capa oculta 1 → ... → Capa oculta N → Capa de salida
```

Cada capa realiza: **V_{m} = g_m(W_m · V_{m-1} + b_m)**

### 2.2 Algoritmo de Retropropagación (Backpropagation)

El entrenamiento del MLP utiliza el algoritmo de retropropagación, que calcula el gradiente del error respecto a cada peso propagando el error desde la salida hacia atrás:

**Paso 1 — Forward:** Se propaga la entrada capa por capa, almacenando los potenciales de activación h_m y las salidas V_m de cada capa.

**Paso 2 — Backward (cálculo de deltas):**
- Capa de salida M: δ_M = g'_M(h_M) * (ζ − O)
- Capa oculta m: δ_m = g'_m(h_m) * (W_{m+1}^T · δ_{m+1})

**Paso 3 — Gradientes:**
- ΔW_m = η · δ_m ⊗ V_{m-1} (producto exterior)
- Δb_m = η · δ_m

**Paso 4 — Actualización:** W_m ← W_m − ΔW_m, b_m ← b_m − Δb_m

### 2.3 Inicialización de Pesos (Xavier Uniform)

Los pesos se inicializan con distribución uniforme en el rango:

```
límite = √(6 / (fan_in + fan_out))
W ∼ Uniforme(−límite, límite)
```

Esta inicialización (Xavier/Glorot) mantiene la varianza de las activaciones constante a través de las capas, evitando que los gradientes se desvanezcan o exploten en redes profundas.

### 2.4 Entrenamiento Online vs Batch

El MLP implementa dos modos de entrenamiento:

- **Online (`train_sample`)**: los pesos se actualizan después de cada muestra. Converge más rápido pero es más sensible al ruido.
- **Batch (`train_batch`)**: se acumulan los gradientes de todas las muestras y se actualizan los pesos una vez por época. Es más estable pero requiere más memoria.

---

## 3. Estructura del Código

```
tp2PerceptronMulticapa/
├── main.py              # Punto de entrada
├── __init__.py           # Exportaciones
├── activation.py         # Funciones de activación (idéntico a TP1)
├── mlp.py                # Clase ModularMultiLayerPerceptron
├── data.py               # Carga de dígitos, split, normalización, evaluación
├── visualization.py      # Visualizaciones (frontera, error, dígitos, ruido)
├── app.py                # Aplicación con menú interactivo
└── datos/
    └── TP2-ej3-mapa-de-pixeles-digitos-decimales.txt  # 10 dígitos en 7×5 píxeles
```

---

## 4. Ejercicio 1: XOR con Perceptrón Multicapa

### 4.1 Qué hace

Entrena un MLP con arquitectura **2 entradas → 2 neuronas ocultas (tanh) → 1 salida (tanh)** para aprender la función XOR.

### 4.2 Datos

| ξ₁ | ξ₂ | ζ (XOR) |
|---|---|---|
| −1 | −1 | −1 |
| −1 |  1 |  1 |
|  1 | −1 |  1 |
|  1 |  1 | −1 |

### 4.3 Cómo funciona

A diferencia del perceptrón simple (TP1), el MLP tiene una **capa oculta** de 2 neuronas con activación tanh. Esta capa oculta transforma el espacio de entrada en un nuevo espacio de representación donde las clases se vuelven linealmente separables.

El proceso de entrenamiento:
1. **Forward:** ξ → capa oculta (tanh) → capa de salida (tanh) → O
2. **Backward:** se calcula el error en la salida, se propaga hacia atrás para ajustar los pesos de ambas capas
3. Se repite por hasta 10000 épocas

Para encontrar la mejor configuración, se realiza una **búsqueda de hiperparámetros** probando todas las combinaciones de:
- **η** (tasa de aprendizaje): 0.01, 0.05, 0.1, 0.5, 1.0
- **β** (constante de la función de activación): 0.5, 0.8, 1.0, 1.5, 2.0

De todas las combinaciones, se conserva el modelo con menor error final. Luego se evalúan los 4 patrones y se cuentan los aciertos.

### 4.4 Por qué funciona ahora

El perceptrón simple falló en XOR porque solo podía trazar una recta. El MLP con una capa oculta de 2 neuronas puede:

1. La capa oculta transforma el espacio de entrada (2D) en un nuevo espacio de representación.
2. Cada neurona oculta aprende una recta separadora diferente.
3. La capa de salida combina estas rectas para crear una frontera de decisión no lineal.

Geométricamente, las dos neuronas ocultas aprenden dos rectas que, combinadas, producen una región de decisión que separa las cuatro esquinas del XOR correctamente.

### 4.5 Resultado

El MLP **converge** para XOR (a diferencia del perceptrón simple del TP1). La mejor combinación de hiperparámetros suele ser η=0.5, β=1.5, convergiendo en ~200 épocas con error < 0.01 y **4/4 aciertos** en los patrones de entrenamiento. La visualización muestra la frontera de decisión no lineal que separa las cuatro regiones.

---

## 5. Ejercicio 2: Clasificación Par/Impar desde Imágenes 5×7

### 5.1 Qué hace

Entrena un MLP para clasificar dígitos escritos en una matriz de 5×7 píxeles como **par** (0,2,4,6,8 → salida +1) o **impar** (1,3,5,7,9 → salida −1).

### 5.2 Datos

El archivo `TP2-ej3-mapa-de-pixeles-digitos-decimales.txt` contiene 10 dígitos (0 al 9), cada uno representado como una matriz binaria de 7 filas × 5 columnas. Un pixel con valor 1 representa tinta (foreground), 0 es fondo.

Cada dígito se aplana a un vector de 35 elementos (7×5) que sirve como entrada al MLP.

### 5.3 Cómo funciona

- **Arquitectura:** 35 entradas → 10 neuronas ocultas (tanh) → 1 salida (tanh)
- **Target:** +1 para dígitos pares (0,2,4,6,8), −1 para impares (1,3,5,7,9)
- **Split:** 60% entrenamiento (6 dígitos), 40% testeo (4 dígitos)
- **Entrenamiento:** batch, hasta 1000 épocas, η = 0.05

### 5.4 Por qué

Este ejercicio demuestra que un MLP puede aprender una **regla abstracta** (paridad) a partir de representaciones visuales de píxeles. No se le dice explícitamente qué es "par"; debe descubrir la regularidad subyacente en los patrones de bits.

### 5.5 Resultado

El modelo aprende a clasificar par/impar con alta precisión. La curva de error por época muestra convergencia. La generalización se evalúa con dígitos no vistos en entrenamiento.

---

## 6. Ejercicio 3: Reconocimiento de Dígitos con 10 Salidas One-Hot

### 6.1 Qué hace

Entrena un MLP para reconocer los 10 dígitos (0-9) a partir de sus imágenes de 5×7 píxeles, usando codificación **one-hot** en la salida (10 neuronas, una por dígito). Además, evalúa la robustez del modelo ante ruido.

### 6.2 Datos

Los mismos 10 dígitos del archivo de píxeles. Cada dígito se aplana a un vector de 35 elementos. El target es un vector de 10 elementos con 1 en la posición del dígito y 0 en las demás (codificación one-hot).

### 6.3 Cómo funciona

Se entrenan dos modelos:

**Modelo 35-15-10 (one-hot con ruido):**
- Arquitectura: 35 entradas → 15 ocultas (tanh) → 10 salidas (tanh)
- Targets: one-hot (vector de 10 con 1 en la posición del dígito correcto)
- Split: 6 dígitos para entrenar, 4 para testear
- Se evalúa accuracy en test y también en entrenamiento con ruido (inversión de bits con probabilidad p configurable por el usuario, default 0.02)

**Modelo 35-20-10 (reconocimiento completo):**
- Arquitectura: 35 entradas → 20 ocultas (tanh) → 10 salidas (tanh)
- Se entrena con los 10 dígitos (sin split)
- Se usa para la visualización de la grilla de predicción

### 6.4 Por qué

- **One-hot** permite clasificación multiclase con 10 salidas, donde cada neurona de salida representa un dígito.
- **Ruido:** simula errores de sensor o transmisión. Evalúa si el modelo es robusto a pequeñas perturbaciones en la entrada.
- **Paridad:** además de reconocer el dígito, se clasifica en par/impar como verificación adicional.

### 6.5 Resultado

- El modelo 35-15-10 alcanza alta precisión en test (~80-100% con 4 dígitos de test).
- Con ruido p=0.02, la precisión en entrenamiento se degrada ligeramente, demostrando cierta robustez.
- El modelo 35-20-10 (entrenado con los 10 dígitos) reconoce todos los dígitos correctamente.
- La grilla de predicción muestra cada dígito con su clasificación de paridad.

---

## 7. Visualizaciones

### 7.1 Frontera de Decisión (XOR)

**Qué muestra:** Los 4 puntos del XOR coloreados por clase, con el mapa de decisión del MLP (fondo rojo/azul según la región clasificada como −1 o +1) y la frontera de decisión (contorno negro). A diferencia del perceptrón simple, el MLP produce una frontera no lineal que separa correctamente las cuatro esquinas.

### 7.2 Error por Época (XOR)

**Qué muestra:** La evolución del error cuadrático medio durante el entrenamiento del MLP para XOR. La curva desciende y se estabiliza cerca de cero, demostrando convergencia.

### 7.3 Grilla de Reconocimiento de Dígitos

**Qué muestra:** Una grilla 2×5 con los 10 dígitos. Cada celda muestra la imagen 5×7 del dígito, el dígito real, la predicción del modelo, y la clasificación de paridad. El color del título (verde/rojo) indica si la predicción fue correcta.

### 7.4 Comparación Original vs Ruido

**Qué muestra:** Para cada dígito de entrenamiento, dos imágenes lado a lado: la original (izquierda) y la versión con ruido (derecha), con la predicción del modelo en cada caso. El usuario puede elegir la probabilidad de ruido p. Se muestra el accuracy general con ruido.

### 7.5 Resaltado de Dígito en Consola

**Qué muestra:** En la terminal, las 7 líneas del dígito seleccionado con los píxeles en 1 resaltados en verde, permitiendo ver la matriz 5×7 directamente.

---

## 8. Resultados Esperados

| Experimento | Arquitectura | Convergencia | Resultado |
|---|---|---|---|
| **XOR** | 2-2-1 (tanh) | Sí, ~200 épocas (η=0.5, β=1.5) | 4/4 aciertos, frontera no lineal separa las 4 esquinas |
| **Par/Impar** | 35-10-1 (tanh) | Sí, ~1000 épocas | Clasifica correctamente dígitos pares e impares |
| **One-hot + ruido** | 35-15-10 (tanh) | Sí, ~2000 épocas | Accuracy alta en test; robustez parcial ante ruido |
| **Reconocimiento completo** | 35-20-10 (tanh) | Sí, ~5000 épocas | Accuracy ~100% en los 10 dígitos de entrenamiento |

---

## 8. Conclusiones

1. **El MLP resuelve XOR:** A diferencia del perceptrón simple, el MLP con una capa oculta puede aprender fronteras de decisión no lineales. La capa oculta transforma el espacio de entrada en uno donde las clases son separables.

2. **Retropropagación:** El algoritmo de backpropagation permite entrenar redes multicapa propagando el error hacia atrás. La derivada de la función de activación es crucial para calcular los deltas.

3. **Clasificación de patrones:** El MLP puede aprender conceptos abstractos (paridad) a partir de representaciones de píxeles, demostrando capacidad de generalización.

4. **Robustez al ruido:** El modelo entrenado con datos limpios mantiene buena precisión cuando se introducen pequeñas perturbaciones (inversión de bits con p=0.02), mostrando cierta tolerancia al ruido.

5. **One-hot para multiclase:** La codificación one-hot con 10 salidas permite clasificación multiclase, donde cada neurona de salida representa una clase diferente.

6. **Batch vs Online:** El entrenamiento batch (promediando gradientes) es más estable que el online para problemas pequeños, y se usa para los modelos de dígitos.

---

## 9. Dependencias

- **numpy**: operaciones vectoriales y álgebra lineal
- **matplotlib**: visualización de gráficos
