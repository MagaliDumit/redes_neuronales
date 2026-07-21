# TP1 — Perceptrón Simple: Descripción Detallada

## 1. Introducción

Este trabajo práctico implementa un **Perceptrón Simple** desde cero (sin librerías de redes neuronales) y lo aplica a dos tipos de problemas:

- **Clasificación binaria** (compuertas lógicas AND y XOR) — para demostrar el concepto de separabilidad lineal.
- **Regresión** (datos reales de 3 variables de entrada, 1 salida continua) — para modelar una función desconocida a partir de ejemplos.

El objetivo es comprender los fundamentos del aprendizaje supervisado: cómo un modelo lineal aprende pesos sinápticos, qué limitaciones tiene, y cómo funciones de activación no lineales permiten modelar relaciones más complejas.

---

## 2. Fundamentos Teóricos

### 2.1 El Perceptrón Simple

El perceptrón simple es la unidad fundamental de una red neuronal. Fue propuesto por Frank Rosenblatt en 1958. Modela una neurona biológica como una función matemática que:

1. **Recibe** entradas ξ = (ξ₁, ξ₂, ..., ξₙ)
2. **Pondera** cada entrada con un peso sináptico w = (w₁, w₂, ..., wₙ)
3. **Suma** el potencial de activación: h = Σ wᵢ·ξᵢ + w₀ (donde w₀ es el sesgo o bias)
4. **Aplica** una función de activación g(h) para producir la salida O = g(h)

### 2.2 Regla de aprendizaje (Descenso por gradiente)

El perceptrón aprende minimizando el error cuadrático entre la salida deseada ζ y la salida obtenida O:

```
E = ½(ζ − O)²
```

La regla de actualización de pesos se obtiene derivando E respecto a cada peso wⱼ y aplicando descenso por gradiente:

```
Δwⱼ = −η · ∂E/∂wⱼ = η · (ζ − O) · g'(h) · ξⱼ
```

Donde:
- **η** (eta) es la tasa de aprendizaje
- **g'(h)** es la derivada de la función de activación
- **ξⱼ** es la entrada j-ésima

### 2.3 Funciones de activación implementadas

| Función | Fórmula | Derivada | Rango | Uso |
|---|---|---|---|---|
| **Step** (escalón) | θ(h) = 1 si h ≥ 0, −1 si h < 0 | θ'(h) = 1 | {−1, 1} | Clasificación binaria (AND, XOR) |
| **Lineal** | g(h) = h | g'(h) = 1 | (−∞, ∞) | Regresión lineal |
| **Tanh** | tanh(βh) | β(1 − tanh²(βh)) | (−1, 1) | Regresión no lineal |
| **Logística** | σ(2βh) = 1/(1+e^(−2βh)) | 2β·σ·(1−σ) | (0, 1) | Regresión no lineal (alternativa) |
| **ReLU** | max(0, h) | 1 si h > 0, 0 si no | [0, ∞) | No usada en TP1 |

---

## 3. Estructura del Código

```
tp1PerceptronSimple/
├── main.py              # Punto de entrada
├── __init__.py           # Exportaciones
├── activation.py         # Funciones de activación y sus derivadas
├── perceptron.py         # Clase Perceptron (forward, train, predict)
├── data.py               # Carga, split, normalización, evaluación
├── visualization.py      # Visualizaciones (convex hulls, regresión, error, generalización)
├── app.py                # Aplicación con menú interactivo
└── datos/
    ├── TP1-ej2-Conjunto-entrenamiento.txt   # 200 muestras, 3 features
    └── TP1-ej2-Salida-deseada.txt           # 200 targets continuos
```

---

## 4. Parte 1: Clasificación con Compuertas Lógicas

### 4.1 AND — Linealmente separable

**Qué hace:** Entrena un perceptrón con activación escalón (step) para aprender la compuerta AND.

**Datos:** 4 puntos en 2D:
| ξ₁ | ξ₂ | ζ (AND) |
|---|---|---|
| −1 | −1 | −1 |
| −1 |  1 | −1 |
|  1 | −1 | −1 |
|  1 |  1 |  1 |

**Cómo funciona:**
- El perceptrón inicializa pesos aleatorios y procesa cada muestra.
- Para cada muestra, calcula el potencial de activación h = w·ξ + w₀, luego aplica la función escalón: O = 1 si h ≥ 0, −1 si h < 0.
- Compara O con el target ζ. Si hay error, actualiza los pesos con Δw = η(ζ − O)ξ.
- Repite por épocas hasta que el error total es 0 (convergencia) o se alcanza el máximo de épocas.

**Por qué:** AND es el ejemplo canónico de un problema linealmente separable. El perceptrón puede encontrar un hiperplano (recta en 2D) que separa las dos clases sin error.

**Resultado:** El perceptrón converge típicamente en 2-10 épocas. La recta separadora (hiperplano) se calcula a partir de los pesos aprendidos y se visualiza junto con los convexos de cada clase.

### 4.2 XOR — No linealmente separable

**Qué hace:** Entrena un perceptrón escalón para aprender XOR.

**Datos:**
| ξ₁ | ξ₂ | ζ (XOR) |
|---|---|---|
| −1 | −1 | −1 |
| −1 |  1 |  1 |
|  1 | −1 |  1 |
|  1 |  1 | −1 |

**Cómo funciona:** Idéntico al caso AND, pero con los targets de XOR.

**Por qué:** XOR es el contraejemplo clásico que demuestra la limitación fundamental del perceptrón simple: no puede aprender una función que no sea linealmente separable. En XOR, las clases se distribuyen en las esquinas opuestas del cuadrado, y ninguna recta puede separar (1,1) y (−1,−1) de (−1,1) y (1,−1).

**Resultado:** El perceptrón **no converge**. Alcanza el máximo de épocas (200) sin lograr error cero. Los convexos de ambas clases se intersectan, demostrando visualmente que no existe un hiperplano separador. Esto motiva la necesidad de redes multicapa (como el perceptrón multicapa con capa oculta) para resolver XOR.

---

## 5. Parte 2: Regresión con Datos Reales

### 5.1 Datos de entrada

Se cargan 200 muestras desde archivos de texto. Cada muestra tiene 3 variables de entrada (ξ₁, ξ₂, ξ₃) y una salida continua ζ (valores entre ~0.15 y ~99.38). La relación entre entrada y salida es desconocida a priori; el perceptrón debe aproximarla.

### 5.2 Perceptrón Lineal — Regresión lineal

**Qué hace:** Entrena un perceptrón con activación lineal (identidad) para aproximar la función que mapea las 3 entradas a la salida continua.

**Cómo funciona:**
- La salida es simplemente O = w·ξ = Σ wᵢ·ξᵢ + w₀ (combinación lineal de las entradas).
- El error es E = ½(ζ − O)². Como g'(h) = 1, la regla de actualización se simplifica a Δw = η(ζ − O)ξ.
- Se entrena por 200 épocas con tasa de aprendizaje η = 0.0001.
- Los datos se dividen en 80% entrenamiento, 20% testeo (con shuffle).

**Por qué:** La activación lineal asume que la relación entre entradas y salida es aproximadamente lineal. Es el modelo más simple para regresión.

**Resultado:** Se obtienen dos gráficos:
1. **Error por época:** el error cuadrático total disminuye y se estabiliza, mostrando convergencia.
2. **Predicción vs Real:** los puntos se agrupan cerca de la recta y=x, indicando que el modelo captura la tendencia general. Sin embargo, pueden verse desviaciones si la relación real no es perfectamente lineal.

### 5.3 Perceptrón Tanh — Regresión no lineal

**Qué hace:** Entrena un perceptrón con activación tangente hiperbólica (tanh) para modelar relaciones no lineales.

**Cómo funciona:**
- La salida es O = tanh(βh), con β = 1. El rango de salida es (−1, 1).
- Los targets se normalizan al rango [−1, 1] antes de entrenar.
- La derivada g'(h) = β(1 − tanh²(βh)) = β(1 − O²) permite que el gradiente se ajuste según la saturación de la neurona.
- Se entrena por 300 épocas con η = 0.01.
- Al predecir, las salidas se desnormalizan al rango original para comparar con los valores reales.

**Por qué:** La activación tanh introduce no linealidad. A diferencia del perceptrón lineal, puede modelar curvaturas y relaciones más complejas entre entrada y salida. El rango (−1, 1) la hace adecuada para datos centrados en cero.

**Resultado:** El error suele ser menor que con el perceptrón lineal, y la predicción vs real muestra puntos más cercanos a la recta y=x, indicando un mejor ajuste.

### 5.3 Perceptrón Logístico — Regresión no lineal (alternativa)

**Qué hace:** Idem al tanh pero con activación logística (sigmoidea), con rango (0, 1).

**Cómo funciona:**
- O = σ(2βh) = 1/(1 + e^(−2βh)), con β = 1.
- La derivada es g'(h) = 2β · O · (1 − O).
- Los targets se normalizan a [0, 1] en lugar de [−1, 1].

**Por qué:** La logística es otra función de activación no lineal con forma de S. A diferencia de tanh, su rango es (0, 1), lo que la hace interpretable como probabilidad. Sirve como comparación con tanh para ver cómo distintas no linealidades afectan el aprendizaje.

**Resultado:** Similar a tanh, el error disminuye y las predicciones se aproximan a los valores reales. La comparación entre lineal, tanh y logística permite observar qué activación se ajusta mejor a los datos.

### 5.4 Generalización — Error vs tamaño de entrenamiento

**Qué hace:** Evalúa cómo la cantidad de datos de entrenamiento afecta el error en test.

**Cómo funciona:**
- Se prueban 5 tamaños de test: 10%, 20%, 30%, 40%, 50% del total.
- Para cada tamaño, se realizan 15 corridas con diferentes shuffles aleatorios.
- En cada corrida se entrena un perceptrón lineal por 150 épocas y se mide el MSE en test.
- Se grafica la media del error en test ± desvío estándar para cada tamaño.

**Por qué:** La generalización es un concepto fundamental en aprendizaje automático. Con pocos datos de entrenamiento, el modelo no captura bien la distribución subyacente y tiene alto error en test. A medida que se usa más datos para entrenar, el error en test disminuye y se estabiliza. Este experimento cuantifica esa relación.

**Resultado:** El gráfico muestra que a mayor cantidad de datos de entrenamiento (menor test), el error en test disminuye y la varianza entre corridas se reduce. Esto demuestra que más datos de entrenamiento mejoran la generalización.

---

## 6. Visualizaciones

### 6.1 Convex Hulls e Hiperplano Separador

**Qué muestra:** Los puntos de cada clase coloreados (rojo/azul), la envolvente convexa (convex hull) de cada clase como polígono semitransparente, y el hiperplano separador calculado como la recta perpendicular al segmento de mínima distancia entre los dos convexos, pasando por el punto medio.

**Cómo se calcula el hiperplano:**
1. Se calcula la envolvente convexa de cada clase usando `scipy.spatial.ConvexHull`.
2. Se encuentra la distancia mínima entre los dos polígonos convexos (punto a punto, punto a segmento, o segmento a segmento).
3. El hiperplano es la recta perpendicular al segmento de mínima distancia, trazada en el punto medio.

**Por qué:** Esta visualización permite ver geométricamente si las clases son separables. Si los convexos no se intersectan, existe un hiperplano que las separa (AND). Si se intersectan, no hay separación lineal posible (XOR).

### 6.2 Regresión: Predicción vs Real

**Qué muestra:** Un scatter plot donde cada punto es (ζ_real, O_predicha). La recta y=x (roja discontinua) representa la predicción perfecta. Cuanto más cerca están los puntos de esta recta, mejor es el modelo.

### 6.3 Error por época

**Qué muestra:** La evolución del error cuadrático total (o medio) a lo largo de las épocas de entrenamiento. Una curva decreciente que se estabiliza indica convergencia.

### 6.4 Generalización

**Qué muestra:** Un gráfico de error en test (media ± desvío estándar de 15 corridas) vs. el porcentaje de datos usados para testeo. Muestra cómo la cantidad de datos de entrenamiento afecta la capacidad de generalización.

---

## 7. Resultados Esperados

| Experimento | Convergencia | Error esperado | Interpretación |
|---|---|---|---|
| **AND** (step) | Sí, ~2-10 épocas | 0 (error cero) | Las clases son linealmente separables. El hiperplano separa los convexos sin intersección. |
| **XOR** (step) | No (200 épocas máx) | > 0 | Las clases NO son linealmente separables. Los convexos se intersectan. |
| **Lineal** | Sí, ~200 épocas | MSE bajo (~1-10) | El modelo lineal captura la tendencia general de los datos. |
| **Tanh** | Sí, ~300 épocas | MSE más bajo que lineal | La no linealidad de tanh permite un mejor ajuste a la relación subyacente. |
| **Logística** | Sí, ~300 épocas | MSE similar a tanh | La activación logística ofrece una alternativa no lineal con rango (0,1). |
| **Generalización** | N/A | Disminuye con más datos | A mayor cantidad de datos de entrenamiento, menor error en test y menor varianza. |

---

## 8. Conclusiones

1. **Separabilidad lineal:** El perceptrón simple solo puede resolver problemas linealmente separables (AND sí, XOR no). Esto demuestra la necesidad de arquitecturas más complejas (multicapa) para problemas no lineales.

2. **Regresión lineal vs no lineal:** La activación lineal captura tendencias generales pero tiene limitaciones si la relación subyacente es no lineal. Las activaciones tanh y logística, al ser no lineales, permiten modelar relaciones más complejas y generalmente obtienen menor error.

3. **Importancia de la normalización:** Para activaciones con rango acotado (tanh: (−1,1), logística: (0,1)), es necesario normalizar los targets al rango de la función para que el aprendizaje sea efectivo.

4. **Generalización:** Más datos de entrenamiento producen modelos con menor error en test y menor varianza. El experimento de generalización cuantifica esta relación y muestra la importancia de tener suficientes datos para evitar underfitting.

5. **Arquitectura del código:** El TP se organiza en módulos separados (activaciones, perceptrón, datos, visualización, app) siguiendo principios de diseño modular. Esto facilita la extensión a futuros trabajos prácticos (TP2, TP3, etc.).

---

## 9. Dependencias

- **numpy**: operaciones vectoriales y álgebra lineal
- **matplotlib**: visualización de gráficos
- **scipy.spatial.ConvexHull**: cálculo de envolventes convexas para visualización de separabilidad
