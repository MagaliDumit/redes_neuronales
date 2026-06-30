# Plan de interfaz unificada para TP1.py

## Estructura

### Menú principal (consola)
```
============================================
   TP1 - PERCEPTRÓN SIMPLE
============================================
Seleccioná una opción:

  [1] AND — Convexos e hiperplano separador
  [2] XOR — Convexos intersectados
  [3] Lineal — Error cuadrático por época
  [4] Lineal — Predicción vs Real (Train)
  [5] Lineal — Predicción vs Real (Test)
  [6] Tanh — Error cuadrático por época
  [7] Tanh — Predicción vs Real (Train)
  [8] Tanh — Predicción vs Real (Test)
  [9] Generalización — Error vs tamaño de entrenamiento
 [10] Mostrar todas las ventanas
  [0] Salir

============================================
Opción:
```

### Cada ventana (matplotlib)
```
+-------------------------------------------+
|  TÍTULO                                    |
|  +-------------------+-------------------+ |
|  |                   | Algoritmo:         | |
|  |                   | Perceptrón escalón | |
|  |    GRÁFICO        |--------------------| |
|  |                   | Fórmula:           | |
|  |                   | O = θ(w·ξ − w₀)    | |
|  |                   | Δw = η(ζ−O)ξ       | |
|  |                   |--------------------| |
|  |                   | Desc: AND es       | |
|  |                   | linealmente sep.   | |
|  +-------------------+-------------------+ |
+-------------------------------------------+
```
- `fig.subplots(1, 2, width_ratios=[3, 2])`
- Subplot izquierdo: gráfico correspondiente
- Subplot derecho: `ax.axis('off')` + `ax.text()` con descripción

---

## Contenido de cada descripción

| Opción | Título | Algoritmo | Ecuación | Descripción |
|---|---|---|---|---|
| 1 AND | AND - Convexos e hiperplano | Perceptrón escalón | `O = θ(w·ξ − w₀)` `Δw = η(ζ−O)ξ` | AND es linealmente separable: existe un hiperplano que separa ambas clases |
| 2 XOR | XOR - Convexos intersectados | Perceptrón escalón | `O = θ(w·ξ − w₀)` `Δw = η(ζ−O)ξ` | XOR no es linealmente separable: los convexos se intersectan, no hay hiperplano que separe las clases |
| 3 | Lineal - Error por época | Perceptrón lineal | `O = w·ξ` `E = ½Σ(ζ−O)²` | Evolución del error cuadrático medio durante las épocas de entrenamiento |
| 4 | Lineal - Predicción vs Real (Train) | Perceptrón lineal | `O = w·ξ` | Comparación entre el valor real y la predicción del modelo lineal en el conjunto de entrenamiento |
| 5 | Lineal - Predicción vs Real (Test) | Perceptrón lineal | `O = w·ξ` | Comparación entre el valor real y la predicción del modelo lineal en el conjunto de testeo |
| 6 | Tanh - Error por época | Perceptrón no lineal | `O = tanh(βh)` `g'(h) = β(1−tanh²(h))` `Δw = η(ζ−O)·g'(h)·ξ` | Evolución del error durante entrenamiento con activación tangente hiperbólica |
| 7 | Tanh - Predicción vs Real (Train) | Perceptrón no lineal | `O = tanh(βh)` con β=1 | Salidas normalizadas a [-0.9, 0.9]; se desnormalizan para mostrar predicción vs real |
| 8 | Tanh - Predicción vs Real (Test) | Perceptrón no lineal | `O = tanh(βh)` con β=1 | Idem en conjunto de testeo |
| 9 | Generalización - Error vs tamaño | Perceptrón lineal | Evaluación con splits 10% a 50%, 15 corridas c/u | Error en test versus cantidad de datos usados para entrenar |

---

## Cambios en el código

1. Agregar función `show_plot_with_info(fig, title, algorithm, equation, description)` que recibe la figura y renderiza el panel derecho
2. Cada plot existente se modifica para recibir `ax_left` en vez de crear su propia figura
3. `if __name__ == "__main__"` se reemplaza por bucle con menú y carga única de datos
4. Cada opción llama a su función correspondiente que construye la ventana dividida
5. Opción 10 ejecuta todas en secuencia (con pausa entre cada una)
