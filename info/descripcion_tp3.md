# TP 3: Autocodificadores

**Emiliano Churruca** — 9 de junio de 2026
Redes Neuronales — Universidad de la Matanza

---

## Resumen del trabajo

El trabajo práctico consta de tres partes:

1. **Autoencoder de caracteres Font3** — implementar un autoencoder con espacio
   latente 2D para los 32 caracteres binarios de 7×5 del archivo `caracteres.h`,
   estudiar arquitecturas y parámetros, graficar el espacio latente y generar
   un nuevo carácter fuera del conjunto de entrenamiento.

2. **Denoising** — sobre el mismo conjunto, implementar un eliminador de ruido,
   justificar la arquitectura, distorsionar las entradas en diferentes niveles
   y evaluar la capacidad de eliminar ruido.

3. **Escenario generativo propio** — elegir un problema donde un autoencoder
   pueda generar nuevas muestras. Se eligió **MNIST** por ser el dataset
   estándar de dígitos manuscritos (60.000 imágenes de 28×28).

---

## 1. Autoencoder de caracteres Font3

### 1.1 Introducción

Un autoencoder es una red neuronal que aprende a copiar su entrada a la salida,
pasando por un **cuello de botella** (espacio latente) que fuerza una
representación comprimida. Está compuesto por dos subredes:

- **Encoder**: comprime la entrada de alta dimensión a un código de baja dimensión.
- **Decoder**: expande ese código y reconstruye la entrada original.

El dataset Font3 contiene 32 patrones binarios de 7×5 píxeles (35 bits c/u).

### 1.2 Los datos: Font3

El archivo `datos/caracteres.h` contiene tres fuentes (Font1, Font2, Font3),
cada una con 32 caracteres de 7×5 bits. Para este TP se usa únicamente **Font3**,
cuyos caracteres van desde `` ` `` (0x60) hasta `DEL` (0x7F).

Cada carácter se almacena como 7 valores hexadecimales (uno por fila),
donde cada valor usa 5 bits (LSB). El parser en `data.py`:

1. Lee el archivo y localiza el array `Font3[32][7]`.
2. Extrae solo los valores hexadecimales del código (ignora comentarios con `//`).
3. Convierte cada hex a 5 bits y los concatenan en un vector de 35 floats (0/1).
4. Asigna las etiquetas `` ` `` `a` `b` ... `z` `{` `|` `}` `~` `DEL`.

```python
# data.py — FontLoader.parse()
# Por cada hex value, extrae 5 bits (bit 4 a bit 0)
def _hex_to_bits(self, val):
    bits = []
    for bit in range(4, -1, -1):
        bits.append((val >> bit) & 1)
    return bits
```

El resultado son **32 vectores** de 35 componentes cada uno, con valores
binarios (0.0 o 1.0).

---

### 1.3 Arquitectura del autoencoder

#### 1.3.1 Diseño

La arquitectura elegida fue `35 → 17 → 7 → 2 → 7 → 17 → 35`, propuesta
por la cátedra:

```
Capas          Activación   Parámetros
————————————————————————————————————————
Entrada (35)      —            —
Dense(17)        relu       35×17 + 17 = 612
Dense(7)         relu       17×7  + 7  = 126
Latente (2)      tanh        7×2  + 2  = 16
Dense(7)         relu        2×7  + 7  = 21
Dense(17)        relu        7×17 + 17 = 136
Salida (35)      sigmoid    17×35 + 35 = 630
                         Total: ~1541 parámetros
```

**Decisiones de diseño:**

| Capa | Activación | Motivo |
|------|-----------|--------|
| Ocultas encoder/decoder | **relu** | No satura para valores positivos, gradientes no se desvanecen |
| Latente | **tanh** | Acota la salida a `[-1, 1]`, genera código simétrico, evita deriva |
| Salida | **sigmoid** | Acota la salida a `(0, 1)`, ideal para datos binarios |

**Simetría**: encoder y decoder son simétricos (17→7→2 vs 2→7→17).
Esto es una práctica común porque la tarea inversa tiene complejidad similar
a la directa.

**Cuello de botella**: con solo 2 dimensiones latentes para 35 bits de entrada,
el encoder debe aprender a preservar la estructura esencial. Para 32 patrones
binarios, 2 dimensiones alcanzan porque cada carácter puede ocupar un punto
distinto en el plano.

#### 1.3.2 Entrenamiento

```python
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
```

**Loss**: `binary_crossentropy` es la elección natural para datos binarios
con salida sigmoide. Para un target `t ∈ {0,1}` y predicción `p ∈ (0,1)`:

```
BCE = -t·log(p) - (1-t)·log(1-p)
```

Penaliza fuertemente las predicciones seguras pero incorrectas
(e.g., `p=0.99` cuando `t=0`).

##### 1.3.2.1 El problema del split train/test

Inicialmente se usó un split 80/20 (25 train, 7 test). Con solo 32 muestras,
esto fue un **error grave**: el modelo no lograba converger.

```
Split 80/20 (25 train, 7 test):
  115 épocas, loss=0.51, val_loss=0.64
  Precisión global: 66.79%
  Caracteres perfectos: 0/32
```

El modelo no tenía suficientes datos para aprender, y la validación en 7 muestras
era ruidosa. La solución fue **entrenar sobre las 32 muestras completas**:

```
Sin split (32 train, 0 test):
  10000 épocas, loss=0.0019
  Precisión global: 1120/1120 (100.00%)
  Caracteres perfectos: 32/32
```

Al no necesitar generalizar a caracteres nuevos (solo hay 32 en el dataset),
el autoencoder puede **memorizar** los patrones, lo cual es perfectamente
válido en este contexto.

##### 1.3.2.2 Hiperparámetros

- **Epochs**: 10.000 (con ReduceLROnPlateau, paciencia 200)
- **Batch size**: 32 (todos los datos en cada paso)
- **Learning rate**: 0.001 inicial, reduce a 0.5× cada 200 épocas sin mejora
- **Optimizer**: Adam

El LR se reduce automáticamente cuando el loss se estanca. 10.000 épocas
son necesarias para que la loss baje de ~0.69 a ~0.0019.
Cada época es rápida (~10 ms) porque solo hay 32 muestras.

---

## 2. Denoising (eliminador de ruido)

### 2.1 Justificación de la arquitectura

El denoising se implementa usando el **mismo AE básico**, no un modelo separado.
Esto es intencional: se evalúa la capacidad del cuello de botella latente
para filtrar ruido.

### 2.2 Funcionamiento

```python
noisy = DataProcessor.add_noise(test_chars, pct)
pred = ae.predict(noisy)
```

`add_noise()` toma cada bit y lo invierte con probabilidad `pct`:

```python
@staticmethod
def add_noise(X, noise_level):
    noise = np.random.binomial(1, noise_level, size=X.shape)
    return np.where(noise == 1, 1.0 - X, X)
```

### 2.3 Evaluación en consola

Para el primer carácter se muestra:

1. **Matriz original** (7×5, bits 0/1).
2. **Matriz con ruido**: bits alterados en **amarillo** (ANSI `\033[93m`).
3. **Matriz reconstruida**: bits tras redondear la salida sigmoide.
4. **Exactitud total**: aciertos sobre total de bits (5 caracteres × 35 bits).

Ejemplo con 30% de ruido:

```
=== Denoising: 30% ===
  Primer carácter: [|]
  Bits alterados: 10/35 (30%)
  Original:
    0 0 1 0 0
    0 0 1 0 0
    0 0 1 0 0
    0 0 0 0 0
    0 0 1 0 0
    0 0 1 0 0
    0 0 1 0 0

  Con ruido:
    1 1 1 0 1      ← bits en amarillo
    0 0 0 0 0
    0 0 1 0 0
    ...

  Reconstruido:
    0 0 1 0 0
    0 0 1 0 0
    ...

  Exactitud total (5 chars): 112/175 (64.0%)
```

### 2.4 Ventana gráfica

Además de la consola, se muestra una figura con 5 caracteres
(original, ruidoso, reconstruido) para inspección visual.

---

### 1.4 Espacio latente interactivo (Opción 2 del menú)

La opción 2 abre una ventana interactiva con:

- **Scatter plot 2D** de los 32 caracteres en el espacio latente.
  Cada punto está etiquetado con su carácter.
- **Click en un punto**: selecciona el carácter, muestra su matriz 7×5.
- **Click en otro punto**: interpola linealmente entre ambos en el espacio
  latente, genera 5 pasos (α = 0, 0.25, 0.5, 0.75, 1.0), los muestra
  concatenados.
- **Click en espacio vacío**: genera un nuevo carácter desde los 3 vecinos
  más cercanos (promedio ponderado por distancia inversa).

#### 1.4.1 Interpolación

```python
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
z_all = alphas[:, None] * code_a[None, :] + (1 - alphas[:, None]) * code_b[None, :]
gen_batch = np.round(decoder.predict(z_all))
```

La interpolación muestra la transición suave entre dos caracteres,
demostrando que el espacio latente es **continuo** y **significativo**
(puntos cercanos producen caracteres similares).

#### 1.4.2 Generación por cercanía

Cuando se clickea en una zona sin caracteres, se toman los k=3 vecinos
más cercanos y se promedia su código latente ponderado por distancia:

```python
weights = 1.0 / (dists[nearest_k] + 1e-10)
weights /= weights.sum()
z_new = np.sum(codes[nearest_k] * weights[:, None], axis=0)
```

Esto genera un carácter "nuevo" que combina rasgos de los vecinos.

---

### 1.5 Interpolación y ruido en el espacio latente (Opciones 5 y 6)

#### 1.5.1 Interpolación lineal (Opción 5)

Se seleccionan 2 caracteres aleatorios de Font3. Se obtienen sus códigos
latentes y se interpola con 5 valores α. Al descodificar cada paso,
se observa una **transición morfológica** entre los dos caracteres.

Por ejemplo, una interpolación `[a] → [z]` muestra cómo el autoencoder
transiciona suavemente de una forma a otra, generando caracteres
intermedios que parecen mezclas híbridas.

#### 1.5.2 Ruido gaussiano en latente (Opción 6)

Se selecciona un carácter, se obtiene su código latente `z`, y se añade
ruido gaussiano con desviaciones crecientes: ε = {0.0, 0.5, 1.0, 1.5, 2.0}.

```python
z_noisy = z + np.random.normal(0, eps, size=z.shape)
```

A medida que ε aumenta, el código latente se aleja del original y el
carácter se degrada progresivamente, mostrando cómo la vecindad del
espacio latente codifica variaciones del mismo patrón.

---

### 1.6 Generación de un nuevo carácter (Opción 7)

Para cumplir con el requisito de generar un carácter que no pertenece al
conjunto de entrenamiento, se muestrean puntos aleatorios en el espacio
latente y se descodifican. Como el espacio latente es continuo y cubre
`[-1, 1]²`, cualquier punto en esa región produce un patrón de 35 bits.
Algunos coinciden con caracteres reales, otros son **combinaciones
novedosas** de trazos.

Se muestrean 8 puntos uniformes en `[-1, 1]²` y se descodifican:

```python
z = np.random.uniform(-1, 1, size=2)
gen = np.round(decoder.predict(z[None, :]))
```

Esto genera **caracteres completamente nuevos**, que nunca estuvieron
en el dataset. Visualmente suelen combinar trazos de varios caracteres
reales del Font3, demostrando que el autoencoder aprendió una
representación generativa del espacio de caracteres.

---

## 3. Autoencoder MNIST (escenario generativo propio)

### 3.1 Elección del escenario

Se eligió **MNIST** (60.000 dígitos manuscritos 28×28 en escala de grises)
como escenario generativo porque:
- Es un dataset estándar, permitiendo comparar resultados con la literatura.
- Tiene suficiente complejidad (784 dimensiones → 2 latentes) para que el
  problema de colapso latente sea relevante y didáctico.
- La generación de nuevos dígitos desde el espacio latente tiene aplicaciones
  concretas (aumento de datos, síntesis de caracteres).

### 3.2 Datos

MNIST se carga directamente desde `tensorflow.keras.datasets.mnist`.
Las imágenes se normalizan dividiendo por 255.0 para quedar en `[0, 1]`,
y se aplanan a vectores de 784 componentes.

### 3.3 Primera arquitectura (fallida)

Inicialmente se usó:

```
784 → 32 (relu) → 2 (tanh) → 32 (relu) → 784 (sigmoid)
```

**Resultado: colapso total.** Todas las reconstrucciones eran idénticas
(una imagen promedio borrosa, parecida a un 8 o 9 difuso):

```python
val_loss = 0.2627
Unique reconstructions: 1/100   # ← todas iguales
MAE = 0.147
Correlación media: ~0.45
```

### 3.4 Causa raíz: saturación del tanh latente

Investigando, se encontró que el encoder estaba colapsando todas las
entradas al **mismo código latente**. Con 200 muestras y pesos inicializados
al azar, el encoder producía códigos diversos. Pero tras 50 épocas de
entrenamiento, todas las entradas convergían a:

```
z = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]   # para latente 8
```

El tanh estaba completamente **saturado** en ±1. Cuando `tanh(x)` opera
con `|x| > 2`, su derivada es ~0, y el gradiente no fluye hacia el encoder.
El decoder, al recibir siempre el mismo código, aprendió a producir la
**imagen promedio** de MNIST.

```
Mecanismo del colapso:
1. Pesos del encoder crecen → activaciones grandes
2. tanh(>2) = ±1 → gradiente ~0
3. Encoder deja de aprender (gradientes muertos)
4. Decoder recibe código constante
5. Decoder aprende a reconstruir la media
```

### 3.5 Solución: BatchNormalization

Se agregó una capa `BatchNormalization` justo **antes** del `tanh` latente:

```
784 → 256 (relu) → 64 (relu) → BN → 2 (tanh) → 64 (relu) → 256 (relu) → 784 (sigmoid)
```

BN normaliza las activaciones a media 0 y varianza 1 antes de pasarlas
por el tanh:

```python
x = layers.Dense(64, activation='relu')(encoder_input)
x = layers.BatchNormalization()(x)       # ← clave
latent = layers.Dense(2, activation='tanh')(x)
```

Esto mantiene las entradas del tanh en un rango donde la derivada es
significativa, permitiendo que el gradiente fluya hacia atrás.

### 3.6 Resultados tras la corrección

```python
val_loss = 0.1680        # ↓ desde 0.2627
Unique reconstructions: 100/100   # ↑ desde 1/100
MAE = 0.084              # ↓ desde 0.147
Correlación media: ~0.83 # ↑ desde ~0.45
```

| Métrica | Sin BN (colapsado) | Con BN |
|---------|-------------------|--------|
| val_loss | 0.2627 | 0.1680 |
| Reconstrucciones únicas | 1/100 | 100/100 |
| MAE | 0.147 | 0.084 |
| Correlación media | ~0.45 | ~0.83 |
| z std | 0.968 (saturado) | 0.506 (distribuido) |
| Rango z | [-1.000, 1.000] | [-0.997, 0.996] |

### 3.7 Entrenamiento

```python
epochs = 300
batch_size = 128
patience = 80   # EarlyStopping
loss = 'binary_crossentropy'
optimizer = Adam(0.001)
```

300 épocas con EarlyStopping (paciencia 80) y ReduceLROnPlateau
(paciencia 30, factor 0.5). El entrenamiento se detuvo en 130 épocas
porque el val_loss dejó de mejorar.

---

## 4. El menú del programa

El programa presenta 9 opciones numeradas consecutivamente:

```
  1. Reconstrucciones básicas (Caracteres)
  2. Mostrar Espacio Latente 2D (Caracteres)
  3. Reconstruir Carácter Externo
  4. Denoising con Porcentaje de Ruido Personalizado
  5. Interpolación Lineal (Morphing)
  6. Generar con Ruido Gaussiano en Latente
  7. Muestreo Uniforme / Generación Aleatoria
  8. Ejecutar AE Generativo MNIST
  9. Mostrar todas las ventanas
  0. Salir
```

### 4.1 Flujo de ejecución

1. `main.py` crea una instancia de `MenuApp` y llama a `run()`.
2. `run()` ejecuta `setup()` que carga datos y entrena/carga modelos.
3. Entra en un bucle mostrando el menú y ejecutando la opción elegida.
4. Al salir, cierra todas las figuras de matplotlib.

### 4.2 Caché de modelos

Los modelos entrenados se guardan en `modelos/` como archivos `.keras`:
- `ae.keras`: autoencoder de caracteres (se reentrena si no existe).
- `mnist.keras`: autoencoder MNIST (ídem).

Al iniciar, si el archivo existe, se carga directamente sin reentrenar.
Esto ahorra horas de entrenamiento en ejecuciones sucesivas.

---

## 5. Dificultades encontradas y lecciones aprendidas

### 5.1 Split train/test en datasets pequeños

Con 32 muestras, dividir en train (25) y test (7) fue contraproducente.
El autoencoder no necesita generalizar a caracteres nuevos; alcanza con
memorizar los 32. Para datasets extremadamente pequeños, a veces es mejor
no separar.

### 5.2 El colapso latente y BatchNormalization

El colapso del autoencoder MNIST fue causado por la saturación del tanh.
BN es una solución elegante porque:
- Normaliza las activaciones **dinámicamente** durante el entrenamiento.
- Mantiene las entradas del tanh cerca de 0 donde la derivada es máxima.
- No agrega muchos parámetros (solo 2 por neurona: escala y desplazamiento).

Sin BN, el encoder MNIST es incapaz de aprender representaciones
diferenciadas. No importa cuán grande sea la capa oculta o el latente,
si el tanh se satura, el gradiente muere y el modelo colapsa.

### 5.3 Tiempo de entrenamiento

El autoencoder de caracteres necesitó 10.000 épocas para converger a
loss ~0.0019. Aunque son muchas épocas, cada una es rápida (~10 ms)
porque el dataset tiene solo 32 muestras. MNIST, con 60.000 imágenes,
requiere ~7 s/época y converge en ~130 épocas.

### 5.4 BCE vs MSE para datos binarios

`binary_crossentropy` es superior a `mean_squared_error` para datos
binarios con salida sigmoide. BCE penaliza errores cerca de 0 o 1 más
que MSE, lo que empuja las predicciones hacia los extremos correctos.

### 5.5 El espacio latente es continuo y significativo

La interpolación entre caracteres muestra transiciones suaves,
confirmando que el espacio latente no es un mero índice de memorización.
Puntos cercanos en latente producen caracteres visualmente similares,
lo que permite generar nuevos caracteres por muestreo aleatorio.

---

## 6. Conclusiones

El TP3 demostró la construcción exitosa de dos autoencoders:

1. **Autoencoder de caracteres Font3**: arquitectura `35→17→7→2→7→17→35`
   que logra reconstrucción bit-perfect (100%, 32/32 caracteres) tras
   10.000 épocas sobre el dataset completo. Permite denoising dinámico,
   interpolación morfológica y generación de nuevos caracteres.

2. **Autoencoder MNIST**: arquitectura `784→256→64→BN→2→64→256→784`
   que requirió BatchNormalization para evitar el colapso latente.
   Logra reconstrucciones diversas (100/100 únicas) con MAE de 0.084.

El principal descubrimiento fue la **saturación del tanh latente** como
causa del colapso del autoencoder, y su solución mediante
BatchNormalization, que resultó ser crítica para que el encoder aprenda
representaciones diferenciadas en el espacio latente.
