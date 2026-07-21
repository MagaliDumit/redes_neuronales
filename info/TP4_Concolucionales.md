# TP4 - Redes Neuronales Convolucionales
## Clasificación de Animales (8 clases)

**Materia:** Tecnicatura en Inteligencia Artificial - Universidad Nacional de Hurlingham
**Docente:** Emiliano Churruca
**Fecha:** Julio 2026

---

## 1. Preparación y División de Datos

- **Total de imágenes:** 804
- **Train:** 560 (70 por clase, desde carpetas `entrenamiento/`)
- **Test:** 247 (~30 por clase, desde raíz de cada carpeta)
- **Formato:** Escala de grises, redimensionadas a 224×224×1, normalizadas a [0,1]
- **Codificación de salida:** One-hot (vector de 8 posiciones, un 1 en la clase correspondiente)

### Clases

| Clase       | Train | Test |
|-------------|-------|------|
| Aves        | 70    | 30   |
| Caballos    | 70    | 30   |
| Gatos       | 70    | 29   |
| Hipopótamos | 70    | 38   |
| Perros      | 70    | 30   |
| Pingüinos   | 70    | 30   |
| Serpientes  | 70    | 30   |
| Tortugas    | 70    | 30   |

### Ejemplo de codificación one-hot

Si el modelo clasifica un "Gato" (índice 2), el vector de salida es:
```
[0, 0, 1, 0, 0, 0, 0, 0]
```

---

## 2. Modelo 1: Custom CNN (Arquitectura desde cero)

### Arquitectura

| Capa                    | Detalle                              | Salida       |
|-------------------------|--------------------------------------|--------------|
| Input                   | 224×224×1                            | -            |
| Conv2D                  | 32 filtros 3×3, ReLU, padding='same' | 224×224×32   |
| BatchNormalization      | -                                    | -            |
| MaxPooling2D            | 2×2                                  | 112×112×32   |
| Conv2D                  | 64 filtros 3×3, ReLU, padding='same' | 112×112×64   |
| BatchNormalization      | -                                    | -            |
| MaxPooling2D            | 2×2                                  | 56×56×64     |
| Conv2D                  | 128 filtros 3×3, ReLU, padding='same'| 56×56×128    |
| BatchNormalization      | -                                    | -            |
| MaxPooling2D            | 2×2                                  | 28×28×128    |
| Conv2D                  | 256 filtros 3×3, ReLU, padding='same'| 28×28×256    |
| BatchNormalization      | -                                    | -            |
| MaxPooling2D            | 2×2                                  | 14×14×256    |
| GlobalAveragePooling2D  | -                                    | 256          |
| Dense                   | 128 neuronas, ReLU, L2(1e-4)         | 128          |
| Dropout                 | 50%                                  | 128          |
| Dense (salida)          | 8 neuronas, Softmax                  | 8            |

### Hiperparámetros

- **Optimizador:** Adam (learning_rate=0.001)
- **Función de pérdida:** categorical_crossentropy
- **Épocas máximas:** 100 (con EarlyStopping, paciencia=15)
- **Batch size:** 16
- **ReduceLROnPlateau:** factor=0.5, paciencia=8, lr mínimo=1e-6
- **Regularización:** L2(1e-4) en capa densa + Dropout(50%)

### Resultados en test

- **Accuracy:** 70.45%
- **Errores totales:** 73/247
- **Macro avg F1:** 0.695
- **Clases con peor rendimiento:** Perros (F1=0.51), Gatos (F1=0.41), Hipopótamos (F1=0.54)

La Custom CNN se confunde principalmente entre clases morfológicamente similares (Perros↔Gatos, Caballos↔Hipopótamos). Esto es esperable dado el volumen reducido de datos (~30 imágenes por clase para test), ya que la red no logra generalizar patrones lo suficientemente discriminativos.

---

## 3. Modelo 2: Transfer Learning (MobileNetV2)

### Estrategia

1. **Extracción de características (base congelada):** Se carga MobileNetV2 preentrenado en ImageNet con `include_top=False`. Toda la base convolucional se congela (`trainable=False`) para preservar los pesos de ImageNet.
2. **Cabeza de clasificación personalizada:** Se agrega GAP + Dense(128, ReLU) + Dropout(0.3) + Dense(8, Softmax) adaptada a nuestras 8 clases.
3. **Fine-tuning:** Se descongelan las últimas capas de MobileNetV2 (desde el índice 100 en adelante) y se re-entrena con una tasa de aprendizaje 100× menor (1e-5).

### Pipeline de preprocesamiento

```
(224,224,1) gris  ->  tf.repeat(3 canales)  ->  *255  ->  preprocess_input  ->  [-1,1]  ->  MobileNetV2
```

### Hiperparámetros

| Etapa            | Learning Rate | Épocas | Capas entrenables |
|------------------|---------------|--------|-------------------|
| Feature extraction | 0.001       | 50     | Solo la cabeza    |
| Fine-tuning      | 0.00001       | 30     | Capas > 100       |

### Resultados en test

- **Accuracy post fine-tuning:** 96.76%
- **Errores totales:** 4/247
- **Macro avg F1:** 0.969
- **Loss en validación:** 0.1829

### Comparativa de métricas por clase (F1-score)

| Clase       | Custom CNN | Transfer Learning | Diferencia |
|-------------|-----------|-------------------|------------|
| Aves        | 0.82      | 0.98              | +0.16      |
| Caballos    | 0.84      | 1.00              | +0.16      |
| Gatos       | 0.41      | 0.94              | +0.53      |
| Hipopótamos | 0.54      | 1.00              | +0.46      |
| Perros      | 0.51      | 0.98              | +0.47      |
| Pingüinos   | 0.88      | 0.88              | 0.00       |
| Serpientes  | 0.74      | 0.97              | +0.23      |
| Tortugas    | 0.76      | 1.00              | +0.24      |

### Comparativa global

| Métrica              | Custom CNN | Transfer Learning |
|----------------------|-----------|-------------------|
| Accuracy en test     | 70.45%    | 96.76%            |
| Errores totales      | 73        | 4                 |
| Macro avg F1         | 0.695     | 0.969             |
| Weighted avg F1      | 0.697     | 0.968             |

---

## 4. Código de Inferencia y Visualización

La función `Visualizer.inferencia()` (en `visualizer.py`) permite:

1. Ingresar la ruta de cualquier imagen externa
2. Mostrar la imagen original en pantalla
3. Pasar la imagen por ambos modelos (Custom CNN y Transfer Learning)
4. Mostrar barras de probabilidad para cada clase en ambos modelos
5. Indicar claramente la clase predicha y su nivel de confianza

El menú interactivo (`app.py`) ofrece 13 opciones de visualización:

| Opción | Descripción |
|--------|-------------|
| 1  | Muestras del dataset por clase |
| 2  | Curvas de entrenamiento - Custom CNN |
| 3  | Curvas de entrenamiento - Transfer Learning |
| 4  | Curvas de entrenamiento - Fine-Tuning |
| 5  | Matriz de confusión - Custom CNN |
| 6  | Matriz de confusión - Transfer Learning |
| 7  | Predicciones en test - Custom CNN |
| 8  | Predicciones en test - Transfer Learning |
| 9  | Inferencia (cargar imagen externa) |
| 10 | Comparativa Custom vs Transfer |
| 11 | Errores de clasificación - Custom CNN |
| 12 | Mapas de activación - Custom CNN |
| 13 | Mostrar TODAS las visualizaciones |

---

## 5. Historial de Cambios y Correcciones Realizadas

Durante el desarrollo del TP4 se realizaron las siguientes correcciones y mejoras sobre el código base original:

### 5.1 Corrección de normalización en Transfer Learning
**Problema:** `preprocess_input` de MobileNetV2 espera píxeles en [0, 255], pero `DataLoader` normalizaba a [0, 1]. Esto dejaba la base convolucional "ciega" generando probabilidades uniformes (~12.5%, equivalentes a azar).
**Solución:** Se agregó una capa `Lambda(img * 255.0)` antes del `preprocess_input` en `models.py` para restaurar el rango [0, 255].

### 5.2 Migración a categorical_crossentropy (one-hot)
**Problema:** El código original usaba `sparse_categorical_crossentropy` con etiquetas como índices enteros. El TP exige salida one-hot de 8 neuronas.
**Solución:** Se cambió la función de pérdida a `categorical_crossentropy` y `DataLoader` ahora genera vectores one-hot con `to_categorical()`.

### 5.3 Redimensionado de 512×512 a 224×224
**Problema:** Las imágenes originales eran 512×512, lo que hacía el entrenamiento muy lento y requería mucha memoria. MobileNetV2 además usa 224×224 por defecto.
**Solución:** Se unificó todo a 224×224 con `cv2.resize()` en el `DataLoader`.

### 5.4 Simplificación de la arquitectura Custom CNN
**Problema:** La arquitectura original tenía 6 bloques convolucionales con 512 filtros, excesivo para 560 imágenes de entrenamiento (provocaba overfitting).
**Solución:** Se redujo a 4 bloques (filtros 32→64→128→256) con GlobalAveragePooling2D y Dropout(0.5).

### 5.5 Reorganización del split train/test
**Problema:** Originalmente se usaba `train_test_split` aleatorio de sklearn. Se requería una división bien definida y reproducible.
**Solución:** Se cambió a usar carpetas `entrenamiento/` para train y raíz de cada clase para test, sin mezcla aleatoria.

### 5.6 Migración a escala de grises (1 canal)
**Problema:** El dataset contiene imágenes en escala de grises, pero el código original las convertía a RGB (3 canales), desperdiciando memoria.
**Solución:** Se leen en gris con `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`, forma (224,224,1). MobileNetV2 recibe el canal gris repetido 3 veces mediante `tf.repeat()`.

### 5.7 Guardado de modelos en formato .keras
**Problema:** Los modelos no se persistían después del entrenamiento.
**Solución:** Se agregó `model.save('modelo_custom.keras')` y `model.save('modelo_transfer.keras')` al final del entrenamiento en `app.py`.

### 5.8 Corrección visual en inferencia (barra de predicción)
**Problema:** Las barras de predicción se pintaban de rojo, color asociado intuitivamente a error.
**Solución:** Se cambió a verde (`set_color('green')`) en `visualizer.py` para indicar la clase predicha.

### 5.9 Soporte para archivos .webp
**Problema:** El dataset contenía archivos .webp que no eran detectados por el filtro de extensiones.
**Solución:** Se agregó `.webp` a `EXTENSIONES_VALIDAS` en `data.py`.

---

## 6. Discusión y Dificultades

### Dificultades encontradas

1. **Desajuste de normalización en Transfer Learning:** El principal problema técnico fue que `preprocess_input` de MobileNetV2 espera píxeles en [0, 255] pero los datos llegaban en [0, 1]. Esto anuló por completo la capacidad de la red preentrenada, generando predicciones aleatorias. Una vez corregido, el accuracy saltó de ~12% a ~97%.

2. **Dataset pequeño:** Con solo ~30 imágenes de test por clase y 70 de entrenamiento, la Custom CNN sufre de overfitting. Las curvas de entrenamiento muestran alta accuracy en train pero estancamiento en validación. El EarlyStopping y Dropout fueron cruciales para mitigarlo.

3. **Confusión entre clases similares:** La Custom CNN confunde sistemáticamente Perros con Gatos (F1=0.41-0.51) y Caballos con Hipopótamos (F1=0.54). Esto se debe a que comparten siluetas, fondos y texturas similares que la red no logra discriminar con pocos ejemplos.

4. **Formato heterogéneo de imágenes:** El dataset contenía imágenes RGB y en escala de grises, en distintos tamaños (desde 290×307 hasta 1800×1280) y formatos (.jpg, .png, .webp). Se unificó todo a grises 224×224.

5. **Espacios en nombres de carpetas:** Algunas carpetas del dataset tenían espacios al final del nombre ("Hipopótamos ", "Pingüinos "), lo que requería manejo especial en el DataLoader.

### Conclusiones

1. **Transfer Learning supera ampliamente a la arquitectura custom:** 96.76% vs 70.45% de accuracy. La ganancia es de +26 puntos porcentuales.

2. **Con datasets pequeños (<1000 imágenes) no es recomendable entrenar desde cero:** La Custom CNN carece de suficientes ejemplos para aprender características generalizables.

3. **El fine-tuning con learning rate bajo (1e-5) mejora los resultados:** Permite adaptar las características preentrenadas al dominio específico sin destruir los pesos de ImageNet (catastrophic forgetting).

4. **La correcta alineación del pipeline de preprocesamiento es crítica:** Un error en el rango de píxeles puede anular por completo el conocimiento de una red preentrenada.

5. **MobileNetV2 es una excelente opción para entornos con recursos limitados:** Es liviana, rápida y alcanza resultados comparables a redes más grandes como ResNet50 o EfficientNet.

---

## 7. Archivos del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `data.py` | Carga y preprocesamiento de imágenes (grises, 224×224, one-hot) |
| `models.py` | Arquitecturas: Custom CNN y Transfer Learning (MobileNetV2) |
| `trainer.py` | Entrenamiento con EarlyStopping, ReduceLROnPlateau y fine-tuning |
| `visualizer.py` | Visualizaciones: muestras, curvas, matrices, inferencia, mapas de activación |
| `app.py` | Menú interactivo con 13 opciones de visualización |
| `main.py` | Punto de entrada (`python3 main.py`) |
| `modelo_custom.keras` | Modelo Custom CNN entrenado (guardado después del entrenamiento) |
| `modelo_transfer.keras` | Modelo Transfer Learning entrenado (guardado después del entrenamiento) |
