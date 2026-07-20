import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_custom_cnn(input_shape=(224, 224, 1), num_classes=8):
    """
    Modelo 1: Arquitectura Custom (desde cero).

    Red convolucional diseñada específicamente para este TP.
    Arquitectura:
      - 4 bloques convolucionales (Conv2D + BatchNormalization + MaxPooling)
        con filtros crecientes: 32 -> 64 -> 128 -> 256
      - GlobalAveragePooling2D para reducir parámetros y evitar overfitting
      - Capa densa intermedia con regularización L2
      - Dropout para evitar sobreajuste
      - Salida softmax con 8 neuronas (una por clase)

    Hiperparámetros:
      - Filtros: 32, 64, 128, 256 (se duplican en cada bloque)
      - Kernel: 3x3 con padding 'same'
      - Activación: ReLU en capas ocultas, Softmax en salida
      - Regularización: L2 (1e-4) en capa densa
      - Dropout: 50%
    """
    model = models.Sequential([
        # Capa de entrada explícita
        layers.Input(shape=input_shape),

        # Bloque 1: 32 filtros
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 2: 64 filtros
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 3: 128 filtros
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque 4: 256 filtros
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Cabeza clasificadora
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        # 8 neuronas con softmax para clasificación one-hot
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_transfer_model(input_shape=(224, 224, 1), num_classes=8):
    """
    Modelo 2: Transfer Learning con MobileNetV2.

    MobileNetV2 espera entradas de 3 canales (RGB), pero nuestro dataset
    está en escala de grises (1 canal). Por eso repetimos el canal gris
    3 veces con tf.repeat para alimentar la red preentrenada.

    Estrategia:
      1. Se congela la base convolucional (feature extractor)
      2. Se agrega una nueva cabeza de clasificación para 8 clases
      3. Opcionalmente se aplica fine-tuning descongelando las últimas capas
    """
    # MobileNetV2 preentrenado en ImageNet, espera 3 canales
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Congelar la base
    base_model.trainable = False

    # Construir el modelo completo
    inputs = layers.Input(shape=input_shape)  # (224, 224, 1)

    # Repetir el canal gris 3 veces: (224, 224, 1) -> (224, 224, 3)
    x = layers.Lambda(lambda img: tf.repeat(img, 3, axis=-1))(inputs)

    # Escalar de [0, 1] a [0, 255] como espera preprocess_input
    x = layers.Lambda(lambda img: img * 255.0)(x)

    # preprocess_input: [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Pasar por la base convolucional congelada
    x = base_model(x, training=False)

    # Cabeza de clasificación para 8 clases
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, lr=0.001):
    """
    Configura el optimizador, la función de pérdida y las métricas.

    categorical_crossentropy: función de pérdida para clasificación one-hot.
    La salida softmax genera probabilidades para cada clase, y la pérdida
    compara el vector one-hot real con el vector de probabilidades predicho.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
