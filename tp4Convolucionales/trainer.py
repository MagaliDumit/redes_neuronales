import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np


class Trainer:
    """
    Maneja el entrenamiento, fine-tuning, evaluación y predicción de los modelos.

    EarlyStopping: detiene el entrenamiento si la métrica de validación
    no mejora después de 'patience' épocas. Restaura los mejores pesos.

    ReduceLROnPlateau: reduce la tasa de aprendizaje cuando la pérdida
    se estanca, permitiendo que el optimizador encuentre mínimos más finos.
    """

    @staticmethod
    def train(model, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=16, verbose=1):
        """
        Entrenamiento principal del modelo con EarlyStopping y reducción de LR.
        y_train, y_val están en formato one-hot (8 columnas).
        """
        cb = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=verbose
        )
        return history

    @staticmethod
    def fine_tune(model, base_model, X_train, y_train, X_val, y_val,
                  unfreeze_from=100, epochs=30, batch_size=16, verbose=1):
        """
        Aplica fine-tuning descongelando parcialmente la base convolucional.

        Se mantienen congeladas las primeras 'unfreeze_from' capas y se
        descongelan las últimas. Se recompila con un learning rate muy bajo
        (1e-5) para no destruir los pesos preentrenados de ImageNet.

        El fine-tuning permite que el modelo adapte sus características
        más específicas (texturas, formas) a nuestro dataset de animales.
        """
        # Descongelar la base
        base_model.trainable = True

        # Congelar capas tempranas (características genéricas: bordes, colores)
        # Descongelar capas tardías (características específicas: formas, objetos)
        for layer in base_model.layers[:unfreeze_from]:
            layer.trainable = False

        # Recompilar con learning rate bajo para fine-tuning fino
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        cb = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=verbose
        )
        return history

    @staticmethod
    def evaluate(model, X_test, y_test):
        """Evalúa el modelo en el conjunto de test. Retorna (loss, accuracy)."""
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    @staticmethod
    def predict(model, X):
        """
        Genera predicciones para un conjunto de datos.
        Retorna (pred_classes, probabilities):
          pred_classes: array (N,) con índices de clase predichos
          probabilities: array (N, 8) con probabilidades de cada clase
        """
        probs = model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs
