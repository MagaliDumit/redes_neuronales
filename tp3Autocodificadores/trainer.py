from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class AutoencoderTrainer:
    @staticmethod
    def train(model, X_train, X_val=None, epochs=500, batch_size=8, verbose=0):
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy'
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=50,
                min_delta=1e-5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=20,
                min_lr=1e-6
            ),
        ]

        validation_data = (X_val, X_val) if X_val is not None else None

        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        return history
