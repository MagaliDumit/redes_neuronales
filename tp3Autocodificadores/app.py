import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from .data import FontLoader, DataProcessor
    from .models import build_autoencoder, build_mnist_autoencoder
    from .trainer import AutoencoderTrainer
    from .visualizer import Visualizer
except ImportError:
    from data import FontLoader, DataProcessor
    from models import build_autoencoder, build_mnist_autoencoder
    from trainer import AutoencoderTrainer
    from visualizer import Visualizer

from tensorflow.keras.models import load_model


class MenuApp:
    MODEL_DIR = 'modelos'

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.loader = None
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.X_all = None
        self.mnist_ae = None
        self.mnist_encoder = None
        self.mnist_decoder = None
        self.mnist_data = None

    def _model_path(self, name):
        return os.path.join(self.MODEL_DIR, name)

    def _load_or_train_ae(self):
        path = self._model_path('ae.keras')
        if os.path.exists(path):
            print("  ae.keras encontrado, cargando...")
            self.ae = load_model(path)
            self.encoder = self.ae.get_layer('encoder')
            self.decoder = self.ae.get_layer('decoder')
            return
        print("Entrenando autoencoder (todos los 32 chars)...")
        self.ae, self.encoder, self.decoder = build_autoencoder()
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        self.ae.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
        h = self.ae.fit(
            self.X_all, self.X_all,
            epochs=10000, batch_size=32, verbose=0,
            callbacks=[
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-7),
            ]
        )
        epochs_done = len(h.history['loss'])
        loss = h.history['loss'][-1]
        print(f"  {epochs_done} épocas, loss={loss:.4f}")
        self.ae.save(path)
        # restore best weights after training
        self.ae = load_model(path)
        self.encoder = self.ae.get_layer('encoder')
        self.decoder = self.ae.get_layer('decoder')

    def _load_or_train_mnist(self):
        path = self._model_path('mnist.keras')
        if os.path.exists(path):
            print("  mnist.keras encontrado, cargando...")
            self.mnist_ae = load_model(path)
            self.mnist_encoder = self.mnist_ae.get_layer('encoder')
            self.mnist_decoder = self.mnist_ae.get_layer('decoder')
            try:
                from tensorflow.keras.datasets import mnist
                (_, _), (x_test, _) = mnist.load_data()
                x_test = x_test.astype(np.float32) / 255.0
                x_test = x_test.reshape(-1, 784)
                self.mnist_data = {'X_train': None, 'X_test': x_test}
            except Exception:
                self.mnist_data = {'X_train': None, 'X_test': None}
            return
        print("Entrenando autoencoder MNIST...")
        try:
            from tensorflow.keras.datasets import mnist
            (x_train, _), (x_test, _) = mnist.load_data()
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0
            x_train = x_train.reshape(-1, 784)
            x_test = x_test.reshape(-1, 784)
            self.mnist_data = {'X_train': x_train, 'X_test': x_test}

            self.mnist_ae, self.mnist_encoder, self.mnist_decoder = \
                build_mnist_autoencoder()
            from tensorflow.keras.optimizers import Adam
            self.mnist_ae.compile(
                optimizer=Adam(0.001), loss='binary_crossentropy'
            )
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            h = self.mnist_ae.fit(
                x_train, x_train,
                epochs=300, batch_size=128, verbose=0,
                validation_data=(x_test, x_test),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=80, min_delta=1e-5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6),
                ]
            )
            me = len(h.history['loss'])
            loss_val = h.history['val_loss'][-1]
            print(f"  {me} épocas, val_loss={loss_val:.4f}")
            self.mnist_ae.save(path)
        except Exception as e:
            print(f"  Error cargando MNIST: {e}")
            self.mnist_ae = None

    def _accuracy(self, pred, target):
        bin_pred = np.round(pred)
        return int(np.sum(bin_pred == target))

    def _print_matrix(self, title, vector, flips=None):
        img = vector.reshape(7, 5)
        print(f"  {title}")
        for r in range(7):
            row = ''
            for c in range(5):
                idx = r * 5 + c
                if flips is not None and idx in flips:
                    row += f"\033[93m{int(img[r, c])}\033[0m "
                else:
                    row += f"{int(img[r, c])} "
            print(f"    {row}")

    def setup(self):
        print("Cargando datos Font3...")
        self.loader = FontLoader('datos/caracteres.h').parse()
        self.X_all = self.loader.chars
        print(f"  {self.X_all.shape[0]} caracteres cargados.")

        models_info = [
            ('AE caracteres', self._load_or_train_ae),
            ('MNIST', self._load_or_train_mnist),
        ]
        for name, fn in models_info:
            print(f"\n{name}:")
            fn()
        print("\n¡Todos los modelos listos!\n")

    def _menu(self):
        print("=" * 60)
        print("   TP3 — AUTOCODIFICADORES (Keras/TensorFlow)")
        print("   Font3 — Arquitectura 35→17→7→2→7→17→35")
        print("=" * 60)
        print()
        print("Seleccioná una opción:")
        print()
        print("  1. Reconstrucciones básicas (Caracteres)")
        print("  2. Mostrar Espacio Latente 2D (Caracteres)")
        print("  3. Reconstruir Carácter Externo")
        print("  4. Denoising con Porcentaje de Ruido Personalizado (Dinámico)")
        print("  5. Interpolación Lineal (Morphing)")
        print("  6. Generar con Ruido Gaussiano en Latente")
        print("  7. Muestreo Uniforme / Generación Aleatoria")
        print("  8. Ejecutar AE Generativo MNIST y Guardar Modelos")
        print("  9. Mostrar todas las ventanas")
        print("  0. Salir")
        print()
        return input("Opción: ").strip()

    def _handle_option(self, opt):
        if opt == '0':
            return False

        elif opt == '1':
            X_all = self.X_all
            pred = self.ae.predict(X_all, verbose=0)
            n_show = min(8, len(X_all))
            show_idx = np.random.choice(len(X_all), n_show, replace=False)
            labels = [self.loader.get_label(i) for i in show_idx]
            Visualizer.reconstructions(
                X_all[show_idx], pred[show_idx], labels
            )
            acc = self._accuracy(pred, X_all)
            total = X_all.shape[0] * 35
            print(f"  Precisión global: {acc}/{total} ({100*acc/total:.2f}%)")
            per_char = np.sum(np.round(pred) == X_all, axis=1)
            perfect = np.sum(per_char == 35)
            print(f"  Caracteres perfectos: {perfect}/{X_all.shape[0]}")

        elif opt == '2':
            X_all = self.loader.chars
            codes = self.encoder.predict(X_all, verbose=0)
            labels = self.loader.labels
            Visualizer.interactive_latent_space(
                codes, labels, X_all, self.decoder
            )

        elif opt == '3':
            ext_char = np.random.randint(0, 2, size=(1, 35)).astype(np.float32)
            pred = self.ae.predict(ext_char, verbose=0)
            Visualizer.reconstructions(ext_char, pred, ['Externo'])
            acc = self._accuracy(pred, ext_char)
            print(f"  Precisión bits: {acc}/35 ({100*acc/35:.2f}%)")

        elif opt == '4':
            inp = input("Ingresá porcentaje de ruido (0–100): ").strip()
            try:
                pct = float(inp)
            except ValueError:
                print("  Número inválido.")
                return True
            pct = max(0, min(100, pct)) / 100.0

            X_all = self.X_all
            n_show = min(5, len(X_all))
            show_idx = np.random.choice(len(X_all), n_show, replace=False)
            labels = [self.loader.get_label(i) for i in show_idx]
            noisy = DataProcessor.add_noise(X_all[show_idx].copy(), pct)
            pred = self.ae.predict(noisy, verbose=0)
            bin_pred = np.round(pred)

            total_bits = n_show * 35
            aciertos = int(np.sum(bin_pred == X_all[show_idx]))
            exactitud = aciertos / total_bits * 100

            idx0 = show_idx[0]
            orig0 = X_all[idx0]
            lab0 = self.loader.get_label(idx0)
            noisy0 = noisy[0]
            flip_idx0 = set(np.where(noisy0 != orig0)[0])

            print(f"\n=== Denoising: {int(pct*100)}% ===")
            print(f"  Primer carácter: [{lab0}]")
            print(f"  Bits alterados: {len(flip_idx0)}/35 ({int(pct*100)}%)")
            self._print_matrix("Original:", orig0)
            print()
            self._print_matrix(f"Con ruido:", noisy0, flip_idx0)
            print()
            self._print_matrix("Reconstruido:", np.round(pred[0]))
            print(f"\n  Exactitud total ({n_show} chars): {aciertos}/{total_bits} ({exactitud:.1f}%)")

            Visualizer.denoising_grid(X_all[show_idx], noisy, bin_pred, pct)

        elif opt == '5':
            X_all = self.X_all
            idx_a = np.random.choice(len(X_all))
            idx_b = np.random.choice(len(X_all))
            label_a = self.loader.get_label(idx_a)
            label_b = self.loader.get_label(idx_b)
            code_a = self.encoder.predict(X_all[idx_a][np.newaxis, :], verbose=0)[0]
            code_b = self.encoder.predict(X_all[idx_b][np.newaxis, :], verbose=0)[0]

            alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            z_all = (alphas[:, np.newaxis] * code_a[np.newaxis, :]
                     + (1.0 - alphas[:, np.newaxis]) * code_b[np.newaxis, :])
            gen = np.round(self.decoder.predict(z_all, verbose=0))

            Visualizer.interpolation_chars(
                [X_all[idx_a], X_all[idx_b]],
                alphas, gen, label_a, label_b
            )

        elif opt == '6':
            X_all = self.X_all
            idx = np.random.choice(len(X_all))
            original = X_all[idx]
            label = self.loader.get_label(idx)
            code = self.encoder.predict(original[np.newaxis, :], verbose=0)[0]

            noises = [0.0, 0.5, 1.0, 1.5, 2.0]
            generated = []
            for eps in noises:
                z_noisy = code + np.random.normal(0, eps, size=code.shape)
                gen = np.round(self.decoder.predict(z_noisy[np.newaxis, :], verbose=0)[0])
                generated.append(gen)

            Visualizer.latent_noise(original, code, noises, generated)

        elif opt == '7':
            n_samples = 8
            samples = []
            for _ in range(n_samples):
                z = np.random.uniform(-1, 1, size=2)
                gen = np.round(self.decoder.predict(z[np.newaxis, :], verbose=0)[0])
                samples.append(gen)
            samples = np.array(samples)
            Visualizer.generated_grid(samples, n_samples)

        elif opt == '8':
            if self.mnist_ae is None:
                print("Modelo MNIST no disponible.")
                return True
            idx = np.random.choice(len(self.mnist_data['X_test']), 10, replace=False)
            imgs = self.mnist_data['X_test'][idx]
            preds = self.mnist_ae.predict(imgs, verbose=0)

            novel = []
            for _ in range(5):
                z = np.random.uniform(-1, 1, size=2)
                novel.append(self.mnist_decoder.predict(z[np.newaxis, :], verbose=0)[0])
            novel = np.array(novel)

            Visualizer.mnist_reconstructions(imgs, preds, novel)

        elif opt == '9':
            for i in range(1, 9):
                print(f"Mostrando opción {i}...")
                self._handle_option(str(i))

        else:
            print("Opción no válida.")

        return True

    def run(self):
        self.setup()
        running = True
        while running:
            opt = self._menu()
            running = self._handle_option(opt)
        plt.close('all')
        print("Salir.")
