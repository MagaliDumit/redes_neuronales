import os
import numpy as np
from data import DataLoader
from models import build_custom_cnn, build_transfer_model, compile_model
from trainer import Trainer
from visualizer import Visualizer


class MenuApp:
    """
    Aplicación interactiva con menú para el TP4 - Redes Convolucionales.

    Flujo de trabajo:
      1. Carga y preparación de datos (train/test split, one-hot)
      2. Construcción y entrenamiento del modelo Custom CNN
      3. Construcción y entrenamiento del modelo Transfer Learning (MobileNetV2)
      4. Fine-tuning del modelo Transfer Learning
      5. Menú interactivo para visualizar resultados
    """

    def __init__(self):
        # Datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = []

        # Modelos
        self.model_custom = None
        self.model_transfer = None
        self.transfer_base = None

        # Historiales de entrenamiento
        self.history_custom = None
        self.history_transfer = None
        self.history_finetune = None

        # Predicciones
        self.y_pred_custom = None
        self.y_pred_transfer = None
        self.probs_custom = None
        self.probs_transfer = None

    def setup(self):
        """Carga datos, construye y entrena ambos modelos."""
        data_dir = os.path.join(os.path.dirname(__file__), 'datos')

        # ============================================================
        # PUNTO 1 y 2: Preparación de datos y capa de salida one-hot
        # ============================================================
        print("=" * 65)
        print("  TP4 - REDES NEURONALES CONVOLUCIONALES")
        print("  Clasificacion de animales - 8 clases")
        print("=" * 65)
        print("\n[1/5] Cargando y preparando datos...")
        print("  - Redimensionando a 224x224")
        print("  - Escala de grises (1 canal)")
        print("  - Normalizando pixeles a [0, 1]")
        print("  - Codificacion one-hot (8 neuronas)")
        print("  - Train: imagenes en carpetas 'entrenamiento/'")
        print("  - Test:  imagenes en raiz de cada clase")
        print()

        self.X_train, self.X_test, self.y_train, self.y_test, self.classes, _ = \
            DataLoader.load_images(data_dir)

        print(f"\n  Train: {self.X_train.shape[0]} imagenes")
        print(f"  Test:  {self.X_test.shape[0]} imagenes")

        # ============================================================
        # PUNTO 3: Modelo Custom (Arquitectura desde cero)
        # ============================================================
        print("\n[2/5] Construyendo y entrenando modelo Custom CNN...")
        print("  Arquitectura: 4 bloques Conv2D(32->64->128->256)")
        print("  + BatchNorm + MaxPooling + GAP + Dense(128) + Dropout(0.5)")
        print("  + Softmax(8)")
        print()

        self.model_custom = build_custom_cnn(
            input_shape=(224, 224, 1),
            num_classes=len(self.classes)
        )
        compile_model(self.model_custom, lr=0.001)
        self.model_custom.summary()

        print("\n  Entrenando Custom CNN (con EarlyStopping)...")
        self.history_custom = Trainer.train(
            self.model_custom,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=100, batch_size=16, verbose=1
        )
        epochs_c = len(self.history_custom.history['loss'])
        acc_c = self.history_custom.history['accuracy'][-1]
        val_c = self.history_custom.history['val_accuracy'][-1]
        print(f"\n  Custom CNN: {epochs_c} ep, acc={acc_c:.4f}, val_acc={val_c:.4f}")

        # ============================================================
        # PUNTO 4: Transfer Learning con MobileNetV2
        # ============================================================
        print("\n[3/5] Construyendo y entrenando modelo Transfer Learning...")
        print("  Base: MobileNetV2 preentrenado en ImageNet (congelado)")
        print("  Cabeza: GAP + Dense(128) + Dropout(0.3) + Softmax(8)")
        print()

        self.model_transfer, self.transfer_base = build_transfer_model(
            input_shape=(224, 224, 1),
            num_classes=len(self.classes)
        )
        compile_model(self.model_transfer, lr=0.001)
        self.model_transfer.summary()

        print("\n  Entrenando cabeza clasificadora (base congelada)...")
        self.history_transfer = Trainer.train(
            self.model_transfer,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=50, batch_size=16, verbose=1
        )
        epochs_t = len(self.history_transfer.history['loss'])
        acc_t = self.history_transfer.history['accuracy'][-1]
        val_t = self.history_transfer.history['val_accuracy'][-1]
        print(f"\n  Transfer: {epochs_t} ep, acc={acc_t:.4f}, val_acc={val_t:.4f}")

        # ============================================================
        # PUNTO 4 (cont): Fine-tuning
        # ============================================================
        print("\n[4/5] Aplicando fine-tuning (descongelando capas > 100)...")
        print("  Learning rate: 0.00001 (100x menor)")
        print("  Se descongelan las ultimas capas de MobileNetV2")
        print()

        self.history_finetune = Trainer.fine_tune(
            self.model_transfer, self.transfer_base,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            unfreeze_from=100,
            epochs=30, batch_size=16, verbose=1
        )
        ft_epochs = len(self.history_finetune.history['loss'])
        ft_acc = self.history_finetune.history['accuracy'][-1]
        ft_val = self.history_finetune.history['val_accuracy'][-1]
        print(f"\n  Fine-tune: {ft_epochs} ep, acc={ft_acc:.4f}, val_acc={ft_val:.4f}")

        # ============================================================
        # Evaluación final en test
        # ============================================================
        print("\n[5/5] Evaluando modelos en conjunto de test...")
        loss_c, acc_c = Trainer.evaluate(self.model_custom, self.X_test, self.y_test)
        loss_t, acc_t = Trainer.evaluate(self.model_transfer, self.X_test, self.y_test)
        print(f"\n  Custom CNN:   loss={loss_c:.4f}, accuracy={acc_c:.4f}")
        print(f"  Transfer Learning: loss={loss_t:.4f}, accuracy={acc_t:.4f}")

        # Guardar modelos en formato .keras
        print("\n  Guardando modelos en formato .keras...")
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        modelos_dir = os.path.join(ruta_base, 'modelos')
        os.makedirs(modelos_dir, exist_ok=True)
        self.model_custom.save(os.path.join(modelos_dir, 'modelo_custom.keras'))
        self.model_transfer.save(os.path.join(modelos_dir, 'modelo_transfer.keras'))
        print(f"  -> modelos/modelo_custom.keras")
        print(f"  -> modelos/modelo_transfer.keras")

        # Generar predicciones para visualización
        print("\n  Generando predicciones...")
        self.y_pred_custom, self.probs_custom = Trainer.predict(
            self.model_custom, self.X_test)
        self.y_pred_transfer, self.probs_transfer = Trainer.predict(
            self.model_transfer, self.X_test)

        print("\n" + "=" * 65)
        print("  Modelos listos! Ingresando al menu interactivo...")
        print("=" * 65)

    def _menu(self):
        """Muestra el menú interactivo y retorna la opción seleccionada."""
        print()
        print("=" * 65)
        print("   MENU DE VISUALIZACION")
        print("=" * 65)
        print()
        print("  [1] Muestras del dataset por clase")
        print("  [2] Curvas de entrenamiento - Custom CNN")
        print("  [3] Curvas de entrenamiento - Transfer Learning")
        print("  [4] Curvas de entrenamiento - Fine-Tuning")
        print("  [5] Matriz de confusion - Custom CNN")
        print("  [6] Matriz de confusion - Transfer Learning")
        print("  [7] Predicciones en test - Custom CNN")
        print("  [8] Predicciones en test - Transfer Learning")
        print("  [9] Inferencia (cargar imagen externa)")
        print(" [10] Comparativa Custom vs Transfer")
        print(" [11] Errores de clasificacion - Custom CNN")
        print(" [12] Mapas de activacion - Custom CNN")
        print(" [13] Mostrar TODAS las visualizaciones")
        print("  [0] Salir")
        print()
        return input("Opcion: ").strip()

    def _handle_option(self, opt):
        """Ejecuta la acción correspondiente a la opción seleccionada."""
        if opt == '0':
            return False

        if opt == '1':
            Visualizer.muestras_por_clase(self.X_test, self.y_test, self.classes)

        elif opt == '2':
            Visualizer.historial_entrenamiento(
                self.history_custom,
                title='Curvas de entrenamiento - Custom CNN desde cero'
            )

        elif opt == '3':
            Visualizer.historial_entrenamiento(
                self.history_transfer,
                title='Curvas de entrenamiento - Transfer Learning (MobileNetV2)'
            )

        elif opt == '4':
            Visualizer.historial_entrenamiento(
                self.history_finetune,
                title='Curvas de entrenamiento - Fine-Tuning'
            )

        elif opt == '5':
            Visualizer.matriz_confusion(
                self.y_test, self.y_pred_custom, self.classes
            )

        elif opt == '6':
            Visualizer.matriz_confusion(
                self.y_test, self.y_pred_transfer, self.classes
            )

        elif opt == '7':
            Visualizer.predicciones_test(
                self.X_test, self.y_test, self.y_pred_custom,
                self.probs_custom, self.classes
            )

        elif opt == '8':
            Visualizer.predicciones_test(
                self.X_test, self.y_test, self.y_pred_transfer,
                self.probs_transfer, self.classes
            )

        elif opt == '9':
            path = input("  Ruta de la imagen: ").strip()
            if not path:
                print("  Ruta vacia.")
                return True
            path = os.path.expanduser(path)
            if not os.path.isfile(path):
                print(f"  Archivo no encontrado: {path}")
                return True
            img_array, img_display = DataLoader.load_external_image(path)
            if img_array is None:
                print("  No se pudo cargar la imagen.")
                return True
            Visualizer.inferencia(
                img_array, img_display,
                self.model_custom, self.model_transfer,
                self.classes
            )

        elif opt == '10':
            Visualizer.comparativa(
                self.y_test, self.y_pred_custom,
                self.y_pred_transfer, self.classes
            )

        elif opt == '11':
            Visualizer.errores_clasificacion(
                self.X_test, self.y_test, self.y_pred_custom, self.classes
            )

        elif opt == '12':
            idx = np.random.randint(len(self.X_test))
            Visualizer.mapas_activacion(self.model_custom, self.X_test, idx=idx)

        elif opt == '13':
            for i in range(1, 13):
                print(f"\n  Mostrando opcion {i}...")
                self._handle_option(str(i))

        else:
            print("  Opcion no valida.")

        return True

    def run(self):
        """Punto de entrada: ejecuta setup y luego el menú interactivo."""
        self.setup()
        running = True
        while running:
            opt = self._menu()
            running = self._handle_option(opt)
        import matplotlib.pyplot as plt
        plt.close('all')
        print("\n  Salir.")
