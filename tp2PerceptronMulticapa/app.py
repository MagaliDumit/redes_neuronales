import numpy as np
import matplotlib.pyplot as plt
from mlp import ModularMultiLayerPerceptron
from data import DataHandler
from visualization import Visualizer


OPTIONS = [
    {
        'id': 1,
        'label': 'XOR \u2014 Frontera de decisi\u00f3n (MLP 2-2-1)',
        'info': {
            'title': 'XOR con Perceptr\u00f3n Multicapa',
            'algorithm': 'MLP 2-2-1 (tanh-tanh)',
            'equation': '\u03b4_M = g\'(h_M)*(z - O)\n\u03b4_m = g\'(h_m)*(W_{m+1}^T @ \u03b4_{m+1})\n\u0394W_m = \u03b7 * \u03b4_m \u2297 V_{m-1}',
            'description': 'XOR no es linealmente separable\ncon una capa, pero el MLP\ncon capa oculta lo resuelve.',
        }
    },
    {
        'id': 2,
        'label': 'Error cuadr\u00e1tico por \u00e9poca (XOR)',
        'info': {
            'title': 'XOR \u2014 Error por \u00e9poca',
            'algorithm': 'MLP 2-2-1 (tanh-tanh)',
            'equation': 'E = \u00bd\u03a3(\u03b6 - O)\u00b2',
            'description': 'Evoluci\u00f3n del error durante\nel entrenamiento del MLP\npara el problema XOR.',
        }
    },
    {
        'id': 3,
        'label': 'Todos los d\u00edgitos \u2014 Predicci\u00f3n y paridad',
        'info': {
            'title': 'Reconocimiento de d\u00edgitos 5\u00d77',
            'algorithm': 'MLP 35-15-10 (tanh-tanh)',
            'equation': '10 salidas one-hot\nPar = {0,2,4,6,8}\nImpar = {1,3,5,7,9}',
            'description': 'Los 10 d\u00edgitos 0-9 con su\npredicci\u00f3n y clasificaci\u00f3n\nen par o impar.',
        }
    },
    {
        'id': 4,
        'label': 'Ruido \u2014 Originales vs ruidosos (entrenamiento)',
        'info': {
            'title': 'D\u00edgitos de entrenamiento \u2014 Original vs con ruido',
            'algorithm': 'MLP 35-15-10 (tanh-tanh)',
            'equation': 'Ruido: invertir bits\ncon probabilidad p=0.02',
            'description': 'Comparaci\u00f3n de cada d\u00edgito\nde entrenamiento original\nvs su versi\u00f3n con ruido,\ny la predicci\u00f3n en cada caso.',
        }
    },
    {
        'id': 5,
        'label': 'Resaltar d\u00edgito \u2014 Ingresar por consola',
        'info': {
            'title': 'D\u00edgito resaltado',
            'algorithm': 'Visualizaci\u00f3n de matriz 5\u00d77',
            'equation': 'Pixel 1 = azul (#1E90FF)\nPixel 0 = gris (#F0F0F0)',
            'description': 'Ingres\u00e1 un n\u00famero (0-9)\npor consola para ver su\nmatriz de p\u00edxeles 5\u00d77\nresaltada.',
        }
    },
]


class MenuApp:
    def __init__(self):
        self.xor_model = None
        self.xor_errors = []
        self.par_impar_model = None
        self.par_impar_errors = []
        self.par_impar_data = {}
        self.onehot_model = None
        self.onehot_errors = []
        self.onehot_data = {}
        self.digit_labels = list('0123456789')

    def _train_xor(self):
        X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        y = np.array([[1], [1], [-1], [-1]])

        model = ModularMultiLayerPerceptron([2, 2, 1], ['tanh', 'tanh'], eta=0.1)
        errors = []
        for epoch in range(2000):
            total_err = 0
            for V, t in zip(X, y):
                total_err += model.train_sample(V, t)
            errors.append(total_err / len(X))
            if total_err < 0.01:
                break
        self.xor_model = model
        self.xor_errors = errors
        return epoch + 1, errors[-1]

    def _train_par_impar(self):
        X = DataHandler.load_digit_pixels('datos/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt')
        y_binary = np.array([1 if d % 2 == 0 else -1 for d in range(10)])
        y = y_binary.reshape(-1, 1)

        X_train, X_test, y_train, y_test = DataHandler.train_test_split(X, y, test_size=0.4)
        self.par_impar_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_binary': y_binary,
        }

        model = ModularMultiLayerPerceptron([35, 10, 1], ['tanh', 'tanh'], eta=0.05)
        errors = []
        for epoch in range(1000):
            err = model.train_batch(X_train, y_train)
            errors.append(err)
            if err < 0.001:
                break

        self.par_impar_model = model
        self.par_impar_errors = errors
        return epoch + 1, errors[-1]

    def _train_onehot(self):
        X_original = DataHandler.load_digit_pixels('datos/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt')
        y_original = np.eye(10, dtype=np.float32)

        indices = np.arange(10)
        np.random.shuffle(indices)
        X = X_original[indices]
        y = y_original[indices]
        split = 6
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = ModularMultiLayerPerceptron([35, 15, 10], ['tanh', 'tanh'], eta=0.05)
        errors = []
        for epoch in range(2000):
            err = model.train_batch(X_train, y_train)
            errors.append(err)
            if err < 0.001:
                break

        def add_noise(x, p=0.02):
            noise = np.random.binomial(1, p, size=x.shape).astype(np.float32)
            return np.where(noise == 1, 1.0 - x, x)

        X_train_noisy = add_noise(X_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        pred_train_noisy = model.predict(X_train_noisy)

        acc_train = np.mean(np.argmax(pred_train, axis=1) == np.argmax(y_train, axis=1))
        acc_test = np.mean(np.argmax(pred_test, axis=1) == np.argmax(y_test, axis=1))
        acc_train_noisy = np.mean(np.argmax(pred_train_noisy, axis=1) == np.argmax(y_train, axis=1))

        self.onehot_model = model
        self.onehot_errors = errors
        self.onehot_data = {
            'X_train': X_train, 'X_test': X_test,
            'X_train_noisy': X_train_noisy,
            'y_train': y_train, 'y_test': y_test,
            'X_original': X_original, 'y_original': y_original,
            'pred_train': pred_train,
            'pred_train_noisy': pred_train_noisy,
            'acc_train': acc_train, 'acc_test': acc_test,
            'acc_train_noisy': acc_train_noisy,
        }
        return epoch + 1, acc_test, acc_train_noisy

    def _train_digit_recognizer(self):
        X = DataHandler.load_digit_pixels('datos/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt')
        y = np.eye(10, dtype=np.float32)

        model = ModularMultiLayerPerceptron([35, 20, 10], ['tanh', 'tanh'], eta=0.1)
        errors = []
        for epoch in range(5000):
            err = model.train_batch(X, y)
            errors.append(err)
            if err < 0.001:
                break

        preds = model.predict(X)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
        self.digit_model = model
        self.digit_errors = errors
        return epoch + 1, acc

    def setup(self):
        np.random.seed(42)

        print("Ejercicio 1: XOR con MLP...")
        xor_epochs, xor_err = self._train_xor()
        print(f"  Convergi\u00f3 en {xor_epochs} \u00e9pocas (error final: {xor_err:.6f})")

        print("Ejercicio 2: Par/Impar...")
        pi_epochs, pi_err = self._train_par_impar()
        print(f"  {pi_epochs} \u00e9pocas (error final: {pi_err:.6f})")

        print("Ejercicio 3: One-hot con ruido...")
        oh_epochs, acc, acc_noisy = self._train_onehot()
        print(f"  {oh_epochs} \u00e9pocas, acc test: {acc:.3f}, acc train con ruido: {acc_noisy:.3f}")

        print("Reconocimiento completo de d\u00edgitos...")
        dr_epochs, dr_acc = self._train_digit_recognizer()
        print(f"  {dr_epochs} \u00e9pocas, acc total: {dr_acc:.3f}")

        print("\n\u00a1Todos los modelos entrenados!\n")

    def _menu(self):
        last_id = max(o['id'] for o in OPTIONS)
        show_all = last_id + 1
        print("=" * 60)
        print("   TP2 \u2014 PERCEPTR\u00d3N MULTICAPA")
        print("=" * 60)
        print()
        print("Seleccion\u00e1 una opci\u00f3n:")
        print()
        for opt in OPTIONS:
            print(f"  [{opt['id']:2d}] {opt['label']}")
        print(f"  [{show_all:2d}] Mostrar todas las ventanas")
        print("  [ 0] Salir")
        print()
        return input("Opci\u00f3n: ").strip()

    def _handle_option(self, opt):
        if opt == '0':
            return False

        elif opt == '1':
            X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
            y = np.array([[1], [1], [-1], [-1]])
            Visualizer.render_split_plot(
                Visualizer.decision_boundary_xor,
                (self.xor_model, X, y),
                OPTIONS[0]['info']
            )

        elif opt == '2':
            Visualizer.render_split_plot(
                Visualizer.error_history,
                (self.xor_errors,),
                OPTIONS[1]['info']
            )

        elif opt == '3':
            X = DataHandler.load_digit_pixels('datos/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt')
            y = np.eye(10, dtype=np.float32)
            Visualizer.digit_recognition_grid(
                self.digit_model,
                X, y,
                self.digit_labels
            )

        elif opt == '4':
            d = self.onehot_data
            try:
                p = float(input("Ingres\u00e1 la probabilidad de ruido (0-1, default 0.02): ") or "0.02")
                if p < 0 or p > 1:
                    print("Debe ser un valor entre 0 y 1.")
                else:
                    Visualizer.noisy_comparison(
                        self.onehot_model,
                        d['X_train'], d['y_train'],
                        p,
                        self.digit_labels
                    )
            except ValueError:
                print("Entrada inv\u00e1lida. Debe ser un n\u00famero.")

        elif opt == '5':
            try:
                dig = int(input("Ingres\u00e1 un d\u00edgito (0-9): "))
                if dig < 0 or dig > 9:
                    print("Debe ser un n\u00famero entre 0 y 9.")
                else:
                    Visualizer.highlight_digit_in_file(
                        'datos/TP2-ej3-mapa-de-pixeles-digitos-decimales.txt', dig
                    )
            except ValueError:
                print("Entrada inv\u00e1lida. Debe ser un n\u00famero.")

        elif opt == '6':
            last_id = max(o['id'] for o in OPTIONS)
            for i in range(1, last_id + 1):
                print(f"Mostrando opci\u00f3n {i}...")
                self._handle_option(str(i))

        else:
            print("Opci\u00f3n no v\u00e1lida.")

        return True

    def run(self):
        self.setup()
        running = True
        while running:
            opt = self._menu()
            running = self._handle_option(opt)
        plt.close('all')
        print("Salir.")
