import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from perceptron import Perceptron
from data import DataHandler
from visualization import Visualizer


class MenuApp:
    def __init__(self):
        self.models = {}
        self.data = {}
        self.predictions = {}
        self.errors = {}

    def _train_and(self):
        X_and = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        y_and = np.array([[-1], [-1], [-1], [1]])
        model = Perceptron(input_dim=2, output_dim=1, activation='step', eta=0.1)
        for epoch in range(100):
            total_error = 0
            for V, t in zip(X_and, y_and):
                total_error += model.train_sample(V, t)
            if total_error == 0:
                break
        self.data['and'] = (X_and, y_and)
        self.models['and'] = model
        return epoch + 1

    def _train_xor(self):
        X_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        y_xor = np.array([[1], [1], [-1], [-1]])
        model = Perceptron(input_dim=2, output_dim=1, activation='step', eta=0.1)
        converged = False
        for epoch in range(200):
            total_error = 0
            for V, t in zip(X_xor, y_xor):
                total_error += model.train_sample(V, t)
            if total_error == 0:
                converged = True
                break
        self.data['xor'] = (X_xor, y_xor)
        self.models['xor'] = model
        return converged, epoch + 1

    def _train_linear(self):
        X, y = DataHandler.load_tp1_data(
            "datos/TP1-ej2-Conjunto-entrenamiento.txt",
            "datos/TP1-ej2-Salida-deseada.txt"
        )
        self.data['tp1'] = (X, y)

        X_train, X_test, y_train, y_test = DataHandler.train_test_split(
            X, y, test_size=0.2
        )
        model = Perceptron(input_dim=3, output_dim=1, activation='linear', eta=0.0001)
        errors = []
        for _ in range(200):
            total_err = 0
            for V, t in zip(X_train, y_train):
                total_err += model.train_sample(V, t)
            errors.append(total_err / len(X_train))

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        self.models['linear'] = model
        self.errors['linear'] = errors
        self.data['linear_train'] = (X_train, y_train)
        self.data['linear_test'] = (X_test, y_test)
        self.predictions['linear_train'] = y_train_pred
        self.predictions['linear_test'] = y_test_pred

        train_mse = DataHandler.evaluate_mse(model, X_train, y_train)
        test_mse = DataHandler.evaluate_mse(model, X_test, y_test)
        return train_mse, test_mse

    def _train_tanh(self):
        X, y = self.data['tp1']

        X_train, X_test, y_train, y_test = DataHandler.train_test_split(
            X, y, test_size=0.2
        )
        y_train_norm, y_min, y_max = DataHandler.normalize_targets(
            y_train, low=-1.0, high=1.0
        )
        y_test_norm = (y_test - y_min) * (1 - (-1)) / (y_max - y_min) + (-1)

        model = Perceptron(input_dim=3, output_dim=1, activation='tanh', eta=0.01)
        errors = []
        for _ in range(300):
            total_err = 0
            for V, t in zip(X_train, y_train_norm):
                total_err += model.train_sample(V, t)
            errors.append(total_err / len(X_train))

        y_train_norm_pred = model.predict(X_train)
        y_test_norm_pred = model.predict(X_test)
        y_train_pred = DataHandler.denormalize_targets(
            y_train_norm_pred, y_min, y_max, low=-1.0, high=1.0
        )
        y_test_pred = DataHandler.denormalize_targets(
            y_test_norm_pred, y_min, y_max, low=-1.0, high=1.0
        )

        self.models['tanh'] = model
        self.errors['tanh'] = errors
        self.data['tanh_train'] = (X_train, y_train)
        self.data['tanh_test'] = (X_test, y_test)
        self.predictions['tanh_train'] = y_train_pred
        self.predictions['tanh_test'] = y_test_pred

        train_mse = np.mean((y_train.flatten() - y_train_pred.flatten()) ** 2)
        test_mse = np.mean((y_test.flatten() - y_test_pred.flatten()) ** 2)
        return train_mse, test_mse

    def _train_logistic(self):
        X, y = self.data['tp1']
        X_train, X_test, y_train, y_test = DataHandler.train_test_split(
            X, y, test_size=0.2
        )
        y_train_norm, y_min, y_max = DataHandler.normalize_targets(
            y_train, low=0.0, high=1.0
        )
        y_test_norm = (y_test - y_min) * (1 - 0) / (y_max - y_min) + 0

        model = Perceptron(input_dim=3, output_dim=1, activation='logistic', eta=0.01)
        errors = []
        for _ in range(300):
            total_err = 0
            for V, t in zip(X_train, y_train_norm):
                total_err += model.train_sample(V, t)
            errors.append(total_err / len(X_train))

        y_train_norm_pred = model.predict(X_train)
        y_test_norm_pred = model.predict(X_test)
        y_train_pred = DataHandler.denormalize_targets(
            y_train_norm_pred, y_min, y_max, low=0.0, high=1.0
        )
        y_test_pred = DataHandler.denormalize_targets(
            y_test_norm_pred, y_min, y_max, low=0.0, high=1.0
        )

        self.models['logistic'] = model
        self.errors['logistic'] = errors
        self.data['logistic_train'] = (X_train, y_train)
        self.data['logistic_test'] = (X_test, y_test)
        self.predictions['logistic_train'] = y_train_pred
        self.predictions['logistic_test'] = y_test_pred

        train_mse = np.mean((y_train.flatten() - y_train_pred.flatten()) ** 2)
        test_mse = np.mean((y_test.flatten() - y_test_pred.flatten()) ** 2)
        return train_mse, test_mse

    def _run_generalization(self):
        X, y = self.data['tp1']
        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        mse_by_size = {f"Test {int(s * 100)}%": [] for s in test_sizes}
        n_runs = 15
        for s in test_sizes:
            for _ in range(n_runs):
                X_tr, X_te, y_tr, y_te = DataHandler.train_test_split(
                    X, y, test_size=s
                )
                model = Perceptron(
                    input_dim=3, output_dim=1, activation='linear', eta=0.0001
                )
                for _ in range(150):
                    for V, t in zip(X_tr, y_tr):
                        model.train_sample(V, t)
                mse_by_size[f"Test {int(s * 100)}%"].append(
                    DataHandler.evaluate_mse(model, X_te, y_te)
                )
        labels = sorted(mse_by_size.keys())
        means = [np.mean(mse_by_size[l]) for l in labels]
        stds = [np.std(mse_by_size[l]) for l in labels]
        return labels, means, stds

    def setup(self):
        np.random.seed(42)

        and_epochs = self._train_and()
        xor_converged, xor_epochs = self._train_xor()
        lin_train_mse, lin_test_mse = self._train_linear()
        tanh_train_mse, tanh_test_mse = self._train_tanh()
        log_train_mse, log_test_mse = self._train_logistic()
        self.gen_labels, self.gen_means, self.gen_stds = self._run_generalization()

        print(f"AND: convergi\u00f3 en {and_epochs} \u00e9pocas")
        if xor_converged:
            print(f"XOR: convergi\u00f3 en {xor_epochs} \u00e9pocas")
        else:
            print(f"XOR: no convergi\u00f3 (no es linealmente separable)")
        print(f"Lineal    \u2014 Train MSE: {lin_train_mse:.4f}  Test MSE: {lin_test_mse:.4f}")
        print(f"Tanh      \u2014 Train MSE: {tanh_train_mse:.4f}  Test MSE: {tanh_test_mse:.4f}")
        print(f"Log\u00edstica \u2014 Train MSE: {log_train_mse:.4f}  Test MSE: {log_test_mse:.4f}")
        print(f"Generalizaci\u00f3n lista.")
        print("\n\u00a1Todos los modelos entrenados! Seleccion\u00e1 una opci\u00f3n del men\u00fa.\n")

    def _menu(self):
        last_id = max(o['id'] for o in Visualizer.OPTIONS)
        show_all = last_id + 1
        print("=" * 60)
        print("   TP1 \u2014 PERCEPTR\u00d3N SIMPLE")
        print("=" * 60)
        print()
        print("Seleccion\u00e1 una opci\u00f3n:")
        print()
        for opt in Visualizer.OPTIONS:
            print(f"  [{opt['id']:2d}] {opt['label']}")
        print(f"  [{show_all:2d}] Mostrar todas las ventanas")
        print("  [ 0] Salir")
        print()
        return input("Opci\u00f3n: ").strip()

    def _handle_option(self, opt):
        if opt == '0':
            return False

        elif opt == '1':
            X_and, y_and = self.data['and']
            Visualizer.render_split_plot(
                Visualizer.convex_hulls_and_hyperplane,
                (X_and, y_and),
                Visualizer.OPTIONS[0]['info']
            )

        elif opt == '2':
            X_xor, y_xor = self.data['xor']
            Visualizer.render_split_plot(
                Visualizer.convex_hulls_and_hyperplane,
                (X_xor, y_xor),
                Visualizer.OPTIONS[1]['info']
            )

        elif opt == '3':
            Visualizer.render_split_plot(
                Visualizer.error_history,
                (self.errors['linear'],),
                Visualizer.OPTIONS[2]['info']
            )

        elif opt == '4':
            _, y_train = self.data['linear_train']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_train, self.predictions['linear_train']),
                Visualizer.OPTIONS[3]['info']
            )

        elif opt == '5':
            _, y_test = self.data['linear_test']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_test, self.predictions['linear_test']),
                Visualizer.OPTIONS[4]['info']
            )

        elif opt == '6':
            Visualizer.render_split_plot(
                Visualizer.error_history,
                (self.errors['tanh'],),
                Visualizer.OPTIONS[5]['info']
            )

        elif opt == '7':
            _, y_train = self.data['tanh_train']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_train, self.predictions['tanh_train']),
                Visualizer.OPTIONS[6]['info']
            )

        elif opt == '8':
            _, y_test = self.data['tanh_test']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_test, self.predictions['tanh_test']),
                Visualizer.OPTIONS[7]['info']
            )

        elif opt == '9':
            Visualizer.render_split_plot(
                Visualizer.error_history,
                (self.errors['logistic'],),
                Visualizer.OPTIONS[8]['info']
            )

        elif opt == '10':
            _, y_train = self.data['logistic_train']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_train, self.predictions['logistic_train']),
                Visualizer.OPTIONS[9]['info']
            )

        elif opt == '11':
            _, y_test = self.data['logistic_test']
            Visualizer.render_split_plot(
                Visualizer.regression_results,
                (y_test, self.predictions['logistic_test']),
                Visualizer.OPTIONS[10]['info']
            )

        elif opt == '12':
            info = Visualizer.OPTIONS[11]['info']
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, width_ratios=[3, 2], figsize=(10, 5)
            )
            Visualizer.generalization(
                self.gen_labels, self.gen_means, self.gen_stds, ax_left,
                title='Generalizaci\u00f3n: Error en test vs tama\u00f1o de entrenamiento'
            )
            Visualizer._render_info_panel(ax_right, info)
            fig.suptitle(info['title'], fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        elif opt == '13':
            last_id = max(o['id'] for o in Visualizer.OPTIONS)
            for i in range(1, last_id + 1):
                print(f"Mostrando opci\u00f3n {i}...")
                self._handle_option(str(i))

        else:
            print("Opci\u00f3n no v\u00e1lida. Intent\u00e1 de nuevo.")

        return True

    def run(self):
        self.setup()
        running = True
        while running:
            opt = self._menu()
            running = self._handle_option(opt)
        plt.close('all')
        print("Salir.")
