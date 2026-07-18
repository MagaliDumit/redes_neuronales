import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Visualizer:
    @staticmethod
    def _render_info_panel(ax, info):
        ax.axis('off')
        lines = [
            ("Algoritmo", info['algorithm']),
            ("Ecuaci\u00f3n", info['equation']),
            ("Descripci\u00f3n", info['description']),
        ]
        y_pos = 0.95
        for label, content in lines:
            ax.text(0.05, y_pos, label, transform=ax.transAxes,
                     fontsize=9, fontweight='bold', fontfamily='monospace',
                     verticalalignment='top')
            y_pos -= 0.04
            for line in content.split('\n'):
                ax.text(0.05, y_pos, line, transform=ax.transAxes,
                         fontsize=9, fontfamily='monospace',
                         verticalalignment='top')
                y_pos -= 0.03
            y_pos -= 0.03

    @staticmethod
    def render_split_plot(plot_func, plot_args, info):
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, width_ratios=[3, 2], figsize=(10, 5)
        )
        plot_func(*plot_args, ax=ax_left)
        Visualizer._render_info_panel(ax_right, info)
        fig.suptitle(info['title'], fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def error_history(errors, ax, title=""):
        ax.plot(errors, linewidth=1.5)
        ax.set_xlabel('\u00c9poca')
        ax.set_ylabel('Error cuadr\u00e1tico total')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def decision_boundary_xor(model, X, y, ax, title=""):
        y = y.flatten()
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=60, edgecolors='k', label='Clase +1')
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=60, edgecolors='k', label='Clase -1')

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                              np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'blue'], alpha=0.2)
        ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels, ax, title=""):
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.colorbar(im, ax=ax)

    @staticmethod
    def digit_recognition_grid(model, X, y_onehot, digit_labels):
        n = len(X)
        fig = plt.figure(figsize=(14, 8))
        for i in range(n):
            ax = fig.add_subplot(2, 5, i + 1)
            img = X[i].reshape(7, 5)
            ax.imshow(img, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            true_digit = np.argmax(y_onehot[i])
            preds = model.predict(X[i:i+1])
            pred_digit = np.argmax(preds[0])
            parity = "Par" if pred_digit % 2 == 0 else "Impar"
            correct = pred_digit == true_digit
            color = 'green' if correct else 'red'
            ax.set_title(
                f"D\u00edgito {digit_labels[true_digit]}\n"
                f"Pred: {pred_digit} ({parity})",
                fontsize=10, color=color
            )
        fig.suptitle(
            "Reconocimiento de d\u00edgitos 5\u00d77 \u2014 Predicci\u00f3n y paridad",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def noisy_comparison(model, X_orig, y_orig, p, digit_labels):
        n = len(X_orig)
        fig = plt.figure(figsize=(14, 4 * ((n + 2) // 3)))

        noise = np.random.binomial(1, p, size=X_orig.shape).astype(np.float32)
        X_noisy = np.where(noise == 1, 1.0 - X_orig, X_orig)

        pred_orig_all = model.predict(X_orig)
        pred_noisy_all = model.predict(X_noisy)
        pred_orig_digits = np.argmax(pred_orig_all, axis=1)
        pred_noisy_digits = np.argmax(pred_noisy_all, axis=1)
        true_digits = np.argmax(y_orig, axis=1)
        acc_noisy = np.mean(pred_noisy_digits == true_digits)

        for i in range(n):
            true_d = true_digits[i]

            ax_orig = fig.add_subplot(n, 2, 2 * i + 1)
            img_orig = X_orig[i].reshape(7, 5)
            ax_orig.imshow(img_orig, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            c_orig = 'green' if pred_orig_digits[i] == true_d else 'red'
            ax_orig.set_title(f"Original: {digit_labels[true_d]} \u2192 {pred_orig_digits[i]}",
                              fontsize=10, color=c_orig)

            ax_noisy = fig.add_subplot(n, 2, 2 * i + 2)
            img_noisy = X_noisy[i].reshape(7, 5)
            ax_noisy.imshow(img_noisy, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
            ax_noisy.set_xticks([])
            ax_noisy.set_yticks([])
            c_noisy = 'green' if pred_noisy_digits[i] == true_d else 'red'
            ax_noisy.set_title(f"Ruidoso: {digit_labels[true_d]} \u2192 {pred_noisy_digits[i]}",
                               fontsize=10, color=c_noisy)

        fig.suptitle(
            f"D\u00edgitos de entrenamiento \u2014 Original (izq) vs Con ruido p={p:.3f} (der)\n"
            f"Accuracy con ruido: {acc_noisy:.2%}",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def highlight_digit_in_file(filepath, digit):
        GREEN = '\033[32m'
        GRAY = '\033[90m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        with open(filepath) as f:
            all_lines = [l.rstrip('\n') for l in f]

        digit_start = digit * 7
        digit_end = digit_start + 7

        print(f"\n{BOLD}Archivo: {filepath}{RESET}")
        print(f"{BOLD}D\u00edgito resaltado: {digit}{RESET}")
        print()
        context_before = max(0, digit_start - 3)
        context_after = min(len(all_lines), digit_end + 3)

        for i in range(context_before, context_after):
            line_num = i + 1
            is_digit_line = digit_start <= i < digit_end
            if is_digit_line:
                nums = all_lines[i].split()
                painted = ' '.join(
                    f"{GREEN}{n}{RESET}" if n == '1' else n
                    for n in nums
                )
                print(f"  {line_num:4d}  {painted}  {GRAY}\u2190 D\u00cdGITO {digit}{RESET}")
            else:
                print(f"{GRAY}  {line_num:4d}  {all_lines[i]}{RESET}")
        print()

    @staticmethod
    def accuracy_comparison(labels, values, ax, title=""):
        bars = ax.bar(labels, values, color=['blue', 'red'])
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
