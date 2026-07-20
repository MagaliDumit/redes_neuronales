import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


class Visualizer:
    """
    Visualizaciones requeridas por el TP:
      - Muestras por clase
      - Curvas de entrenamiento (accuracy y loss)
      - Matriz de confusión
      - Predicciones en test
      - Inferencia sobre imagen externa
      - Comparativa de métricas
      - Mapas de activación convolucional
      - Errores de clasificación
    """

    @staticmethod
    def _show_image(ax, img, title='', fontsize=7):
        """Muestra una imagen en el eje dado. Soporta RGB y escala de grises."""
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img, vmin=0, vmax=1)
        else:
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=fontsize)

    @staticmethod
    def _render_info(ax, lines, title_section=''):
        """Renderiza texto informativo en un panel lateral."""
        ax.axis('off')
        y_pos = 0.95
        if title_section:
            ax.text(0.05, y_pos, title_section, transform=ax.transAxes,
                     fontsize=10, fontweight='bold', fontfamily='monospace',
                     verticalalignment='top')
            y_pos -= 0.06
        for line in lines:
            ax.text(0.05, y_pos, line, transform=ax.transAxes,
                     fontsize=8, fontfamily='monospace',
                     verticalalignment='top')
            y_pos -= 0.035

    @staticmethod
    def muestras_por_clase(X, y, classes):
        """
        Muestra 4 ejemplos aleatorios de cada clase.
        y está en one-hot, se convierte a índices con argmax.
        """
        y_idx = np.argmax(y, axis=1) if y.ndim > 1 else y
        n_classes = len(classes)
        fig, axes = plt.subplots(n_classes, 5, figsize=(14, 2 * n_classes))
        fig.suptitle('Muestras del dataset por clase', fontsize=14, fontweight='bold')

        for i, cls in enumerate(classes):
            mask = y_idx == i
            idxs = np.where(mask)[0]
            chosen = np.random.choice(idxs, min(4, len(idxs)), replace=False)

            axes[i, 0].text(0.5, 0.5, cls, fontsize=11, fontweight='bold',
                             ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')

            for j, idx in enumerate(chosen):
                Visualizer._show_image(axes[i, j + 1], X[idx])

            for j in range(5):
                if j > len(chosen):
                    axes[i, j].axis('off')

        info = [
            f'Total de clases: {n_classes}',
            f'Total de imagenes: {len(X)}',
            f'Tamaño de entrada: {X.shape[1]}x{X.shape[2]}',
            f'Canales: {X.shape[3]} (RGB)',
            '',
            'Codificación one-hot:',
            'Vector de 8 posiciones',
            'con un 1 en la clase',
            'correspondiente y 0',
            'en el resto.',
            '',
            'Ej: gato en pos 1 ->',
            '[0,1,0,0,0,0,0,0]',
        ]
        ax_info = fig.add_subplot(111, frameon=False)
        ax_info.axis('off')
        ax_info.text(0.92, 0.5, '\n'.join(info), transform=ax_info.transAxes,
                      fontsize=8, fontfamily='monospace', verticalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()

    @staticmethod
    def historial_entrenamiento(history, title='Curvas de entrenamiento'):
        """Grafica accuracy y loss a lo largo de las épocas."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title, fontsize=13, fontweight='bold')

        epochs = range(1, len(history.history['loss']) + 1)

        # Accuracy
        ax1.plot(epochs, history.history['accuracy'], 'b-', label='Train')
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validacion')
        ax1.set_xlabel('Epoca')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Epocas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(epochs, history.history['loss'], 'b-', label='Train')
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validacion')
        ax2.set_xlabel('Epoca')
        ax2.set_ylabel('Loss')
        ax2.set_title('Perdida (Loss) vs Epocas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        info = [
            f'Epocas entrenadas: {len(epochs)}',
            f'Accuracy final train: {history.history["accuracy"][-1]:.4f}',
            f'Accuracy final val: {history.history["val_accuracy"][-1]:.4f}',
            f'Loss final train: {history.history["loss"][-1]:.4f}',
            f'Loss final val: {history.history["val_loss"][-1]:.4f}',
        ]
        if history.history.get('lr'):
            info.append(f'LR final: {history.history["lr"][-1]:.6f}')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax_info = fig.add_axes([0.85, 0.3, 0.13, 0.4])
        ax_info.axis('off')
        ax_info.text(0, 0.5, '\n'.join(info), transform=ax_info.transAxes,
                      fontsize=8, fontfamily='monospace', verticalalignment='center',
                      bbox=props)

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show()

    @staticmethod
    def matriz_confusion(y_true, y_pred, classes):
        """
        Muestra la matriz de confusión con colores.
        y_true está en one-hot, se convierte a índices.
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('Matriz de confusion', fontsize=14, fontweight='bold')

        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel('Clase predicha', fontsize=11)
        ax.set_ylabel('Clase real', fontsize=11)
        plt.colorbar(im, ax=ax)

        # Mostrar valores en cada celda
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=10, color=color)

        # Métricas
        acc = np.trace(cm) / cm.sum()
        info = [
            f'Accuracy: {acc:.4f}',
            '',
            'Lectura: fila = clase real,',
            'columna = clase predicha.',
            'Diagonal principal =',
            'aciertos.',
            'Fuera de diagonal =',
            'errores.',
        ]
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax_info = fig.add_axes([0.82, 0.4, 0.16, 0.3])
        ax_info.axis('off')
        ax_info.text(0, 0.5, '\n'.join(info), transform=ax_info.transAxes,
                      fontsize=8, fontfamily='monospace', verticalalignment='center',
                      bbox=props)

        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.show()

    @staticmethod
    def predicciones_test(X, y_true, y_pred, probs, classes, n_show=16):
        """
        Muestra n_show imágenes de test con predicción real vs predicha.
        y_true en one-hot -> argmax. Verde = acierto, rojo = error.
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        idxs = np.random.choice(len(X), min(n_show, len(X)), replace=False)
        cols = 4
        rows = int(np.ceil(len(idxs) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        fig.suptitle('Predicciones en test', fontsize=14, fontweight='bold')
        axes = axes.flatten() if rows > 1 else [axes]

        for pos, idx in enumerate(idxs):
            ax = axes[pos]
            Visualizer._show_image(ax, X[idx])

            true_label = classes[int(y_true[idx])]
            pred_label = classes[int(y_pred[idx])]
            prob = probs[idx, int(y_pred[idx])]
            color = 'green' if int(y_pred[idx]) == int(y_true[idx]) else 'red'
            ax.set_title(f'Real: {true_label}\nPred: {pred_label} ({prob:.2f})',
                          fontsize=8, color=color)

        for i in range(len(idxs), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def inferencia(img_array, img_display, model_custom, model_transfer,
                   classes):
        """
        Toma una imagen externa, la pasa por ambos modelos y muestra:
          - La imagen original
          - Barras de probabilidad para cada clase en ambos modelos
        """
        # Predicciones
        pred_c, probs_c = None, None
        pred_t, probs_t = None, None

        if model_custom is not None:
            probs_c = model_custom.predict(img_array, verbose=0)[0]
            pred_c = np.argmax(probs_c)
        if model_transfer is not None:
            probs_t = model_transfer.predict(img_array, verbose=0)[0]
            pred_t = np.argmax(probs_t)

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('Inferencia - Comparacion Custom vs Transfer Learning',
                      fontsize=14, fontweight='bold')
        gs = GridSpec(1, 3, width_ratios=[1, 1.5, 1.5], figure=fig)

        # Imagen original
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_display)
        ax_img.axis('off')
        ax_img.set_title('Imagen de entrada', fontsize=10)

        colores = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                    '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

        # Barras Custom CNN
        if probs_c is not None:
            ax_c = fig.add_subplot(gs[0, 1])
            ypos = range(len(classes))
            bars = ax_c.barh(ypos, probs_c, color=colores)
            bars[pred_c].set_color('green')
            ax_c.set_yticks(ypos)
            ax_c.set_yticklabels(classes, fontsize=8)
            ax_c.set_xlim(0, 1)
            ax_c.set_title(f'Custom CNN\nPred: {classes[pred_c]} ({probs_c[pred_c]:.2f})',
                            fontsize=10, fontweight='bold')
            ax_c.invert_yaxis()
            ax_c.set_xlabel('Probabilidad')

        # Barras Transfer Learning
        if probs_t is not None:
            ax_t = fig.add_subplot(gs[0, 2])
            ypos = range(len(classes))
            bars = ax_t.barh(ypos, probs_t, color=colores)
            bars[pred_t].set_color('green')
            ax_t.set_yticks(ypos)
            ax_t.set_yticklabels(classes, fontsize=8)
            ax_t.set_xlim(0, 1)
            ax_t.set_title(f'Transfer Learning\nPred: {classes[pred_t]} ({probs_t[pred_t]:.2f})',
                            fontsize=10, fontweight='bold')
            ax_t.invert_yaxis()
            ax_t.set_xlabel('Probabilidad')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def comparativa(y_test, y_pred_custom, y_pred_transfer, classes):
        """
        Compara lado a lado las métricas de ambos modelos:
          - Accuracy global
          - Precision, Recall, F1-score por clase
          - Macro y weighted average
        """
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Comparativa: Custom CNN vs Transfer Learning',
                      fontsize=14, fontweight='bold')
        ax.axis('off')

        acc_c = np.mean(y_test == y_pred_custom)
        acc_t = np.mean(y_test == y_pred_transfer)

        report_c = classification_report(y_test, y_pred_custom,
                                          target_names=classes,
                                          output_dict=True, zero_division=0)
        report_t = classification_report(y_test, y_pred_transfer,
                                          target_names=classes,
                                          output_dict=True, zero_division=0)

        # Armar texto comparativo
        lines = []
        lines.append(f"{'=' * 65}")
        lines.append(f"  COMPARATIVA CUSTOM CNN vs TRANSFER LEARNING")
        lines.append(f"{'=' * 65}")
        lines.append("")
        lines.append(f"  {'Accuracy Custom:':30s} {acc_c:.4f}")
        lines.append(f"  {'Accuracy Transfer:':30s} {acc_t:.4f}")
        lines.append(f"  {'Diferencia (Transfer - Custom):':30s} {acc_t - acc_c:+.4f}")
        lines.append("")
        lines.append(f"{'-' * 65}")
        lines.append(f"  {'Clase':15s} {'Custom F1':>10s} {'Transfer F1':>12s} {'Diff':>10s}")
        lines.append(f"{'-' * 65}")

        for cls_name in classes:
            f1_c = report_c[cls_name]['f1-score']
            f1_t = report_t[cls_name]['f1-score']
            diff = f1_t - f1_c
            lines.append(f"  {cls_name:15s} {f1_c:10.3f} {f1_t:12.3f} {diff:+10.3f}")

        lines.append(f"{'-' * 65}")
        lines.append(f"  {'Macro avg F1':15s} "
                      f"{report_c['macro avg']['f1-score']:10.3f} "
                      f"{report_t['macro avg']['f1-score']:12.3f}")
        lines.append(f"  {'Weighted avg F1':15s} "
                      f"{report_c['weighted avg']['f1-score']:10.3f} "
                      f"{report_t['weighted avg']['f1-score']:12.3f}")
        lines.append("")
        lines.append(f"{'=' * 65}")
        lines.append("  Analisis:")
        if acc_t > acc_c + 0.05:
            lines.append("  Transfer Learning supera claramente al modelo custom.")
        elif acc_t > acc_c:
            lines.append("  Transfer Learning tiene mejor rendimiento que el custom.")
        elif abs(acc_t - acc_c) < 0.03:
            lines.append("  Ambos modelos tienen rendimiento similar.")
        else:
            lines.append("  El modelo custom supera al Transfer Learning.")

        y_pos = 0.95
        for line in lines:
            ax.text(0.05, y_pos, line, transform=ax.transAxes,
                     fontsize=8, fontfamily='monospace', verticalalignment='top')
            y_pos -= 0.028

        plt.tight_layout()
        plt.show()

    @staticmethod
    def errores_clasificacion(X, y_true, y_pred, classes):
        """
        Muestra ejemplos donde el modelo se equivocó.
        y_true en one-hot -> argmax.
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        errors = np.where(y_pred != y_true)[0]
        if len(errors) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'Sin errores de clasificacion!',
                     ha='center', va='center', transform=ax.transAxes,
                     fontsize=14)
            ax.axis('off')
            plt.show()
            return

        n_show = min(16, len(errors))
        idxs = np.random.choice(errors, n_show, replace=False)
        cols = 4
        rows = int(np.ceil(n_show / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        fig.suptitle(f'Errores de clasificacion (total: {len(errors)})',
                      fontsize=14, fontweight='bold')
        axes = axes.flatten() if rows > 1 else [axes]

        for pos, idx in enumerate(idxs):
            ax = axes[pos]
            Visualizer._show_image(ax, X[idx])
            ax.set_title(
                f'Real: {classes[int(y_true[idx])]}\nPred: {classes[int(y_pred[idx])]}',
                fontsize=8, color='red'
            )

        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def mapas_activacion(model, X, idx=0):
        """
        Muestra los mapas de activación de la última capa convolucional.
        Permite ver qué características detecta la red en una imagen.
        """
        # Encontrar la última capa Conv2D
        conv_layers = [l for l in model.layers
                       if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No se encontraron capas convolucionales',
                     ha='center', va='center', transform=ax.transAxes,
                     fontsize=12)
            ax.axis('off')
            plt.show()
            return

        last_conv = conv_layers[-1]
        feature_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=last_conv.output
        )

        features = feature_model.predict(X[idx:idx + 1], verbose=0)[0]

        n_filters = min(16, features.shape[-1])
        cols = 4
        rows = int(np.ceil(n_filters / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        fig.suptitle(f'Mapas de activacion - Ultima capa Conv2D ({last_conv.name})',
                      fontsize=14, fontweight='bold')
        axes = axes.flatten() if rows > 1 else [axes]

        # Mostrar imagen original
        ax_orig = fig.add_axes([0.01, 0.85, 0.08, 0.08])
        Visualizer._show_image(ax_orig, X[idx], title='Original', fontsize=7)

        for i in range(n_filters):
            ax = axes[i]
            ax.imshow(features[:, :, i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filtro {i}', fontsize=7)

        for i in range(n_filters, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
