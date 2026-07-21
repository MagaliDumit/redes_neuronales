import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Visualizer:

    @staticmethod
    def _draw_char(ax, vector, title=''):
        img = np.round(vector).reshape(7, 5)
        ax.imshow(img, cmap='binary', aspect='equal', vmin=0, vmax=1)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=8)

    @staticmethod
    def reconstructions(originals, reconstructions, labels):
        n = len(originals)
        fig = plt.figure(figsize=(10, 2 + n * 0.9))
        gs = GridSpec(n, 4, width_ratios=[0.4, 1, 0.4, 1], figure=fig)

        for i in range(n):
            ax_ol = fig.add_subplot(gs[i, 1])
            Visualizer._draw_char(ax_ol, originals[i])
            ax_rl = fig.add_subplot(gs[i, 3])
            Visualizer._draw_char(ax_rl, reconstructions[i])

            if i == 0:
                ax_ol.set_title('Original', fontsize=9, fontweight='bold')
                ax_rl.set_title('Reconst.', fontsize=9, fontweight='bold')

            fig.text(0.05, 0.5 + 0.5 * (1 - 2 * i / n),
                     f'[{labels[i] if labels else i}]',
                     fontsize=8, fontfamily='monospace',
                     verticalalignment='center')

        fig.suptitle('Autoencoder — Reconstrucciones', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def interactive_latent_space(codes, labels, all_chars, decoder):
        fig = plt.figure(figsize=(13, 6))
        gs = GridSpec(1, 3, width_ratios=[3.2, 1.2, 1.6], figure=fig)

        ax_scatter = fig.add_subplot(gs[0, 0])

        ax_scatter.scatter(codes[:, 0], codes[:, 1],
                           c='#2196F3', alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
        for i in range(len(codes)):
            lbl = labels[i] if i < len(labels) else ''
            ax_scatter.annotate(lbl, (codes[i, 0], codes[i, 1]),
                                fontsize=7, alpha=0.85)

        ax_scatter.set_xlabel('z\u2081')
        ax_scatter.set_ylabel('z\u2082')
        ax_scatter.set_title('Espacio latente 2D — Click interactivo', fontsize=11)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_aspect('equal')

        ax_preview = fig.add_subplot(gs[0, 1])
        ax_preview.axis('off')
        ax_preview.set_title('Previsualización', fontsize=9, fontweight='bold')
        ax_preview.text(
            0.5, 0.5, 'Hacé clic en un punto\npara seleccionarlo,\no en espacio vacío\npara generar.',
            transform=ax_preview.transAxes, fontsize=8, fontfamily='monospace',
            ha='center', va='center'
        )

        ax_info = fig.add_subplot(gs[0, 2])
        ax_info.axis('off')
        info_lines = [
            ('Modelo', 'AE 35→17→7→2→7→17→35'),
            ('Latente', '2D (tanh, [-1,1])'),
            ('Loss', 'binary_crossentropy'),
            ('', ''),
            ('Interacción', 'Click punto → selecciona'),
            ('', 'Click otro punto → interpola'),
            ('', 'Click vacío → genera'),
        ]
        y_pos = 0.95
        for label, content in info_lines:
            if label:
                ax_info.text(0.05, y_pos, label, transform=ax_info.transAxes,
                             fontsize=9, fontweight='bold', fontfamily='monospace',
                             verticalalignment='top')
                y_pos -= 0.04
            ax_info.text(0.05, y_pos, content, transform=ax_info.transAxes,
                         fontsize=9, fontfamily='monospace',
                         verticalalignment='top')
            y_pos -= 0.03

        fig.suptitle('Autoencoder — Espacio latente 2D', fontsize=14, fontweight='bold')

        state = {'sel_idx': None, 'highlight': None}

        def _show_img(vector, title=''):
            ax_preview.clear()
            ax_preview.axis('off')
            ax_preview.set_title('Previsualización', fontsize=9, fontweight='bold')
            img = np.round(vector).reshape(7, 5)
            ax_preview.imshow(img, cmap='binary', aspect='equal', vmin=0, vmax=1,
                              interpolation='nearest')
            if title:
                ax_preview.set_xlabel(title, fontsize=8, fontfamily='monospace')

        def _show_interpolation(code_a, code_b, label_a, label_b):
            ax_preview.clear()
            ax_preview.axis('off')
            alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            z_all = (alphas[:, np.newaxis] * code_a[np.newaxis, :]
                     + (1.0 - alphas[:, np.newaxis]) * code_b[np.newaxis, :])
            gen_batch = np.round(decoder.predict(z_all, verbose=0))
            steps = [g.reshape(7, 5) for g in gen_batch]
            sep = np.zeros((7, 1))
            combined = steps[0]
            for s in steps[1:]:
                combined = np.hstack([combined, sep, s])
            ax_preview.imshow(combined, cmap='binary', aspect='equal', vmin=0, vmax=1,
                              interpolation='nearest')
            ax_preview.set_xticks([])
            ax_preview.set_yticks([])
            for j, a in enumerate(alphas):
                x_pos = (j + 0.5) / len(alphas)
                ax_preview.text(x_pos, -0.15, f'\u03b1={a:.2f}',
                                transform=ax_preview.transAxes,
                                fontsize=7, ha='center', fontfamily='monospace')
            ax_preview.set_title(f'Interpolación [{label_a}] \u2192 [{label_b}]',
                                 fontsize=8, fontweight='bold')

        def on_click(event):
            if event.inaxes != ax_scatter:
                return

            click_pt = np.array([event.xdata, event.ydata])
            dists = np.sqrt(np.sum((codes - click_pt) ** 2, axis=1))
            nearest = np.argmin(dists)

            x_range = ax_scatter.get_xlim()[1] - ax_scatter.get_xlim()[0]
            y_range = ax_scatter.get_ylim()[1] - ax_scatter.get_ylim()[0]
            threshold = 0.04 * max(x_range, y_range)

            if state['highlight'] is not None:
                state['highlight'].remove()
                state['highlight'] = None

            if dists[nearest] < threshold:
                if state['sel_idx'] is None:
                    state['sel_idx'] = nearest
                    state['highlight'] = ax_scatter.scatter(
                        [codes[nearest, 0]], [codes[nearest, 1]],
                        c='none', edgecolors='red', s=200, linewidths=2, zorder=10
                    )
                    _show_img(all_chars[nearest], f'Seleccionado [{labels[nearest]}]')
                else:
                    idx_a = state['sel_idx']
                    idx_b = nearest
                    _show_interpolation(codes[idx_a], codes[idx_b],
                                        labels[idx_a], labels[idx_b])
                    state['sel_idx'] = None
            else:
                k = 3
                nearest_k = np.argsort(dists)[:k]
                weights = 1.0 / (dists[nearest_k] + 1e-10)
                weights /= weights.sum()
                z_new = np.sum(codes[nearest_k] * weights[:, np.newaxis], axis=0)
                gen = decoder.predict(z_new[np.newaxis, :], verbose=0)[0]
                neighbor_str = ', '.join([labels[j] for j in nearest_k])
                _show_img(gen, f'Generado de: [{neighbor_str}]')
                state['sel_idx'] = None

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def denoising_grid(originals, noisy, reconstructions, noise_level):
        n = len(originals)
        fig = plt.figure(figsize=(10, 2 + n * 0.9))
        gs = GridSpec(n, 6, width_ratios=[0.4, 1, 0.3, 1, 0.3, 1], figure=fig)

        for i in range(n):
            ax_ol = fig.add_subplot(gs[i, 1])
            Visualizer._draw_char(ax_ol, originals[i])
            ax_n = fig.add_subplot(gs[i, 3])
            Visualizer._draw_char(ax_n, noisy[i])
            ax_r = fig.add_subplot(gs[i, 5])
            Visualizer._draw_char(ax_r, reconstructions[i])

            if i == 0:
                ax_ol.set_title('Original', fontsize=9, fontweight='bold')
                ax_n.set_title(f'Ruido {noise_level:.0%}', fontsize=9, fontweight='bold')
                ax_r.set_title('Reconst.', fontsize=9, fontweight='bold')

        fig.suptitle(f'Denoising — {noise_level:.0%} ruido', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def interpolation_chars(originals, alphas, generated, label_a, label_b):
        fig = plt.figure(figsize=(10, 4))
        gs = GridSpec(2, len(alphas), figure=fig, left=0.05, right=0.95, top=0.85, bottom=0.15)

        for j, (alpha, gen) in enumerate(zip(alphas, generated)):
            ax = fig.add_subplot(gs[1, j])
            Visualizer._draw_char(ax, gen, f'\u03b1={alpha:.2f}')

        ax_ol = fig.add_subplot(gs[0, 0])
        Visualizer._draw_char(ax_ol, originals[0], f'Inicio [{label_a}]')
        ax_or = fig.add_subplot(gs[0, -1])
        Visualizer._draw_char(ax_or, originals[1], f'Fin [{label_b}]')

        for j in range(1, len(alphas) - 1):
            ax_h = fig.add_subplot(gs[0, j])
            ax_h.axis('off')

        fig.suptitle(f'Interpolación [{label_a}] \u2192 [{label_b}]', fontsize=13, fontweight='bold')
        plt.show()

    @staticmethod
    def latent_noise(original, code, noises, generated):
        fig = plt.figure(figsize=(10, 3.5))
        gs = GridSpec(1, len(noises) + 1, figure=fig, left=0.05, right=0.95, top=0.75, bottom=0.15)

        ax_o = fig.add_subplot(gs[0, 0])
        Visualizer._draw_char(ax_o, original, 'Original')

        for j, (eps, gen) in enumerate(zip(noises, generated)):
            ax = fig.add_subplot(gs[0, j + 1])
            Visualizer._draw_char(ax, gen, f'\u03b5={eps:.1f}')

        fig.suptitle('Ruido Gaussiano en espacio latente', fontsize=13, fontweight='bold')
        plt.show()

    @staticmethod
    def generated_grid(samples, n_samples):
        cols = 4
        rows = (n_samples + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(10, 2 * rows))
        if rows == 1:
            axs = axs[np.newaxis, :]

        for i in range(n_samples):
            r, c = i // cols, i % cols
            Visualizer._draw_char(axs[r, c], samples[i], f'Nuevo {i + 1}')

        for i in range(n_samples, rows * cols):
            r, c = i // cols, i % cols
            axs[r, c].axis('off')

        fig.suptitle('Generación — Muestreo uniforme del latente', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def mnist_reconstructions(originals, reconstructions, novel):
        n = len(originals)
        fig = plt.figure(figsize=(14, 5))
        gs_top = GridSpec(1, n, left=0.04, right=0.98, top=0.88, bottom=0.55, figure=fig)
        gs_mid = GridSpec(1, n, left=0.04, right=0.98, top=0.52, bottom=0.19, figure=fig)
        gs_bot = GridSpec(1, min(len(novel), 5), left=0.04, right=0.65, top=0.16, bottom=0.01, figure=fig)

        for i in range(n):
            ax_orig = fig.add_subplot(gs_top[0, i])
            ax_orig.imshow(originals[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_title('Originales', fontsize=9, fontweight='bold')

            ax_rec = fig.add_subplot(gs_mid[0, i])
            ax_rec.imshow(reconstructions[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax_rec.axis('off')
            if i == 0:
                ax_rec.set_title('Reconstruidas', fontsize=9, fontweight='bold')

        for i in range(min(len(novel), 5)):
            ax_gen = fig.add_subplot(gs_bot[0, i])
            ax_gen.imshow(novel[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax_gen.axis('off')
            if i == 0:
                ax_gen.set_title('Generadas (latente)', fontsize=9, fontweight='bold')

        fig.suptitle('MNIST — Autoencoder', fontsize=14, fontweight='bold')
        plt.show()
