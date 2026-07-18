import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Visualizer:
    OPTIONS = [
        {
            'id': 1,
            'label': 'AND \u2014 Convexos e hiperplano separador',
            'info': {
                'title': 'AND \u2014 Convexos e hiperplano separador',
                'algorithm': 'Perceptr\u00f3n escal\u00f3n',
                'equation': 'O = \u03b8(w\u00b7\u03be \u2212 w\u2080)\n\u0394w = \u03b7(\u03b6 \u2212 O)\u03be',
                'description': 'AND es linealmente separable:\nexiste un hiperplano de margen\nm\u00e1ximo que separa ambas\nclases sin error.',
            }
        },
        {
            'id': 2,
            'label': 'XOR \u2014 Convexos intersectados',
            'info': {
                'title': 'XOR \u2014 Convexos intersectados',
                'algorithm': 'Perceptr\u00f3n escal\u00f3n',
                'equation': 'O = \u03b8(w\u00b7\u03be \u2212 w\u2080)\n\u0394w = \u03b7(\u03b6 \u2212 O)\u03be',
                'description': 'XOR no es linealmente separable:\nlos convexos se intersectan,\nno existe hiperplano que\nsepare las clases.',
            }
        },
        {
            'id': 3,
            'label': 'Lineal \u2014 Error cuadr\u00e1tico por \u00e9poca',
            'info': {
                'title': 'Perceptr\u00f3n Lineal \u2014 Error por \u00e9poca',
                'algorithm': 'Perceptr\u00f3n lineal',
                'equation': 'O = w\u00b7\u03be\nE = \u00bd\u03a3(\u03b6 \u2212 O)\u00b2',
                'description': 'Evoluci\u00f3n del error cuadr\u00e1tico\nmedio durante las \u00e9pocas de\nentrenamiento con activaci\u00f3n\nlineal.',
            }
        },
        {
            'id': 4,
            'label': 'Lineal \u2014 Predicci\u00f3n vs Real (Train)',
            'info': {
                'title': 'Lineal \u2014 Train: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n lineal',
                'equation': 'O = w\u00b7\u03be',
                'description': 'Comparaci\u00f3n entre el valor real\ny la predicci\u00f3n del modelo lineal\nen el conjunto de entrenamiento.',
            }
        },
        {
            'id': 5,
            'label': 'Lineal \u2014 Predicci\u00f3n vs Real (Test)',
            'info': {
                'title': 'Lineal \u2014 Test: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n lineal',
                'equation': 'O = w\u00b7\u03be',
                'description': 'Comparaci\u00f3n entre el valor real\ny la predicci\u00f3n del modelo lineal\nen el conjunto de testeo.',
            }
        },
        {
            'id': 6,
            'label': 'Tanh \u2014 Error cuadr\u00e1tico por \u00e9poca',
            'info': {
                'title': 'Perceptr\u00f3n Tanh \u2014 Error por \u00e9poca',
                'algorithm': 'Perceptr\u00f3n no lineal (tanh)',
                'equation': 'O = tanh(\u03b2h)\ng\'(h) = \u03b2(1 \u2212 O\u00b2)\n\u0394w = \u03b7(\u03b6\u2212O)g\'(h)\u03be',
                'description': 'Evoluci\u00f3n del error durante\nentrenamiento con activaci\u00f3n\ntangente hiperb\u00f3lica.\nRango: (\u22121, 1)',
            }
        },
        {
            'id': 7,
            'label': 'Tanh \u2014 Predicci\u00f3n vs Real (Train)',
            'info': {
                'title': 'Tanh \u2014 Train: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n no lineal (tanh)',
                'equation': 'O = tanh(\u03b2h)\nRango: (\u22121, 1)',
                'description': 'Salidas normalizadas a\n[\u22121, 1]; se desnormalizan\npara mostrar predicci\u00f3n vs real\nen entrenamiento.',
            }
        },
        {
            'id': 8,
            'label': 'Tanh \u2014 Predicci\u00f3n vs Real (Test)',
            'info': {
                'title': 'Tanh \u2014 Test: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n no lineal (tanh)',
                'equation': 'O = tanh(\u03b2h)\nRango: (\u22121, 1)',
                'description': 'Salidas normalizadas a\n[\u22121, 1]; se desnormalizan\npara mostrar predicci\u00f3n vs real\nen conjunto de testeo.',
            }
        },
        {
            'id': 9,
            'label': 'Log\u00edstica \u2014 Error cuadr\u00e1tico por \u00e9poca',
            'info': {
                'title': 'Perceptr\u00f3n Log\u00edstico \u2014 Error por \u00e9poca',
                'algorithm': 'Perceptr\u00f3n no lineal (log\u00edstica)',
                'equation': 'O = \u03c3(2\u03b2h) con \u03b2=1\n\u03c3(x) = 1/(1+e\u207b\u02e3)\ng\'(h) = 2\u03b2\u00b7O\u00b7(1\u2212O)\n\u0394w = \u03b7(\u03b6\u2212O)g\'(h)\u03be',
                'description': 'Evoluci\u00f3n del error durante\nentrenamiento con activaci\u00f3n\nlog\u00edstica (sigmoidea).\nRango: (0, 1)',
            }
        },
        {
            'id': 10,
            'label': 'Log\u00edstica \u2014 Predicci\u00f3n vs Real (Train)',
            'info': {
                'title': 'Log\u00edstica \u2014 Train: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n no lineal (log\u00edstica)',
                'equation': 'O = \u03c3(2\u03b2h) con \u03b2=1\nRango: (0, 1)',
                'description': 'Salidas normalizadas a\n[0, 1]; se desnormalizan\npara mostrar predicci\u00f3n vs real\nen entrenamiento.',
            }
        },
        {
            'id': 11,
            'label': 'Log\u00edstica \u2014 Predicci\u00f3n vs Real (Test)',
            'info': {
                'title': 'Log\u00edstica \u2014 Test: Predicci\u00f3n vs Real',
                'algorithm': 'Perceptr\u00f3n no lineal (log\u00edstica)',
                'equation': 'O = \u03c3(2\u03b2h) con \u03b2=1\nRango: (0, 1)',
                'description': 'Salidas normalizadas a\n[0, 1]; se desnormalizan\npara mostrar predicci\u00f3n vs real\nen conjunto de testeo.',
            }
        },
        {
            'id': 12,
            'label': 'Generalizaci\u00f3n \u2014 Error vs tama\u00f1o de entrenamiento',
            'info': {
                'title': 'Generalizaci\u00f3n \u2014 Error vs tama\u00f1o de entrenamiento',
                'algorithm': 'Perceptr\u00f3n lineal',
                'equation': 'Evaluaci\u00f3n con splits\n10% a 50%, 15 corridas c/u',
                'description': 'Error en test versus cantidad\nde datos usados para entrenar.\nMuestra media \u00b1 desv\u00edo est\u00e1ndar.',
            }
        },
    ]

    @staticmethod
    def _get_hull_vertices(points):
        if len(points) <= 1:
            return points
        if len(points) == 2:
            return points
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            return points

    @staticmethod
    def _point_to_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        dist = np.linalg.norm(p - closest)
        return dist, closest

    @staticmethod
    def _segments_intersect(a, b, c, d):
        ab = b - a
        cd = d - c
        denom = np.cross(ab, cd)
        if abs(denom) < 1e-12:
            return False, None
        t = np.cross(c - a, cd) / denom
        u = np.cross(c - a, ab) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True, a + t * ab
        return False, None

    @staticmethod
    def _min_distance_between_hulls(verts_a, verts_b):
        min_dist = np.inf
        p1_best = None
        p2_best = None

        if len(verts_a) >= 2 and len(verts_b) >= 2:
            for i in range(len(verts_a)):
                a1 = verts_a[i]
                a2 = verts_a[(i + 1) % len(verts_a)]
                for j in range(len(verts_b)):
                    b1 = verts_b[j]
                    b2 = verts_b[(j + 1) % len(verts_b)]
                    intersects, pt = Visualizer._segments_intersect(a1, a2, b1, b2)
                    if intersects:
                        return 0.0, pt, pt

        for pa in verts_a:
            for pb in verts_b:
                d = np.linalg.norm(pa - pb)
                if d < min_dist:
                    min_dist = d
                    p1_best = pa
                    p2_best = pb

        if len(verts_b) >= 2:
            for i in range(len(verts_b)):
                b1 = verts_b[i]
                b2 = verts_b[(i + 1) % len(verts_b)]
                for pa in verts_a:
                    d, closest = Visualizer._point_to_segment(pa, b1, b2)
                    if d < min_dist:
                        min_dist = d
                        p1_best = pa
                        p2_best = closest

        if len(verts_a) >= 2:
            for i in range(len(verts_a)):
                a1 = verts_a[i]
                a2 = verts_a[(i + 1) % len(verts_a)]
                for pb in verts_b:
                    d, closest = Visualizer._point_to_segment(pb, a1, a2)
                    if d < min_dist:
                        min_dist = d
                        p1_best = closest
                        p2_best = pb

        return min_dist, p1_best, p2_best

    @staticmethod
    def convex_hulls_and_hyperplane(X, y, ax, title=""):
        y = y.flatten()
        classes = np.unique(y)
        colors = {classes[0]: 'red', classes[1]: 'blue'}
        labels = {classes[0]: f'Clase {classes[0]}', classes[1]: f'Clase {classes[1]}'}

        hull_sets = []
        for cls in classes:
            pts = X[y == cls]
            hull_sets.append(pts)
            ax.scatter(pts[:, 0], pts[:, 1], c=colors[cls], label=labels[cls],
                        s=60, edgecolors='k', zorder=5)

        hull_vertices = []
        for i, pts in enumerate(hull_sets):
            verts = Visualizer._get_hull_vertices(pts)
            hull_vertices.append(verts)
            if len(verts) >= 2:
                poly = np.vstack([verts, verts[0:1]])
                ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color=colors[classes[i]])
                ax.plot(poly[:, 0], poly[:, 1], color=colors[classes[i]], linewidth=2,
                         linestyle='--')

        if len(hull_vertices[0]) > 0 and len(hull_vertices[1]) > 0:
            dist, p1, p2 = Visualizer._min_distance_between_hulls(
                hull_vertices[0], hull_vertices[1]
            )

            if p1 is not None and p2 is not None:
                midpoint = (p1 + p2) / 2
                direction = p2 - p1
                if np.linalg.norm(direction) > 1e-10:
                    normal = direction / np.linalg.norm(direction)
                    perp = np.array([-normal[1], normal[0]])

                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    scale = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

                    hp_p1 = midpoint - perp * scale
                    hp_p2 = midpoint + perp * scale
                    ax.plot([hp_p1[0], hp_p2[0]], [hp_p1[1], hp_p2[1]],
                             'k-', linewidth=2, label='Hiperplano separador', zorder=4)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    @staticmethod
    def regression_results(y_true, y_pred, ax, title=""):
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', label='y = x (ideal)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Valor real (\u03b6)')
        ax.set_ylabel('Predicci\u00f3n (O)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    @staticmethod
    def error_history(errors, ax, title=""):
        ax.plot(errors, linewidth=1.5)
        ax.set_xlabel('\u00c9poca')
        ax.set_ylabel('Error cuadr\u00e1tico total')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def generalization(labels, means, stds, ax, title=""):
        ax.errorbar(labels, means, yerr=stds, marker='o', capsize=5, linewidth=2)
        ax.set_xlabel('Tama\u00f1o del conjunto de testeo')
        ax.set_ylabel('Error cuadr\u00e1tico medio en test')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

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
