import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

EXTENSIONES_VALIDAS = ('.png', '.jpg', '.jpeg', '.webp')


def get_classes(data_dir):
    """Obtiene la lista de clases (nombres de carpetas) ordenadas alfabéticamente."""
    folders = sorted(os.listdir(data_dir))
    classes = []
    for f in folders:
        p = os.path.join(data_dir, f)
        if os.path.isdir(p):
            classes.append(f.strip())
    return classes


def _resolver_carpeta(base, nombre):
    """Busca la carpeta manejando espacios al final del nombre."""
    ruta = os.path.join(base, nombre)
    if os.path.isdir(ruta):
        return ruta
    alt = ruta + ' '
    if os.path.isdir(alt):
        return alt
    alt = ruta.rstrip()
    if os.path.isdir(alt):
        return alt
    return ruta


def _cargar_imagenes_gris(carpeta, label, h, w):
    """
    Escanea 'carpeta' en busca de imágenes, las lee en escala de grises,
    las redimensiona a (h, w) y normaliza a [0, 1].
    Retorna listas (X, y) donde X tiene forma (h, w, 1).
    """
    X, y = [], []
    if not os.path.isdir(carpeta):
        return X, y
    for fname in os.listdir(carpeta):
        if not fname.lower().endswith(EXTENSIONES_VALIDAS):
            continue
        path = os.path.join(carpeta, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        # Agregar dimensión de canal: (h, w) -> (h, w, 1)
        X.append(img[..., np.newaxis])
        y.append(label)
    return X, y


class DataLoader:
    """
    Carga imágenes en escala de grises con forma (224, 224, 1).
    Train desde subcarpeta 'entrenamiento/', test desde la raíz de cada clase.

    La salida one-hot significa que cada clase se representa como un vector
    de 8 elementos con un 1 en la posición de la clase y 0 en el resto.
    Ejemplo: gato en posición 1 -> [0, 1, 0, 0, 0, 0, 0, 0]
    """

    IMG_SIZE = (224, 224)

    @staticmethod
    def load_images(data_dir):
        """
        Carga imágenes en gris separando train (desde 'entrenamiento/')
        y test (desde raíz de cada clase).

        Retorna:
          X_train, X_test: arrays (N, 224, 224, 1) normalizados [0,1]
          y_train, y_test: arrays (N, 8) en codificación one-hot
          classes, class_to_idx
        """
        classes = get_classes(data_dir)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)

        X_train, y_train = [], []
        X_test, y_test = [], []
        h, w = DataLoader.IMG_SIZE

        for cls_name in classes:
            cls_dir = _resolver_carpeta(data_dir, cls_name)
            label = class_to_idx[cls_name]

            # Train desde 'entrenamiento/'
            train_dir = os.path.join(cls_dir, 'entrenamiento')
            xt, yt = _cargar_imagenes_gris(train_dir, label, h, w)
            X_train.extend(xt)
            y_train.extend(yt)

            # Test desde la raíz (excluyendo subcarpetas)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    fpath = os.path.join(cls_dir, fname)
                    if os.path.isdir(fpath):
                        continue
                    if not fname.lower().endswith(EXTENSIONES_VALIDAS):
                        continue
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (w, h))
                    img = img.astype(np.float32) / 255.0
                    X_test.append(img[..., np.newaxis])
                    y_test.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # One-hot encoding
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        print(f"  Train: {len(X_train)} imagenes ({len(classes)} clases)")
        print(f"  Test:  {len(X_test)} imagenes")
        print(f"  Formato: escala de grises {X_train.shape[1]}x{X_train.shape[2]}x1")
        print(f"  Clases: {classes}")
        for i, cls in enumerate(classes):
            tr = int((y_train.argmax(axis=1) == i).sum())
            te = int((y_test.argmax(axis=1) == i).sum())
            print(f"    {cls:15s} train={tr}  test={te}")

        return X_train, X_test, y_train, y_test, classes, class_to_idx

    @staticmethod
    def load_external_image(path):
        """
        Carga imagen externa en gris para inferencia.
        Retorna (img_array, img_display):
          img_array: (1, 224, 224, 1) lista para el modelo
          img_display: (224, 224) en gris para mostrar
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        h, w = DataLoader.IMG_SIZE
        img_resized = cv2.resize(img, (w, h))
        img_display = img_resized.copy()
        img_normalized = img_resized.astype(np.float32) / 255.0
        # (224, 224) -> (1, 224, 224, 1)
        img_array = np.expand_dims(img_normalized[..., np.newaxis], axis=0)
        return img_array, img_display
