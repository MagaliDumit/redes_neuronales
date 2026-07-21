Opción A: Para datos normalizados entre 0 y 1 (Imágenes, probabilidades)
Capas Ocultas (Encoder y Decoder): ReLU.
Capa de Salida: Sigmoide.
Por qué: Asegura que la reconstrucción final tenga exactamente el mismo rango y formato que tus datos originales


Opción B: Para datos sin restricciones (Datos financieros, texto, datos continuos)
Capas Ocultas: ReLU o Leaky ReLU.
Capa de Salida: Lineal (sin función de activación).
Por qué: No limita ni corta los valores, permitiendo que la red prediga cualquier número real al reconstruir la información




1. Datos escalados entre 0 y 1 (Ejemplo: imágenes normalizadas)
Capas ocultas: Usar ReLU (\(f(x) = \max(0, x)\) o Leaky ReLU.
Capa de salida (Decoder): Usar Sigmoide (\(f(x) = \frac{1}{1 + e^{-x}}\)

2. Datos escalados entre -1 y 1 (Estandarización)
Capas ocultas: Usar Tanh (Tangente Hiperbólica).
Capa de salida (Decoder): Usar Tanh

3. Datos continuos sin un rango fijo (Z-score normalizado)
Capas ocultas: Usar ReLU.
Capa de salida (Decoder): Sin activación (activación lineal)


No Lineales (Para capas ocultas)
ReLU: La más famosa. Devuelve 0 si el número es negativo y el mismo número si es positivo. Aprende muy rápido.
Leaky ReLU: Soluciona el problema de las "neuronas muertas" en ReLU. En números negativos, devuelve un valor muy pequeño en lugar de 0.
ELU (Exponential Linear Unit): Es similar a ReLU pero tiene una curva suave para los números negativos.
Sigmoide: Transforma el valor entre 0 y 1. Hoy en día rara vez se usa en las capas ocultas porque puede hacer que el aprendizaje sea muy lento.
Tanh: Transforma el valor entre -1 y 1. Es muy buena cuando los datos están centrados alrededor de cero.

No Lineales (Para capas de salida)
Sigmoide: Ideal para salida de datos que son probabilidades o valores entre 0 y 1.
Softmax: Útil para problemas donde la salida debe ser una categoría. Convierte las salidas en un grupo de probabilidades que suman 1

Las funciones de activación añaden "curvas" a las redes neuronales. Sin ellas, la red solo haría líneas rectas. Una función transforma la entrada de la neurona para que esta decida si "se enciende" o no.
