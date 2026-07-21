# Redes Neuronales

Repositorio con 4 Trabajos Prácticos sobre redes neuronales, implementados en Python.

## Estructura

```
├── tp1PerceptronSimple/   → Perceptrón simple (desde cero)
├── tp2PerceptronMulticapa/ → Perceptrón multicapa + backpropagation (desde cero)
├── tp3Autocodificadores/   → Autoencoders (Keras/TensorFlow)
├── tp4Convolucionales/     → CNN + Transfer Learning (Keras/TensorFlow)
├── docs/                   → PDFs de teoría y enunciados
└── info/                   → Documentación interna detallada
```

## Trabajos Prácticos

**TP1 — Perceptrón Simple:** Clasificación binaria (AND, XOR) y regresión con un perceptrón implementado desde cero en numpy. Demuestra separabilidad lineal y analiza generalización con distintas funciones de activación.

**TP2 — Perceptrón Multicapa:** MLP con backpropagation desde cero. Resuelve XOR, clasificación par/impar de dígitos 5×7 y reconocimiento de dígitos con evaluación de robustez ante ruido.

**TP3 — Autocodificadores:** Autoencoders en Keras para compresión y denoising de caracteres Font3, y autoencoder generativo sobre MNIST con espacio latente 2D interactivo.

**TP4 — Redes Convolucionales:** Clasificación de imágenes de animales (8 clases) con CNN custom y Transfer Learning (MobileNetV2). El modelo con transfer learning alcanza 96.76% de accuracy.
