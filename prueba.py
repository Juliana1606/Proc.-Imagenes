import numpy as np

def calc_means(features_array):
    features_array = np.array(features_array)  # Convertir la lista a un array de NumPy
    return np.mean(features_array, axis=0)  # Calcular la media por columnas

# Ejemplo de uso:
features = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    # ... (hasta 50 filas)
]

means = calc_means(features)
print(means)