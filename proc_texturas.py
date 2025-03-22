import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from skimage.filters.rank import entropy
from skimage.morphology import disk

np.set_printoptions(threshold = np.inf)

#Paths carpetas datasets
datasets_path = ['../Proc.-Imagenes/datasets/potholed', '../Proc.-Imagenes/datasets/bubbly', '../Proc.-Imagenes/datasets/zigzag', '../Proc.-Imagenes/datasets/tests']

#Carga de datasets
dataset_photoled = load_dataset(datasets_path[0])
dataset_bubbly = load_dataset(datasets_path[1])
dataset_zigzag = load_dataset(datasets_path[2])
dataset_tests = load_dataset(datasets_path[3])

#Guardar imagenes en array
photoled_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_photoled['train']]
bubbly_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_bubbly['train']]
zigzag_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_zigzag['train']]

#Guardar imagenes en diccionario para acceder a nombres
tests_dict = {}
for i in range(dataset_tests['test'].num_rows):
    image = cv.cvtColor(np.array(dataset_tests['test'][i]['image']), cv.COLOR_RGB2GRAY)
    tests_dict[f'Imagen {i}'] = image  # Asignar la imagen a la clave correspondiente
print(tests_dict.keys())

#Procesamiento de imagenes de datasets de entreno
photoled_array_processed = photoled_array.copy()
bubbly_array_processed = bubbly_array.copy()
zigzag_array_processed = zigzag_array.copy()


def img_aumentobrillo(img):
    img_aclarada = np.uint8(np.sqrt(255 * np.double(img)))
    return img_aclarada

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree   
    filters = []
    angulos = []
    num_filters = 8  # Number of filters to create
    ksize = 7  # The local area to evaluate // Kernel size
    lambd = 10.0 # Wavelength of the sinusoidal factor
    sigma = 0.5 * lambd # Larger Values produce more edges // Standard deviation of the gaussian envelope
    gamma = 0.2 #
    psi = 0  # Offset value - lower generates cleaner results

    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        angulos.append(theta)
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= np.linalg.norm(kern)  # Usa la norma en lugar de la suma
        filters.append(kern)
        
    return filters, angulos

filters, angulos = create_gaborfilter()

#Aplicar filtro gabor
def apply_gaborfilter(img, filters):
    filtered_img = []
    for i in filters:
        result = cv.filter2D(img, cv.CV_64F, i)  # Asegura precisión con CV_64F
        result = np.array(result, dtype=np.uint8)  # Convierte a uint8 para OpenCV
        filtered_img.append(result)
    return filtered_img

def extract_features(img):
    features = []

    #Media y desviación estandar [0] y [1]
    features.append(np.mean(img))
    features.append(np.std(img))
    #Varianza y mediana [2] y [3]
    features.append(np.var(img))
    features.append(np.median(img))

    #Energía y entropía [4] y [5]
    #energy = np.sum(img**2)
    #features.append(energy)
    #gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #entropy_value = np.sum(entropy(gray_img, disk(5)))
    #features.append(entropy_value)

    return features

def calc_means(features_array):
    features_array = np.array(features_array)  # Convertir la lista a un array de NumPy
    return np.mean(features_array, axis=0)  # Calcular la media por columnas

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

#Preprocesamiento de imagenes datasets de entreno
for i in range(len(photoled_array)):
    photoled_array_processed[i] = img_aumentobrillo(photoled_array[i])

for i in range(len(bubbly_array)):
    bubbly_array_processed[i] = img_aumentobrillo(bubbly_array[i])

for i in range(len(zigzag_array)):
    zigzag_array_processed[i] = img_aumentobrillo(zigzag_array[i])

#Aplicar filtro gabor
filtered_photoled = photoled_array.copy()
filtered_bubbly = bubbly_array.copy()
filtered_zigzag = zigzag_array.copy()


print(type(filters[0]))

for i in range(len(photoled_array_processed)):
    filtered_photoled[i] = apply_gaborfilter(photoled_array_processed[i], filters)

for i in range(len(bubbly_array_processed)):
    filtered_bubbly[i] = apply_gaborfilter(bubbly_array_processed[i], filters)

for i in range(len(zigzag_array_processed)):
    filtered_zigzag[i] = apply_gaborfilter(zigzag_array_processed[i], filters)

#Extraer features datasets de entreno
features_photoled = filtered_photoled.copy()
features_bubbly = filtered_bubbly.copy()
features_zigzag = filtered_zigzag.copy()

for i in range(len(filtered_photoled)):
    for j in range(len(filtered_photoled[i])):
        features_photoled[i] = extract_features(filtered_photoled[i][j])

for i in range(len(filtered_bubbly)):
    for j in range(len(filtered_bubbly[i])):
        features_bubbly[i] = extract_features(filtered_bubbly[i][j])

for i in range(len(filtered_zigzag)):
    for j in range(len(filtered_zigzag[i])):
        features_zigzag[i] = extract_features(filtered_zigzag[i][j])

#Calcular medias de los 8 filtros puestos en la imagen
print(len(features_photoled))
means_photoled = calc_means(features_photoled)
means_bubbly = calc_means(features_bubbly)
means_zigzag = calc_means(features_zigzag)

print(f"PROMEDIOS photoled: {means_photoled}")
print(f"PROMEDIOS bubbly: {means_bubbly}")
print(f"PROMEDIOS zigzag: {means_zigzag}")

#PRUEBA
num_img = 2 #0 hasta N - 1 imagenes cargadas en el dataset test
test_img_name = f"Imagen {num_img}"  # Nombre de la imagen

if num_img < len(tests_dict):
    test_img = tests_dict[test_img_name]  # Cargar la imagen
    cv.imshow(f'Imagen para test: {test_img_name}', test_img)  # Mostrar el nombre de la imagen
    cv.waitKey(0)
    cv.destroyAllWindows()
    test_img_proccessed = img_aumentobrillo(test_img)
    test_img_filtered = apply_gaborfilter(test_img_proccessed, filters)  # Aplica Gabor
    print(len(test_img_filtered))
    test_features = []
    for i in range(len(test_img_filtered)):
        test_features.append(extract_features(test_img_filtered[i]))  # Extrae características
    print(test_features)
    means_test = calc_means(test_features)
    print(f"PROMEDIOS TEST: {means_test}")

    dist_photoled = euclidean_distance(means_test, means_photoled)
    dist_bubbly = euclidean_distance(means_test, means_bubbly)
    dist_zigzag = euclidean_distance(means_test, means_zigzag)

    distances = {'photoled': dist_photoled, 'bubbly': dist_bubbly, 'zigzag': dist_zigzag}
    closest_texture = min(distances, key=distances.get)
    print(distances)
    print(closest_texture)

    sim_photoled = cosine_similarity(means_test, means_photoled)
    sim_bubbly = cosine_similarity(means_test, means_bubbly)
    sim_zigzag = cosine_similarity(means_test, means_zigzag)

    # Determina la textura más similar
    similarities = {'photoled': sim_photoled, 'bubbly': sim_bubbly, 'zigzag': sim_zigzag}
    most_similar_texture = max(similarities, key=similarities.get)
    print(similarities)
    print(most_similar_texture)
else:
    print(f'No se encontró la imagen: {test_img_name}')


