import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from skimage.filters.rank import entropy
from skimage.morphology import disk

np.set_printoptions(threshold = np.inf)

#Paths carpetas datasets
datasets_path = ['../Proc. Imagenes/datasets/potholed', '../Proc. Imagenes/datasets/bubbly', '../Proc. Imagenes/datasets/zigzag']

#Carga de datasets
dataset_photoled = load_dataset(datasets_path[0])
dataset_bubbly = load_dataset(datasets_path[1])
dataset_zigzag = load_dataset(datasets_path[2])

#Guardar imagenes en array
photoled_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2BGR) for sample in dataset_photoled['train']]
bubbly_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2BGR) for sample in dataset_bubbly['train']]
zigzag_array = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2BGR) for sample in dataset_zigzag['train']]

#Procesamiento de imagenes
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
    num_filters = 4
    ksize = 35  # The local area to evaluate // Kernel size
    sigma = 3.0  # Larger Values produce more edges // Standard deviation of the gaussian envelope
    lambd = 10.0 # Wavelength of the sinusoidal factor
    gamma = 0.5 #
    psi = 0  # Offset value - lower generates cleaner results
    
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        angulos.append(theta)
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters, angulos

filters, angulos = create_gaborfilter()

#Aplicar filtro gabor
def apply_gaborfilter(img, filters):
    filtered_img = []
    for i in range(len(filters)):
        filtered_img.append(cv.filter2D(img, cv.CV_8UC3, filters[i]))
    return filtered_img

def extract_features(img):
    features = []

    features.append(np.mean(img))
    features.append(np.std(img))
    features.append(np.var(img))
    features.append(np.median(img))
    #features.append(entropy(img, disk(5)))

    return features

def calc_means(features_array):
    features_array = np.array(features_array)  # Convertir la lista a un array de NumPy
    return np.mean(features_array, axis=0)  # Calcular la media por columnas

#Preprocesamiento de imagenes
for i in range(len(photoled_array)):
    photoled_array_processed[i] = img_aumentobrillo(photoled_array[i])

for i in range(len(bubbly_array)):
    bubbly_array_processed[i] = img_aumentobrillo(bubbly_array[i])

for i in range(len(zigzag_array)):
    zigzag_array_processed[i] = img_aumentobrillo(zigzag_array[i])

#Aplicar filtro gabor
filtered_photoled = photoled_array.copy()
filtered_bubbly = photoled_array.copy()
filtered_zigzag = photoled_array.copy()

for i in range(len(photoled_array_processed)):
    filtered_photoled[i] = apply_gaborfilter(photoled_array_processed[i], filters[1])

for i in range(len(bubbly_array_processed)):
    filtered_bubbly[i] = apply_gaborfilter(bubbly_array_processed[i], filters[3])

for i in range(len(zigzag_array_processed)):
    filtered_zigzag[i] = apply_gaborfilter(zigzag_array_processed[i], filters[3])

#Extraer features
features_photoled = photoled_array.copy()
features_bubbly = photoled_array.copy()
features_zigzag = photoled_array.copy()

for i in range(len(filtered_photoled)):
    features_photoled[i] = extract_features(filtered_photoled[i])

for i in range(len(filtered_bubbly)):
    features_bubbly[i] = extract_features(filtered_bubbly[i])

for i in range(len(filtered_zigzag)):
    features_zigzag[i] = extract_features(filtered_zigzag[i])

#Calcular medias
print(features_photoled)
means_photoled = calc_means(features_photoled)
means_bubbly = calc_means(features_bubbly)
means_zigzag = calc_means(features_zigzag)

print(f"PROMEDIOS 1: {means_photoled}")
print(f"PROMEDIOS 2: {means_bubbly}")
print(f"PROMEDIOS 3: {means_zigzag}")

print(np.rad2deg(angulos))


"""
cv.imshow('photoled', photoled_array[1])
cv.waitKey(0)
cv.imshow('photoled', photoled_array_processed[1])
cv.waitKey(0)
cv.destroyAllWindows()
"""