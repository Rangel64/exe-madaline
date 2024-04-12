import cv2
import os
import numpy as np
import pandas as pd

# Pasta contendo as imagens
pasta = 'all/archive/test'

# Lista para armazenar os pixels das imagens
pixels = []

# Carrega cada imagem da pasta e adiciona os pixels à lista
for arquivo in os.listdir(pasta):
    if arquivo.endswith('.jpg') or arquivo.endswith('.png'):  # Verifica se o arquivo é uma imagem
        caminho_imagem = os.path.join(pasta, arquivo)
        imagem = cv2.imread(caminho_imagem)
        # Converte a imagem para escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        # Redimensiona a imagem para 1 dimensão
        pixels_imagem = imagem_cinza.flatten()
        pixels.append(pixels_imagem)

# Cria a matriz com os pixels das imagens
matriz_pixels = np.array(pixels)

# Exibe a matriz
print(matriz_pixels)

matrizIndices = np.asarray(matriz_pixels)
dfIndices = pd.DataFrame(data=matrizIndices)
        
dfIndices.to_csv('all/archive/test/targets/test.csv', index=False)
