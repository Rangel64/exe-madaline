import os
import shutil
import numpy as np
import pandas as pd
import string
from PIL import Image

url = 'archive/test'
lista1 = os.listdir(url)

lista1 = sorted(lista1, key=str.lower)
cont = 0

targets = []
for i in range(len(lista1)):
    lista = os.listdir(url+'/'+lista1[i])
    # for j in range(len(lista)):
    for j in range(4):
        image_path = url+'/'+lista1[i]+'/'+lista[j]
        image = Image.open(image_path)
        resized_image = image.resize((50, 50))
        resized_image.save(image_path)
        os.rename(url+'/'+lista1[i]+'/'+lista[j],url+'/'+lista1[i]+'/'+str(cont)+'.jpg')# Sobrescreve a imagem original com a nova versão redimensionada
        targets.append(lista1[i])
        cont = cont + 1

newUrl = 'all/archive/test'
for i in range(len(lista1)):
    lista = os.listdir(url+'/'+lista1[i])
    # for j in range(len(lista)):
    for j in range(4):
        shutil.copyfile(url+'/'+lista1[i]+'/'+lista[j], newUrl+'/'+lista[j])
        
#%%

# Lista de letras do alfabeto
letras = list(string.ascii_lowercase)

# Dicionário para mapear cada letra para seu vetor de posições
vetores = {}
for i, letra in enumerate(letras):
    vetor = [-1] * 26
    vetor[i] = 1
    vetores[letra] = vetor

finalTargets = []
for i in range(len(targets)):
    finalTargets.append(vetores[targets[i]])

matrizIndices = np.asarray(finalTargets)
# matrizIndices = np.transpose(matrizIndices)
dfIndices = pd.DataFrame(data=matrizIndices)
        
dfIndices.to_csv('all/archive/test/targets/targetsTest.csv', index=False)      