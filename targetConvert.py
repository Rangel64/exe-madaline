import string
import pandas as pd

# Lista de letras do alfabeto
letras = list(string.ascii_lowercase)

# Dicionário para mapear cada letra para seu vetor de posições
vetores = {}
for i, letra in enumerate(letras):
    vetor = [-1] * 26
    vetor[i] = 1
    vetores[letra] = vetor

