from PIL import Image
import os


lista_pastas_lilian = os.listdir('archive')
url = 'archive/'
for i in range(len(lista_pastas_lilian)):
    lista_pastas = os.listdir(url+lista_pastas_lilian[i])
    for j in range(len(lista_pastas)):
        lista_imagens = os.listdir(url + lista_pastas_lilian[i] + '/' + lista_pastas[j])
        for file_name in lista_imagens:
            path = url + lista_pastas_lilian[i] + '/' + lista_pastas[j]
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):

                # Abre a imagem e rotaciona ela em 90 graus
                with Image.open(os.path.join(path, file_name)) as img:
                    img = img.rotate(0, expand=True)

                    # Salva a imagem invertida com a resolução de 2250x4000
                    img = img.resize((64, 64))
                    img.save(os.path.join(path, f'inverted{file_name}')) 