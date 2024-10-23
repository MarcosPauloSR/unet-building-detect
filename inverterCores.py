import os
from PIL import Image, ImageChops

# Caminho para a pasta com as imagens
pasta_imagens = 'C:/Users/Marcos Paulo/Downloads/archive (1)/alabama/mask/'
pasta = 'C:/Users/Marcos Paulo/Desktop/Unet/train/masks/'

# Lista todos os arquivos na pasta
arquivos = os.listdir(pasta_imagens)

# Filtra apenas arquivos de imagem (opcional)
extensoes_imagem = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

for arquivo in arquivos:
    print('Arquivo: ', arquivo)
    
    if arquivo.lower().endswith(extensoes_imagem):
        # Caminho completo para o arquivo de imagem
        caminho_imagem = os.path.join(pasta_imagens, arquivo)

        # Abre a imagem
        img = Image.open(caminho_imagem)

        # Inverte as cores da imagem
        inv_img = ImageChops.invert(img)

        # Salva a imagem invertida (pode alterar o caminho se desejar)
        nome_arquivo, extensao = os.path.splitext(arquivo)
        nome_novo_arquivo = f"{nome_arquivo}{extensao}"
        caminho_novo_arquivo = os.path.join(pasta, nome_novo_arquivo)

        inv_img.save(caminho_novo_arquivo)
