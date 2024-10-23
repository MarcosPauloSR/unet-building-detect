from torchvision import transforms
from PIL import Image
from main import UNet
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Definição das transformações
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"O arquivo {image_path} não foi encontrado.")
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image)
    image = image.unsqueeze(0)  # Adiciona a dimensão do batch
    return image

# Definição do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Está sendo utilizado o dispositivo:', device)

# Instanciação do modelo
model = UNet(in_channels=3, out_channels=1)
model = model.to(device)

# Carregamento dos pesos
model.load_state_dict(torch.load('buildings.pth', map_location=device))

# Definir o modelo em modo de avaliação
model.eval()

def predict_image(image_path):
    # Carrega e pré-processa a imagem
    image = load_image(image_path)
    image = image.to(device)
    
    # Desabilita o cálculo de gradientes
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        # Binarização da saída
        predicted_mask = (output > 0.5).float()
    
    return predicted_mask.cpu().squeeze(0)  # Remove a dimensão do batch

# Função de denormalização
def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image_tensor.cpu().clone()
    image = image * std + mean
    image = image.clamp(0, 1)
    return image

def visualize_prediction(image_path, title):
    # Carrega e pré-processa a imagem
    image = load_image(image_path)
    image_denorm = denormalize(image.squeeze(0))
    image_np = image_denorm.permute(1, 2, 0).numpy()
    
    # Obtém a predição
    predicted_mask = predict_image(image_path)
    mask_np = predicted_mask.squeeze().numpy()
    
    # Plotagem
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Máscara Predita')
    plt.axis('off')
    
    plt.show()

def save_predicted_mask(image_path, save_path):
    predicted_mask = predict_image(image_path)
    # Converte o tensor para PIL Image
    mask_image = transforms.ToPILImage()(predicted_mask)
    mask_image.save(save_path)

# Diretório das imagens
image_directory = './val/images/'

# Verifica Imagens Disponíveis no diretório
available_images = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

for img in available_images:
    image_path = os.path.join(image_directory, img)
    visualize_prediction(image_path, img)
    save_predicted_mask(image_path, f'./val/masks/mask_{img}')
