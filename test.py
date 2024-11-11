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
    # transforms.Grayscale(num_output_channels=3),
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

def load_mask(mask_path):
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"O arquivo {mask_path} não foi encontrado.")
    mask = Image.open(mask_path).convert("L")  # Converte para escala de cinza
    mask = transforms.Resize((256, 256))(mask)
    mask = transforms.ToTensor()(mask)
    mask = (mask > 0.5).float()  # Binariza a máscara
    mask = mask.unsqueeze(0)  # Adiciona a dimensão do batch
    return mask

def dice_coefficient(predicted, target):
    smooth = 1.0  # Para evitar divisão por zero
    predicted_flat = predicted.view(-1)
    target_flat = target.view(-1)
    intersection = (predicted_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (predicted_flat.sum() + target_flat.sum() + smooth)

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
    
    return predicted_mask.cpu()  # Remove para CPU

# Função de denormalização
def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image_tensor.cpu().clone()
    image = image * std + mean
    image = image.clamp(0, 1)
    return image

def calculate_precision(predicted, target):
    predicted_flat = predicted.view(-1)
    target_flat = target.view(-1)
    TP = ((predicted_flat == 1) & (target_flat == 1)).sum().float()
    FP = ((predicted_flat == 1) & (target_flat == 0)).sum().float()
    precision = TP / (TP + FP + 1e-7)  # Adiciona um pequeno valor para evitar divisão por zero
    return precision.item()

def calculate_recall(predicted, target):
    predicted_flat = predicted.view(-1)
    target_flat = target.view(-1)
    TP = ((predicted_flat == 1) & (target_flat == 1)).sum().float()
    FN = ((predicted_flat == 0) & (target_flat == 1)).sum().float()
    recall = TP / (TP + FN + 1e-7)
    return recall.item()

# função de log que salva em um arquivo txt com nome definido
def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def visualize_prediction(image_path, mask_path, title):
    # Carrega e pré-processa a imagem
    image = load_image(image_path)
    image_denorm = denormalize(image.squeeze(0))
    image_np = image_denorm.permute(1, 2, 0).numpy()
    
    # Carrega a máscara de validação
    ground_truth_mask = load_mask(mask_path)
    ground_truth_mask_np = ground_truth_mask.squeeze().numpy()
    
    # Obtém a predição
    predicted_mask = predict_image(image_path)
    mask_np = predicted_mask.squeeze().numpy()
    
    # Computa as métricas
    dice_score = dice_coefficient(predicted_mask, ground_truth_mask)
    precision = calculate_precision(predicted_mask, ground_truth_mask)
    recall = calculate_recall(predicted_mask, ground_truth_mask)
    
    log_to_file('log_4Camadas.txt', f'Coeficiente de Dice para {title}: {dice_score:.4f}')
    log_to_file('log_4Camadas.txt', f'Precisão para {title}: {precision:.4f}')
    log_to_file('log_4Camadas.txt', f'Recall para {title}: {recall:.4f}')
    
    # Plotagem
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask_np, cmap='gray')
    plt.title('Máscara de Validação')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Máscara Predita')
    plt.axis('off')
    
    plt.show()


def save_predicted_mask(image_path, save_path):
    predicted_mask = predict_image(image_path)
    # Converte o tensor para PIL Image
    mask_image = transforms.ToPILImage()(predicted_mask.squeeze(0))
    mask_image.save(save_path)

# Diretório das imagens e máscaras
image_directory = './val/images/'
mask_directory = './val/val_masks/'

# Verifica Imagens Disponíveis no diretório
available_images = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

for img in available_images:
    image_path = os.path.join(image_directory, img)
    mask_path = os.path.join(mask_directory, img)
    visualize_prediction(image_path, mask_path, img)
    save_predicted_mask(image_path, f'./val/masks/mask_{img}')
