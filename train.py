import torch
from main import dataloader, model, criterion, optimizer, device

def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)
    print('Iniciando o treinamento...')
    print('Está sendo utilizado o dispositivo:', device)
    print('Número de épocas:', num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        # model.train()

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        with open('loss.txt', 'a') as f:
            f.write(f'Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n')

    return model

# Inicia o treinamento da rede
trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=25, device=device)

# Salvar o modelo treinado
torch.save(trained_model.state_dict(), 'buildings.pth')
