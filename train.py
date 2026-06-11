import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import CustomCNN

def treinar_e_avaliar_modelo(output_dir, epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config] Executando o modelo no dispositivo: {device}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")

    for d in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(d) or len(os.listdir(d)) == 0:
            print(f"[Erro] A pasta {d} está vazia ou não existe.")
            return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    print(f"[Dataset] Classes detectadas: {class_names}")

    model = CustomCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n--- Iniciando o Treinamento da CNN ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / len(valid_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Época [{epoch+1}/{epochs}] -> Treino Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% "
              f"|| Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

    print("\n--- Avaliação Final no Conjunto de Teste ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n Matriz de Confusão:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n Relatório de Classificação:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))