import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

# Устройства: CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к данным
data_dir = {
    'train': 'train',
    'val': 'val',
    'test': 'test',
}

# Гиперпараметры
batch_size = 32
num_epochs = 5
learning_rate = 0.001

# Трансформации данных (включена расширенная аугментация для TRAIN)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(224, padding=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Загрузка данных
image_datasets = {x: datasets.ImageFolder(data_dir[x], transform=data_transforms[x]) for x in data_dir}
class_names = image_datasets['train'].classes

# Подсчет количества примеров в каждом классе
class_counts = [1341, 3875]  # Замените на распределение ваших данных
class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device)

# Создание WeightedRandomSampler для сбалансированного обучения
class_weights_sampler = [1.0 / class_counts[cls] for cls in image_datasets['train'].targets]
sample_weights = torch.tensor(class_weights_sampler, dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader для тренировочного набора с WeightedRandomSampler
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False),
}

# Загрузка модели EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(class_names))  # 2 класса: NORMAL и PNEUMONIA
model = model.to(device)

# Определение Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

# Используем Focal Loss с учетом весов классов
criterion = FocalLoss(alpha=class_weights, gamma=2)

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Функция для расчёта метрик
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # NORMAL: 0, PNEUMONIA: 1
    precision_normal = precision_score(all_labels, all_preds, pos_label=0)
    recall_normal = recall_score(all_labels, all_preds, pos_label=0)
    precision_pneumonia = precision_score(all_labels, all_preds, pos_label=1)
    recall_pneumonia = recall_score(all_labels, all_preds, pos_label=1)

    return cm, precision_normal, recall_normal, precision_pneumonia, recall_pneumonia

# Вывод матрицы ошибок
def print_metrics(cm, precision_normal, recall_normal, precision_pneumonia, recall_pneumonia):
    print("Confusion Matrix:")
    print(cm)
    print(f"Healthy - Precision: {precision_normal:.4f}, Recall: {recall_normal:.4f}")
    print(f"Pneumonia - Precision: {precision_pneumonia:.4f}, Recall: {recall_pneumonia:.4f}")

# Обучение модели
def train_model(model, criterion, optimizer, num_epochs=10):
    history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Обучение
        model.train()
        running_loss = 0.0
        with tqdm(total=len(dataloaders['train']), desc=f'Train Epoch {epoch+1}') as pbar:
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)

        epoch_loss = running_loss / len(dataloaders['train'])
        history['train_loss'].append(epoch_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(dataloaders['val'])
        history['val_loss'].append(val_loss)

        # Оценка на тестовом наборе
        cm, precision_normal, recall_normal, precision_pneumonia, recall_pneumonia = evaluate_model(model, dataloaders['test'])
        print_metrics(cm, precision_normal, recall_normal, precision_pneumonia, recall_pneumonia)

        # Сохранение метрик
        history['precision'].append(precision_pneumonia)
        history['recall'].append(recall_pneumonia)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs (EfficientNet)')
    plt.legend()

    # График Precision и Recall
    plt.subplot(1, 2, 2)
    plt.plot(history['precision'], label='Precision (Pneumonia)')
    plt.plot(history['recall'], label='Recall (Pneumonia)')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Precision and Recall over Epochs (EfficientNet)')
    plt.legend()

    plt.show()

    return model

# Тренируем модель
model = train_model(model, criterion, optimizer, num_epochs)
