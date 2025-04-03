import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# Пути к данным
train_dir = "train"
val_dir = "val"
test_dir = "test"

# Загрузка предобученной модели и процессора
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)

# Замена классификатора
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)

# Использование GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Аугментация данных
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224, padding=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Загрузка данных
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# Взвешивание классов
class_counts = np.bincount([label for _, label in train_dataset.samples])
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader с WeightedRandomSampler
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)

# Оптимизация
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Функция вычисления метрик
def calculate_metrics(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Переводим в int для вычисления точности
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()

    # Получаем отчет
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    return precision, recall, f1, accuracy, report

# Обучение
def train(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc="Обучение", leave=False)  # Прогресс-бар

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Обновление прогресс-бара
        progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    return running_loss / len(loader), correct / total

# Оценка
def evaluate(model, loader):
    precision, recall, f1, accuracy, report = calculate_metrics(model, loader)
    return precision, recall, f1, accuracy, report

# Обучение модели
epochs = 4
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader)
    train_precision, train_recall, train_f1, train_accuracy, _ = evaluate(model, train_loader)
    val_precision, val_recall, val_f1, val_accuracy, val_report = evaluate(model, val_loader)
    test_precision, test_recall, test_f1, test_accuracy, test_report = evaluate(model, test_loader)

    # Вывод результатов после каждой эпохи
    print(f"\nЭпоха {epoch + 1}")
    print(f"Обучение: Потеря: {train_loss:.4f}, Точность: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Валидация: Точность: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")
    print(f"Тест: Точность: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")

# Вывод финальных метрик на тестовом наборе
print("\nОкончательные метрики на тестовом наборе:")
print(test_report)
