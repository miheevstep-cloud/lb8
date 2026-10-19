import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================================
# НАСТРОЙКИ (ПОДСТАВЬТЕ СВОИ ЗНАЧЕНИЯ)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
IMAGE_SIZE = 128          # Размер изображений (как в ЛР №6)
TRAIN_RATIO = 0.8         # Доля обучающей выборки
BATCH_SIZE = 32           # Размер батча
EPOCHS = 100              # Количество эпох
DROPOUT = 0.3             # Вероятность Dropout
WEIGHT_DECAY = 1e-4       # L2-регуляризация
LEARNING_RATE = 0.001     # Скорость обучения
# ==========================================

# 1. ПОДГОТОВКА ДАННЫХ С АУГМЕНТАЦИЕЙ
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Загружаем датасет дважды с разными трансформациями
full_dataset_train = ImageFolder(DATASET_PATH, transform=train_transform)
full_dataset_test = ImageFolder(DATASET_PATH, transform=test_transform)

train_size = int(TRAIN_RATIO * len(full_dataset_train))
test_size = len(full_dataset_train) - train_size

# Фиксируем seed, чтобы разделение на train/test совпадало для обоих трансформов
generator = torch.Generator().manual_seed(42)
train_dataset, _ = random_split(full_dataset_train, [train_size, test_size], generator=generator)
_, test_dataset = random_split(full_dataset_test, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = full_dataset_train.classes
NUM_CLASSES = len(class_names)
print(f"Классы: {class_names}")
print(f"Количество классов: {NUM_CLASSES}")

# 2. МОДЕЛЬ С DROPOUT
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier(NUM_CLASSES).to(device)

# 3. ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 4. ЦИКЛ ОБУЧЕНИЯ С НАКОПЛЕНИЕМ МЕТРИК
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []
best_val_acc = 0.0
best_epoch = 0

print("\n=== Начало обучения ===")
for epoch in range(EPOCHS):
    # --- Режим обучения ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    train_preds, train_labels = [], []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        train_preds.extend(outputs.argmax(1).detach().cpu().numpy())
        train_labels.extend(labels.detach().cpu().numpy())
        
    train_loss = running_loss / total
    train_acc = correct / total
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    
    # --- Режим проверки ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            val_preds.extend(outputs.argmax(1).detach().cpu().numpy())
            val_labels.extend(labels.detach().cpu().numpy())
            
    val_loss = running_loss / total
    val_acc = correct / total
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    # Сохраняем метрики
    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc); val_accs.append(val_acc)
    train_f1s.append(train_f1); val_f1s.append(val_f1)
    
    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_model.pth"))
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Эпоха {epoch+1}/{EPOCHS} | "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")

print(f"\nЛучшая эпоха: {best_epoch}, val accuracy: {best_val_acc:.4f}")

# 5. ГРАФИКИ ОБУЧЕНИЯ
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs_range, train_accs, label='Обучение')
axes[0].plot(epochs_range, val_accs, label='Проверка')
axes[0].set_title('Accuracy по эпохам')
axes[0].set_xlabel('Эпоха'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_losses, label='Обучение')
axes[1].plot(epochs_range, val_losses, label='Проверка')
axes[1].set_title('Loss по эпохам')
axes[1].set_xlabel('Эпоха'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, train_f1s, label='Обучение')
axes[2].plot(epochs_range, val_f1s, label='Проверка')
axes[2].set_title('F1-score по эпохам')
axes[2].set_xlabel('Эпоха'); axes[2].set_ylabel('F1')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. ФИНАЛЬНЫЕ МЕТРИКИ И МАТРИЦА ОШИБОК
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_model.pth"), map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(1)
        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\n=== Подробный отчёт ===")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (Лучшая модель)')
plt.tight_layout()
plt.show()

# 7. ТЕКСТОВЫЙ АНАЛИЗ
print("\n=== Анализ результатов ===")
gap = train_accs[-1] - val_accs[-1]
if gap > 0.15:
    print(f"⚠️ Признак переобучения: разрыв train/val accuracy = {gap:.4f}")
else:
    print(f"✅ Переобучение отсутствует или минимально: разрыв train/val accuracy = {gap:.4f}")

cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
max_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
true_idx, pred_idx = max_idx
max_confusion = cm_copy[true_idx, pred_idx]
if max_confusion > 0:
    print(f"\nЧаще всего путаются классы:")
    print(f"  Истинный: '{class_names[true_idx]}'")
    print(f"  Предсказанный: '{class_names[pred_idx]}'")
    print(f"  Количество ошибок: {max_confusion}")
else:
    print("\n✓ Ошибок классификации не обнаружено.")