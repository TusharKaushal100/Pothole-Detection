import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# -------- Config --------
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 12
LR = 0.001

# -------- Transforms --------
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor()
# ])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Dataset --------
dataset = datasets.ImageFolder(
    root="dataset",
    transform=train_transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------- Model --------
model = SimpleCNN()

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([1.5])
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-4
)

# -------- Training Loop --------
for epoch in range(EPOCHS):
    total_loss = 0.0
    print("Starting epoch", epoch+1)
    number = 0
    for images, labels in loader:
        print("Processing batch", number)

        number += 1
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -------- Save Model --------
torch.save(model.state_dict(), "pothole_cnn.pth")
print("Model saved as pothole_cnn.pth")
