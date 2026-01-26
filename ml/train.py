import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# -------- Config --------
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LR = 0.001

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# -------- Dataset --------
dataset = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------- Model --------
model = SimpleCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------- Training Loop --------
for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in loader:
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
