import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Load Model --------
model = SimpleCNN()
model.load_state_dict(torch.load("pothole_cnn.pth"))
model.eval()

# -------- Load Image --------
img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

# -------- Predict --------
with torch.no_grad():
    output = model(img)
    prob = torch.sigmoid(output)

print("Pothole probability:", prob.item())
