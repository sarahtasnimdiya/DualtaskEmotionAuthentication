# test_inference.py
import torch
from PIL import Image
import torchvision.transforms as T
from model import MultiTaskEffNetB0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = MultiTaskEffNetB0(num_emotions=len(EMOTION_LABELS))
state = torch.load("weights/final_multitask_model90.pth", map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

img = Image.open("test.JPG").convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    emo_logits, auth_logits = model(x)
    print("Emotion:", EMOTION_LABELS[emo_logits.argmax(1).item()])
    print("Auth prob:", torch.sigmoid(auth_logits).item())
