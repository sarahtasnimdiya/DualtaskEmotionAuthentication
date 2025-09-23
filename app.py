# app.py
import base64, io, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch
import torchvision.transforms as T

from model import MultiTaskEffNetB0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = MultiTaskEffNetB0(num_emotions=len(EMOTION_LABELS))

    weights_path = "./weights/final_multitask_model90.pth"
    state = torch.load(weights_path, map_location=DEVICE)

    # Filter out unexpected keys (e.g. log_var_e, log_var_a)
    filtered_state = {k: v for k, v in state.items() if k in model.state_dict()}

    model.load_state_dict(filtered_state, strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded on", DEVICE)

@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            b64 = payload.get("image", "")
            if not b64:
                continue
            if b64.startswith("data:image"):
                b64 = b64.split(",",1)[1]

            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emo_logits, auth_logits = model(x)
                emo_probs = torch.softmax(emo_logits,1).cpu().numpy()[0]
                auth_prob = torch.sigmoid(auth_logits.squeeze()).cpu().item()

            top_idx = int(torch.argmax(emo_logits,1).cpu().item())
            emotion_label = EMOTION_LABELS[top_idx]

            # Decide genuine/fake label
            auth_label = "Genuine" if auth_prob >= 0.5 else "Fake"

            resp = {
                "emotion": {
                    "label": emotion_label,
                    "score": float(emo_probs[top_idx]),
                    "probs": {l: float(p) for l,p in zip(EMOTION_LABELS, emo_probs)}
                },
                "authenticity": {
                    "label": auth_label,
                    "genuine_prob": float(auth_prob)
                }
            }

            # Debug log
            print(f"🧠 Prediction → Emotion: {emotion_label} ({emo_probs[top_idx]:.2f}), "
                  f"Authenticity: {auth_label} ({auth_prob:.2f})")

            await websocket.send_text(json.dumps(resp))
    except WebSocketDisconnect:
        print("❌ Client disconnected")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
