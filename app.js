const video = document.getElementById("video"),
      overlay = document.getElementById("overlay"),
      ctx = overlay.getContext("2d"),
      status = document.getElementById("status");

let stream, ws, sending = false;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
  } catch (err) {
    console.error("Camera error:", err);
    status.innerText = "Camera access denied";
  }
}

function startWS() {
  ws = new WebSocket("wss://dualtaskemotionauthentication.onrender.com/ws/predict");

  ws.onopen = () => {
    console.log("✅ WebSocket connected");
    status.innerText = "Connected";
  };

  ws.onclose = () => {
    console.log("❌ WebSocket disconnected");
    status.innerText = "Disconnected";
    sending = false;
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    status.innerText = "Error";
  };

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.error) {
        console.warn("⚠️ Backend:", data.error);
        status.innerText = "No face detected";
        return;
      }
      drawOverlay(data);
      status.innerText = "Streaming";
    } catch (e) {
      console.error("Parse error:", e);
    }
  };
}

function capture() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return c.toDataURL("image/jpeg", 0.7);
}

async function loop() {
  while (sending && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ image: capture() }));
    await new Promise(r => setTimeout(r, 200)); // ~5 fps
  }
}

function drawOverlay(r) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // ✅ Draw face box if available
  if (r.face_box) {
    ctx.strokeStyle = "lime"; // green box
    ctx.lineWidth = 3;
    ctx.strokeRect(r.face_box.x, r.face_box.y, r.face_box.w, r.face_box.h);
  }

  ctx.font = "20px Arial";
  ctx.textBaseline = "top";

  // Emotion (white)
  ctx.fillStyle = "#fff";
  ctx.fillText(
    `Emotion: ${r.emotion.label} (${(r.emotion.score * 100).toFixed(1)}%)`,
    12, 20
  );

  // Authenticity (green if genuine, red if fake)
  let color = r.authenticity.label === "Genuine" ? "#0f0" : "#f00";
  ctx.fillStyle = color;
  ctx.fillText(
    `Authenticity: ${r.authenticity.label} (${(r.authenticity.genuine_prob * 100).toFixed(1)}%)`,
    12, 50
  );
}

document.getElementById("startBtn").onclick = async () => {
  await startCamera();
  startWS();
  sending = true;
  loop();
  status.innerText = "Streaming";
};

document.getElementById("stopBtn").onclick = () => {
  sending = false;
  if (ws) ws.close();
  if (stream) stream.getTracks().forEach(t => t.stop());
  status.innerText = "Stopped";
};
