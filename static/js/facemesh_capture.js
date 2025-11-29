const video = document.getElementById("cam");
const overlay = document.getElementById("overlay");
const btnCapture = document.getElementById("btn-capture");
const ctx = overlay.getContext("2d");

import { FilesetResolver, FaceLandmarker } 
from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let faceLandmarker = null;
let cameraStarted = false;

// BEST camera selection logic
async function startCamera() {
  try {
    let constraints = {
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    let stream = await navigator.mediaDevices.getUserMedia(constraints);

    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);

    video.style.display = "block";
    overlay.style.display = "block";

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    return true;

  } catch (error) {
    console.warn("Front camera failed, trying default camera...");

    try {
      let fallbackStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      video.srcObject = fallbackStream;
      await new Promise(res => video.onloadedmetadata = res);

      video.style.display = "block";
      overlay.style.display = "block";

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;

      return true;

    } catch (err) {
      alert("Camera Permission Denied or No Camera Found!");
      console.error(err);
      return false;
    }
  }
}

async function initFaceMesh() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });

  console.log("FaceMesh Model Loaded");
}

function draw(points) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.drawImage(video, 0, 0, overlay.width, overlay.height);

  if (!points) return;

  ctx.fillStyle = "cyan";
  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x * overlay.width, p.y * overlay.height, 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function loop() {
  if (!faceLandmarker) return requestAnimationFrame(loop);

  const res = faceLandmarker.detectForVideo(video, performance.now());

  if (res.faceLandmarks && res.faceLandmarks.length > 0) {
    draw(res.faceLandmarks[0]);
  } else {
    draw(null);
  }

  requestAnimationFrame(loop);
}

btnCapture.addEventListener("click", async () => {
  if (!cameraStarted) {
    cameraStarted = true;

    btnCapture.innerText = "Camera Running...";

    const ok = await startCamera();
    
    if (ok) {
      await initFaceMesh();
      loop();
    }
  }
});
