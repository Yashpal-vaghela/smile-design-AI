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

  drawSmileBoundingBox(points);

  // === SMILE PERCENTAGE ===
  const smilePercent = calculateSmilePercentage(points);

  document.getElementById("smile-percentage").innerText =
    "Smile Score: " + smilePercent + "%";

  const teeth = detectTeethCount(points);
  document.getElementById("teeth-count").innerText =
    "Teeth Visible: " + teeth;

  // Optionally draw points
  ctx.fillStyle = "cyan";
  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x * overlay.width, p.y * overlay.height, 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function calculateSmilePercentage(points) {
  // MOUTH landmark indexes
  const upperLip = points[13]; // top lip
  const lowerLip = points[14]; // bottom lip
  const leftCorner = points[61]; // left mouth corner
  const rightCorner = points[291]; // right mouth corner

  // Convert to pixel positions
  const UL = { x: upperLip.x * overlay.width, y: upperLip.y * overlay.height };
  const LL = { x: lowerLip.x * overlay.width, y: lowerLip.y * overlay.height };
  const LC = { x: leftCorner.x * overlay.width, y: leftCorner.y * overlay.height };
  const RC = { x: rightCorner.x * overlay.width, y: rightCorner.y * overlay.height };

  // 1️⃣ Mouth Openness (vertical distance)
  let mouthOpen = Math.abs(LL.y - UL.y);

  // 2️⃣ Smile Width (horizontal distance)
  let smileWidth = Math.abs(RC.x - LC.x);

  // Normalize values for better percentage output
  let normalizedOpen = Math.min(1, mouthOpen / 60);
  let normalizedWidth = Math.min(1, smileWidth / 200);

  // 3️⃣ Teeth Visibility (brightness inside bounding box)
  let tv = estimateTeethVisibility(points);

  // Final smile %
  let smilePercent = (normalizedOpen * 0.3 + normalizedWidth * 0.3 + tv * 0.4) * 100;

  return Math.round(smilePercent);
}

function estimateTeethVisibility(points) {
  let xs = [], ys = [];

  MOUTH_LANDMARK_INDEXES.forEach(i => {
    let p = points[i];
    xs.push(p.x * overlay.width);
    ys.push(p.y * overlay.height);
  });

  let minX = Math.min(...xs), minY = Math.min(...ys);
  let maxX = Math.max(...xs), maxY = Math.max(...ys);

  let w = maxX - minX, h = maxY - minY;

  // Sample mouth pixels for whiteness check
  let imgData = ctx.getImageData(minX, minY, w, h);
  let data = imgData.data;

  let whitePixels = 0;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];

    // If pixel is bright = likely tooth
    if (r + g + b > 600) {
      whitePixels++;
    }
  }

  // Normalize
  let whiteness = whitePixels / (w * h);
  whiteness = Math.min(1, whiteness * 4);

  return whiteness; // 0 - 1
}

function detectTeethCount(points) {
  let xs = [], ys = [];

  // Get mouth landmark coordinates
  MOUTH_LANDMARK_INDEXES.forEach(i => {
    let p = points[i];
    xs.push(p.x * overlay.width);
    ys.push(p.y * overlay.height);
  });

  let minX = Math.min(...xs), minY = Math.min(...ys);
  let maxX = Math.max(...xs), maxY = Math.max(...ys);

  let w = Math.max(1, maxX - minX);
  let h = Math.max(1, maxY - minY);

  // Crop only the UPPER HALF of mouth → prevents lower teeth from confusing logic
  let cropY = minY;
  let cropH = Math.max(1, h * 0.55);

  let img = ctx.getImageData(minX, cropY, w, cropH).data;
  let totalPixels = w * cropH;

  let brightPixels = 0;
  let whiteClusters = 0;

  // COMPUTE AVERAGE LIP COLOR (darker zone)
  let sampleCount = 50;
  let lipAvg = 0;
  for (let i = 0; i < sampleCount; i++) {
    let idx = Math.floor(Math.random() * img.length / 4) * 4;
    let brightnessLip = (img[idx] + img[idx+1] + img[idx+2]) / 3;
    lipAvg += brightnessLip;
  }
  lipAvg /= sampleCount;

  // Teeth must be SIGNIFICANTLY brighter than lips  
  let threshold = lipAvg + 40;

  // Pixel groups (teeth clusters)
  let clusterCount = 0;
  let clusterActive = false;

  for (let i = 0; i < img.length; i += 4) {
    let r = img[i], g = img[i + 1], b = img[i + 2];
    let brightness = (r + g + b) / 3;

    let isWhite = brightness > threshold && r > g && g * 0.9 > b;

    if (isWhite) {
      brightPixels++;

      // count cluster as sequence of white pixels separated by dark pixels
      if (!clusterActive) {
        clusterActive = true;
        clusterCount++;
      }
    } else {
      clusterActive = false;
    }
  }

  let whiteRatio = brightPixels / totalPixels;

  // DEBUG (optional):
  // console.log("ratio:", whiteRatio, "clusters:", clusterCount);

  // --- RULES ---
  // 8 teeth = wide smile AND many clusters
  if (whiteRatio > 0.065 && clusterCount >= 6) return 8;

  // 6 teeth = medium smile
  if (whiteRatio > 0.040 && clusterCount >= 4) return 6;

  // 4 teeth = visible but smaller region
  if (whiteRatio > 0.025 && clusterCount >= 3) return 4;

  // 2 teeth = small smile
  if (whiteRatio > 0.015) return 2;

  return 0;
}

// MOUTH LANDMARKS (Mediapipe Indexes)
const MOUTH_LANDMARK_INDEXES = [
  78,  95,  88,  178, 87,  14, 
  317, 402, 318, 324, 308,  61, 
  291, 185, 40,  39,  37
];

function drawSmileBoundingBox(points) {
  let xs = [];
  let ys = [];

  // Collect mouth landmark coordinates
  MOUTH_LANDMARK_INDEXES.forEach(i => {
    let p = points[i];
    xs.push(p.x * overlay.width);
    ys.push(p.y * overlay.height);
  });

  // Get min/max values —> bounding box
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);

  let width = maxX - minX;
  let height = maxY - minY;

  // Draw bounding box
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 4;
  ctx.strokeRect(minX - 10, minY - 10, width + 20, height + 20);
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
