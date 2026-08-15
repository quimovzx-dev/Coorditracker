import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ---------- constants ---------- */

const LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
  "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee",
  "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
];

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const MAX_LOGGED_FRAMES = 3000;

/* ---------- shared model loading ---------- */

let visionResolver = null;
let liveLandmarker = null;
let imageLandmarker = null;

async function getVisionResolver() {
  if (!visionResolver) visionResolver = await FilesetResolver.forVisionTasks(WASM_URL);
  return visionResolver;
}

async function getLiveLandmarker() {
  if (liveLandmarker) return liveLandmarker;
  const vision = await getVisionResolver();
  liveLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  return liveLandmarker;
}

async function getImageLandmarker() {
  if (imageLandmarker) return imageLandmarker;
  const vision = await getVisionResolver();
  imageLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "IMAGE",
    numPoses: 1,
  });
  return imageLandmarker;
}

const modelStatus = document.getElementById("modelStatus");
const modelStatusText = document.getElementById("modelStatusText");

getLiveLandmarker()
  .then(() => {
    modelStatus.classList.add("is-ready");
    modelStatusText.textContent = "MODEL READY";
  })
  .catch((err) => {
    modelStatusText.textContent = "MODEL FAILED TO LOAD";
    console.error(err);
  });

/* ---------- mode switching ---------- */

const modeButtons = document.querySelectorAll(".mode-btn");
const panelViews = {
  live: document.getElementById("view-live"),
  upload: document.getElementById("view-upload"),
};

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((b) => {
      b.classList.remove("is-active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("is-active");
    btn.setAttribute("aria-selected", "true");
    Object.values(panelViews).forEach((v) => v.classList.remove("is-active"));
    panelViews[btn.dataset.mode].classList.add("is-active");
  });
});

/* ---------- geometry helpers ---------- */

function getContainRect(containerW, containerH, mediaW, mediaH) {
  const containerRatio = containerW / containerH;
  const mediaRatio = mediaW / mediaH;
  let renderW, renderH, offsetX, offsetY;
  if (mediaRatio > containerRatio) {
    renderW = containerW;
    renderH = containerW / mediaRatio;
    offsetX = 0;
    offsetY = (containerH - renderH) / 2;
  } else {
    renderH = containerH;
    renderW = containerH * mediaRatio;
    offsetY = 0;
    offsetX = (containerW - renderW) / 2;
  }
  return { renderW, renderH, offsetX, offsetY };
}

function resizeCanvasToContainer(canvas, container) {
  const dpr = window.devicePixelRatio || 1;
  const w = container.clientWidth;
  const h = container.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

/* ---------- drawing ---------- */

function drawSkeleton(ctx, landmarks, rect, threshold, showConnections) {
  const { renderW, renderH, offsetX, offsetY } = rect;
  const toPixel = (lm) => [offsetX + lm.x * renderW, offsetY + lm.y * renderH];

  if (showConnections) {
    ctx.strokeStyle = "rgba(63, 169, 245, 0.85)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(63, 169, 245, 0.6)";
    ctx.shadowBlur = 6;
    PoseLandmarker.POSE_CONNECTIONS.forEach(({ start, end }) => {
      const a = landmarks[start];
      const b = landmarks[end];
      if (!a || !b) return;
      if ((a.visibility ?? 1) < threshold || (b.visibility ?? 1) < threshold) return;
      const [ax, ay] = toPixel(a);
      const [bx, by] = toPixel(b);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    });
  }

  ctx.shadowBlur = 0;
  landmarks.forEach((lm) => {
    const vis = lm.visibility ?? 1;
    if (vis < threshold) return;
    const [x, y] = toPixel(lm);
    ctx.beginPath();
    ctx.fillStyle = "#F2A93B";
    ctx.shadowColor = "rgba(242, 169, 59, 0.85)";
    ctx.shadowBlur = 8;
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.shadowBlur = 0;
}

function populateTable(tbody, countLabel, landmarks, threshold) {
  if (!landmarks) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No pose detected</td></tr>`;
    countLabel.textContent = "0 / 33 tracked";
    return;
  }
  let tracked = 0;
  const rows = landmarks.map((lm, i) => {
    const vis = lm.visibility ?? 1;
    if (vis >= threshold) tracked++;
    const cls = vis >= threshold ? "conf-high" : "conf-low";
    return `<tr>
      <td>${i}</td>
      <td>${LANDMARK_NAMES[i]}</td>
      <td>${lm.x.toFixed(3)}</td>
      <td>${lm.y.toFixed(3)}</td>
      <td class="${cls}">${vis.toFixed(2)}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
  countLabel.textContent = `${tracked} / 33 tracked`;
}

/* ---------- export helpers ---------- */

function downloadBlob(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function framesToCsv(frames) {
  const header = "frame,timestamp_ms,landmark_index,landmark_name,x,y,z,visibility";
  const lines = [header];
  frames.forEach((frame, fi) => {
    frame.landmarks.forEach((lm, i) => {
      lines.push(
        [fi, frame.t.toFixed(1), i, LANDMARK_NAMES[i], lm.x.toFixed(5), lm.y.toFixed(5), lm.z.toFixed(5), (lm.visibility ?? 1).toFixed(3)].join(",")
      );
    });
  });
  return lines.join("\n");
}

function landmarksToCsv(landmarks) {
  const header = "landmark_index,landmark_name,x,y,z,visibility";
  const lines = [header];
  landmarks.forEach((lm, i) => {
    lines.push(
      [i, LANDMARK_NAMES[i], lm.x.toFixed(5), lm.y.toFixed(5), lm.z.toFixed(5), (lm.visibility ?? 1).toFixed(3)].join(",")
    );
  });
  return lines.join("\n");
}

/* ================= LIVE MODE ================= */

const liveVideo = document.getElementById("liveVideo");
const liveCanvas = document.getElementById("liveCanvas");
const liveViewport = document.getElementById("liveViewport");
const livePlaceholder = document.getElementById("livePlaceholder");
const startCameraBtn = document.getElementById("startCameraBtn");
const toggleCameraBtn = document.getElementById("toggleCameraBtn");
const confThreshold = document.getElementById("confThreshold");
const confValue = document.getElementById("confValue");
const showConnections = document.getElementById("showConnections");
const recordLog = document.getElementById("recordLog");
const liveDataBody = document.getElementById("liveDataBody");
const liveTrackedCount = document.getElementById("liveTrackedCount");
const framesLoggedLabel = document.getElementById("framesLoggedLabel");
const exportLiveJson = document.getElementById("exportLiveJson");
const exportLiveCsv = document.getElementById("exportLiveCsv");

let liveStream = null;
let liveRunning = false;
let loggedFrames = [];

confThreshold.addEventListener("input", () => {
  confValue.textContent = Number(confThreshold.value).toFixed(2);
});

async function startCamera() {
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 960 } },
    });
  } catch (err) {
    alert("Camera access denied or unavailable.");
    console.error(err);
    return;
  }
  liveVideo.srcObject = liveStream;
  await liveVideo.play();
  livePlaceholder.classList.add("is-hidden");
  toggleCameraBtn.disabled = false;
  toggleCameraBtn.textContent = "Stop";
  liveRunning = true;
  requestAnimationFrame(liveLoop);
}

function stopCamera() {
  liveRunning = false;
  if (liveStream) {
    liveStream.getTracks().forEach((t) => t.stop());
    liveStream = null;
  }
  livePlaceholder.classList.remove("is-hidden");
  toggleCameraBtn.textContent = "Start";
  const ctx = liveCanvas.getContext("2d");
  ctx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);
}

startCameraBtn.addEventListener("click", startCamera);
toggleCameraBtn.addEventListener("click", () => {
  if (liveRunning) stopCamera();
  else startCamera();
});

async function liveLoop() {
  if (!liveRunning) return;
  if (liveVideo.readyState >= 2 && liveVideo.videoWidth > 0) {
    const landmarker = await getLiveLandmarker();
    const result = landmarker.detectForVideo(liveVideo, performance.now());
    const { w, h } = resizeCanvasToContainer(liveCanvas, liveViewport);
    const ctx = liveCanvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    const threshold = Number(confThreshold.value);
    const landmarks = result.landmarks && result.landmarks[0];

    if (landmarks) {
      const rect = getContainRect(w, h, liveVideo.videoWidth, liveVideo.videoHeight);
      drawSkeleton(ctx, landmarks, rect, threshold, showConnections.checked);
      if (recordLog.checked) {
        loggedFrames.push({ t: performance.now(), landmarks: landmarks.map((l) => ({ ...l })) });
        if (loggedFrames.length > MAX_LOGGED_FRAMES) loggedFrames.shift();
        framesLoggedLabel.textContent = `${loggedFrames.length} frames logged`;
        exportLiveJson.disabled = false;
        exportLiveCsv.disabled = false;
      }
    }
    populateTable(liveDataBody, liveTrackedCount, landmarks, threshold);
  }
  requestAnimationFrame(liveLoop);
}

new ResizeObserver(() => {
  if (liveRunning) resizeCanvasToContainer(liveCanvas, liveViewport);
}).observe(liveViewport);

exportLiveJson.addEventListener("click", () => {
  downloadBlob(
    `pose-session-${Date.now()}.json`,
    JSON.stringify({ landmark_names: LANDMARK_NAMES, frames: loggedFrames }, null, 2),
    "application/json"
  );
});
exportLiveCsv.addEventListener("click", () => {
  downloadBlob(`pose-session-${Date.now()}.csv`, framesToCsv(loggedFrames), "text/csv");
});

/* ================= UPLOAD MODE ================= */

const uploadViewport = document.getElementById("uploadViewport");
const uploadImage = document.getElementById("uploadImage");
const uploadCanvas = document.getElementById("uploadCanvas");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const uploadInput = document.getElementById("uploadInput");
const uploadInputReplace = document.getElementById("uploadInputReplace");
const replaceImageBtn = document.getElementById("replaceImageBtn");
const confThresholdUpload = document.getElementById("confThresholdUpload");
const confValueUpload = document.getElementById("confValueUpload");
const showConnectionsUpload = document.getElementById("showConnectionsUpload");
const uploadDataBody = document.getElementById("uploadDataBody");
const uploadTrackedCount = document.getElementById("uploadTrackedCount");
const uploadStatusLabel = document.getElementById("uploadStatusLabel");
const exportUploadJson = document.getElementById("exportUploadJson");
const exportUploadCsv = document.getElementById("exportUploadCsv");

let currentImageLandmarks = null;

confThresholdUpload.addEventListener("input", () => {
  confValueUpload.textContent = Number(confThresholdUpload.value).toFixed(2);
  if (currentImageLandmarks) renderUploadResult();
});
showConnectionsUpload.addEventListener("change", () => {
  if (currentImageLandmarks) renderUploadResult();
});

function renderUploadResult() {
  const { w, h } = resizeCanvasToContainer(uploadCanvas, uploadViewport);
  const ctx = uploadCanvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const threshold = Number(confThresholdUpload.value);
  if (currentImageLandmarks) {
    const rect = getContainRect(w, h, uploadImage.naturalWidth, uploadImage.naturalHeight);
    drawSkeleton(ctx, currentImageLandmarks, rect, threshold, showConnectionsUpload.checked);
  }
  populateTable(uploadDataBody, uploadTrackedCount, currentImageLandmarks, threshold);
}

async function handleImageFile(file) {
  if (!file) return;
  uploadStatusLabel.textContent = "Analyzing…";
  const url = URL.createObjectURL(file);
  await new Promise((resolve) => {
    uploadImage.onload = resolve;
    uploadImage.src = url;
  });
  uploadPlaceholder.classList.add("is-hidden");
  replaceImageBtn.style.display = "inline-flex";

  const landmarker = await getImageLandmarker();
  const result = landmarker.detect(uploadImage);
  currentImageLandmarks = (result.landmarks && result.landmarks[0]) || null;

  renderUploadResult();

  if (currentImageLandmarks) {
    uploadStatusLabel.textContent = "Pose detected";
    exportUploadJson.disabled = false;
    exportUploadCsv.disabled = false;
  } else {
    uploadStatusLabel.textContent = "No pose detected in image";
    exportUploadJson.disabled = true;
    exportUploadCsv.disabled = true;
  }
}

uploadInput.addEventListener("change", (e) => handleImageFile(e.target.files[0]));
uploadInputReplace.addEventListener("change", (e) => handleImageFile(e.target.files[0]));

["dragover", "dragenter"].forEach((evt) =>
  uploadViewport.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadViewport.style.borderColor = "var(--cyan)";
  })
);
["dragleave", "drop"].forEach((evt) =>
  uploadViewport.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadViewport.style.borderColor = "";
  })
);
uploadViewport.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleImageFile(file);
});

new ResizeObserver(() => {
  if (currentImageLandmarks) renderUploadResult();
}).observe(uploadViewport);

exportUploadJson.addEventListener("click", () => {
  downloadBlob(
    `pose-image-${Date.now()}.json`,
    JSON.stringify({ landmark_names: LANDMARK_NAMES, landmarks: currentImageLandmarks }, null, 2),
    "application/json"
  );
});
exportUploadCsv.addEventListener("click", () => {
  downloadBlob(`pose-image-${Date.now()}.csv`, landmarksToCsv(currentImageLandmarks), "text/csv");
});
