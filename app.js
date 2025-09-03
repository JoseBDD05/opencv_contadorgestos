// ---------- Utilidades ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid  = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const mean = arr => arr.reduce((s,x)=>s+x,0) / Math.max(arr.length,1);
const std  = arr => {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  const v = arr.reduce((s,x)=>(s+(x-m)*(x-m)),0)/(arr.length-1);
  return Math.sqrt(v);
};

const L_EYE = { left: 33, right: 133, top1: 160, top2: 158, bottom1: 144, bottom2: 153 };
const R_EYE = { left: 263, right: 362, top1: 387, top2: 385, bottom1: 373, bottom2: 380 };
const LIPS  = { L:61, T:13, R:291, B:14 };
const BROW  = { leftBrow: 105, rightBrow: 334, leftEyeL: 33, leftEyeR: 133, rightEyeL: 263, rightEyeR: 362 };

function eyeAspectRatio(pts, eye) {
  const p1 = pts[eye.left],  p4 = pts[eye.right];
  const p2 = pts[eye.top1],  p6 = pts[eye.bottom1];
  const p3 = pts[eye.top2],  p5 = pts[eye.bottom2];
  const vert = (dist(p2, p6) + dist(p3, p5)) / 2;
  const horiz = dist(p1, p4);
  return vert / (horiz + 1e-6);
}
function eyebrowHeightNorm(pts) {
  const lEyeL = pts[BROW.leftEyeL], lEyeR = pts[BROW.leftEyeR];
  const rEyeL = pts[BROW.rightEyeL], rEyeR = pts[BROW.rightEyeR];
  const lEyeCenter = mid(lEyeL, lEyeR), rEyeCenter = mid(rEyeL, rEyeR);
  const lBrow = pts[BROW.leftBrow], rBrow = pts[BROW.rightBrow];
  const lH = Math.abs(lBrow.y - lEyeCenter.y) / (dist(lEyeL, lEyeR) + 1e-6);
  const rH = Math.abs(rBrow.y - rEyeCenter.y) / (dist(rEyeL, rEyeR) + 1e-6);
  return (lH + rH) / 2;
}

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const camInd = document.getElementById('cam-ind');
const faceInd = document.getElementById('face-ind');
const fpsEl = document.getElementById('fps');
const dbgMar = document.getElementById('dbgMar');

const blinkEl = document.getElementById('blinkCount');
const mouthEl = document.getElementById('mouthCount');
const browEl  = document.getElementById('browCount');

const startBtn = document.getElementById('startBtn');
const stopBtn  = document.getElementById('stopBtn');
const calBtn   = document.getElementById('calBtn');
const resetBtn = document.getElementById('resetBtn');
const deviceSel = document.getElementById('deviceSel');

let running = false;
let stream = null;

let blinkCount = 0, mouthCount = 0, browCount = 0;
let eyeClosed = false, mouthOpen = false, browRaised = false;

let baseline = { EAR: null, EBH: null, LIPAN: null, LIPAN_SIGMA: 0 };
const THR_REL = { EAR: 0.75, EBH: 1.25 };
let THR_ABS = { LIPAN_OPEN: null, LIPAN_CLOSE: null };

let lipEma = null;
const EMA_ALPHA = 0.25;
const LIP_FRAMES_NEEDED = 3;
let lipAboveFrames = 0, lipBelowFrames = 0;

const COOLDOWN = { blink: 180, mouth: 250, brow: 350 };
let lastEvt = { blink: 0, mouth: 0, brow: 0 };

let lastTime = performance.now(), frames = 0;
let lastMetrics = null;

let autoCalActive = false;
const AUTO_CAL_FRAMES = 60;
let autoLipSamples = [];

let faceMesh = null;

/* === POST snapshot acumulado por cada evento === */
async function registrarGesto() {
  const payload = {
    parpadeo: blinkCount,
    cejas:    browCount,
    boca:     mouthCount,
    fecha:    new Date().toISOString()
  };
  try {
    const resp = await fetch('https://68b8995cb71540504328aaa6.mockapi.io/api/v1/gestos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    console.log('Registro guardado:', data);
  } catch (err) {
    console.error('Error al enviar gesto:', err);
  }
}

function drawLandmarks(pts, isMouthOpen=false) {
  const draw = (idxs, close=false) => {
    ctx.beginPath();
    for (let i = 0; i < idxs.length; i++) {
      const p = pts[idxs[i]];
      const x = p.x * canvas.width;
      const y = p.y * canvas.height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    if (close) ctx.closePath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(65, 175, 255, 0.9)';
    ctx.stroke();
  };
  draw([33,160,158,133,153,144], true);
  draw([263,387,385,362,380,373], true);

  const mouth = [LIPS.L, LIPS.T, LIPS.R, LIPS.B, LIPS.L];
  if (isMouthOpen) {
    ctx.beginPath();
    mouth.forEach((i, k) => {
      const p = pts[i]; const x = p.x * canvas.width, y = p.y * canvas.height;
      k ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 200, 255, 0.15)';
    ctx.fill();
  }
  draw(mouth, true);

  [105,334].forEach(i => {
    const p = pts[i];
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 212, 59, 0.9)';
    ctx.fill();
  });
}

function paintFrame(image, landmarks, isMouthOpen=false) {
  if (canvas.width !== image.videoWidth)  canvas.width  = image.videoWidth;
  if (canvas.height !== image.videoHeight) canvas.height = image.videoHeight;

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (landmarks && landmarks.length > 0) drawLandmarks(landmarks[0], isMouthOpen);
  ctx.restore();
}

function mouthAreaNorm(pts) {
  if (!window.cvReady || !window.cv) return 0;

  const W = video.videoWidth, H = video.videoHeight;
  const pL = { x: pts[LIPS.L].x * W, y: pts[LIPS.L].y * H };
  const pT = { x: pts[LIPS.T].x * W, y: pts[LIPS.T].y * H };
  const pR = { x: pts[LIPS.R].x * W, y: pts[LIPS.R].y * H };
  const pB = { x: pts[LIPS.B].x * W, y: pts[LIPS.B].y * H };

  const arr = new Float32Array([pL.x,pL.y, pT.x,pT.y, pR.x,pR.y, pB.x,pB.y]);
  const cnt = cv.matFromArray(4, 1, cv.CV_32FC2, arr);
  const area = Math.abs(cv.contourArea(cnt, true));
  cnt.delete();

  const mouthWidth = Math.hypot(pR.x - pL.x, pR.y - pL.y) + 1e-6;
  return area / (mouthWidth * mouthWidth);
}

function processMetrics(pts) {
  const EAR = (eyeAspectRatio(pts, L_EYE) + eyeAspectRatio(pts, R_EYE)) / 2;
  const EBH = eyebrowHeightNorm(pts);
  const LIPAN = mouthAreaNorm(pts);
  return { EAR, EBH, LIPAN };
}
function ensureBaseline(m) {
  if (baseline.EAR == null && m.EAR > 0) baseline.EAR = m.EAR;
  if (baseline.EBH == null && m.EBH > 0) baseline.EBH = m.EBH;
  if (baseline.LIPAN == null && m.LIPAN >= 0) baseline.LIPAN = m.LIPAN;
}

function updateEvents(m) {
  const now = performance.now();

  // --- Parpadeo ---
  const blinkCond = (baseline.EAR && m.EAR < baseline.EAR * THR_REL.EAR);
  if (blinkCond && !eyeClosed && (now - lastEvt.blink > COOLDOWN.blink)) {
    eyeClosed = true; lastEvt.blink = now; blinkCount++; blinkEl.textContent = blinkCount;
    registrarGesto(); // snapshot acumulado
  } else if (!blinkCond) { eyeClosed = false; }

  // --- Boca ---
  const ANv = (typeof m.LIPAN_EMA === 'number') ? m.LIPAN_EMA : m.LIPAN;
  if (THR_ABS.LIPAN_OPEN != null && THR_ABS.LIPAN_CLOSE != null) {
    const openCond  = ANv > THR_ABS.LIPAN_OPEN;
    const closeCond = ANv < THR_ABS.LIPAN_CLOSE;

    if (openCond) { lipAboveFrames++; lipBelowFrames = 0; }
    else if (closeCond) { lipBelowFrames++; lipAboveFrames = 0; }
    else { lipAboveFrames = Math.max(0, lipAboveFrames - 1); lipBelowFrames = Math.max(0, lipBelowFrames - 1); }

    if (!mouthOpen && lipAboveFrames >= LIP_FRAMES_NEEDED && (now - lastEvt.mouth > COOLDOWN.mouth)) {
      mouthOpen = true; lastEvt.mouth = now; mouthCount++; mouthEl.textContent = mouthCount;
      registrarGesto(); // snapshot acumulado
    }
    if (mouthOpen && lipBelowFrames >= LIP_FRAMES_NEEDED) {
      mouthOpen = false;
    }
  }

  // --- Cejas ---
  const browCond = (baseline.EBH && m.EBH > baseline.EBH * THR_REL.EBH);
  if (browCond && !browRaised && (now - lastEvt.brow > COOLDOWN.brow)) {
    browRaised = true; lastEvt.brow = now; browCount++; browEl.textContent = browCount;
    registrarGesto(); // snapshot acumulado
  } else if (!browCond) { browRaised = false; }
}

function onResults(results) {
  const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;
  faceInd.classList.toggle('ok', hasFace);
  faceInd.classList.toggle('no', !hasFace);

  let isMouthOpenDraw = false;

  if (hasFace) {
    const pts = results.multiFaceLandmarks[0];
    const m = processMetrics(pts);

    if (lipEma == null) lipEma = m.LIPAN;
    else lipEma = EMA_ALPHA * m.LIPAN + (1 - EMA_ALPHA) * lipEma;

    ensureBaseline(m);
    lastMetrics = { ...m, LIPAN_EMA: lipEma };
    updateEvents(lastMetrics);

    if (THR_ABS.LIPAN_OPEN != null) {
      isMouthOpenDraw = lipEma > THR_ABS.LIPAN_OPEN;
    }

    if (autoCalActive) {
      autoLipSamples.push(m.LIPAN);
      if (autoLipSamples.length >= AUTO_CAL_FRAMES) {
        applyCalibrationFromSamples(autoLipSamples);
        autoCalActive = false;
      }
    }

    dbgMar.textContent =
      `ÁreaN: ${m.LIPAN.toFixed(4)} | EMA: ${lipEma.toFixed(4)} | base: ${baseline.LIPAN?.toFixed(4) ?? '—'} | ` +
      `thrOpen: ${THR_ABS.LIPAN_OPEN?.toFixed(4) ?? '—'} | thrClose: ${THR_ABS.LIPAN_CLOSE?.toFixed(4) ?? '—'}`;
  } else {
    dbgMar.textContent =
      `ÁreaN: — | EMA: — | base: ${baseline.LIPAN?.toFixed(4) ?? '—'} | ` +
      `thrOpen: ${THR_ABS.LIPAN_OPEN?.toFixed(4) ?? '—'} | thrClose: ${THR_ABS.LIPAN_CLOSE?.toFixed(4) ?? '—'}`;
  }

  paintFrame(video, results.multiFaceLandmarks, isMouthOpenDraw);

  frames++;
  const now = performance.now();
  if (now - lastTime >= 500) {
    fpsEl.textContent = Math.round((frames / (now - lastTime)) * 1000);
    frames = 0; lastTime = now;
  }
}

function applyCalibrationFromSamples(samples) {
  const mBase = mean(samples);
  const s     = std(samples);
  baseline.LIPAN = mBase;
  baseline.LIPAN_SIGMA = s;

  const deltaOpen  = Math.max(0.004, 3 * s);
  const deltaClose = Math.max(0.002, 1.5 * s);
  THR_ABS.LIPAN_OPEN  = mBase + deltaOpen;
  THR_ABS.LIPAN_CLOSE = mBase + deltaClose;

  lipEma = mBase;
}

async function initFaceMesh() {
  if (faceMesh) return;

  const FaceMeshClass = (window.FaceMesh && window.FaceMesh.FaceMesh)
                      ? window.FaceMesh.FaceMesh
                      : window.FaceMesh;

  if (!FaceMeshClass) {
    console.error('No se encontró la clase FaceMesh en window.FaceMesh');
    alert('No se pudo cargar MediaPipe FaceMesh. Verifica el <script> del CDN.');
    return;
  }

  faceMesh = new FaceMeshClass({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  faceMesh.onResults(onResults);
}

async function listCams() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === 'videoinput');
  deviceSel.innerHTML = '';
  cams.forEach((d,i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Cámara ${i+1}`;
    deviceSel.appendChild(opt);
  });
}

function waitForOpenCV() {
  return new Promise(resolve => {
    if (window.cvReady) return resolve();
    const iv = setInterval(() => {
      if (window.cvReady) { clearInterval(iv); resolve(); }
    }, 50);
  });
}

async function startCamera() {
  if (running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('Tu navegador no soporta getUserMedia. Usa Chrome/Edge actualizados.'); return;
  }

  await waitForOpenCV();

  try {
    const deviceId = deviceSel.value || undefined;
    const constraints = deviceId
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: { facingMode: 'user', width: {ideal: 960}, height: {ideal: 540} }, audio: false };

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream; video.muted = true; await video.play();

    camInd.classList.add('ok'); camInd.classList.remove('no');

    await initFaceMesh();
    running = true;

    autoCalActive = true; autoLipSamples = [];
    THR_ABS.LIPAN_OPEN = THR_ABS.LIPAN_CLOSE = null;

    const loop = async () => {
      if (!running) return;
      if (!faceMesh) { requestAnimationFrame(loop); return; }
      try { await faceMesh.send({ image: video }); } catch(e) { console.error('FaceMesh:', e); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (err) {
    console.error('Error al iniciar cámara:', err);
    const name = err?.name || 'Error';
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      alert('Permiso de cámara denegado. Revisa el candado del sitio y permite la cámara.');
    } else if (name === 'NotReadableError') {
      alert('La cámara está en uso por otra app. Ciérrala (Zoom/Teams/Meet/OBS) e inténtalo de nuevo.');
    } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      alert('No se encontró cámara o no cumple las restricciones. Prueba elegir otra en el selector.');
    } else {
      alert(`No se pudo iniciar la cámara. (${name})`);
    }
  }
}

function stopCamera() {
  running = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (faceMesh?.close) faceMesh.close();
  faceMesh = null;
  video.srcObject = null;
  camInd.classList.remove('ok'); camInd.classList.add('no');
  faceInd.classList.remove('ok'); faceInd.classList.add('no');

  lipEma = null; lipAboveFrames = 0; lipBelowFrames = 0; mouthOpen = false;
  autoCalActive = false; autoLipSamples = [];
  THR_ABS.LIPAN_OPEN = THR_ABS.LIPAN_CLOSE = null;
}

async function calibrate(framesToUse = 75) {
  if (!running) return;
  calBtn.disabled = true; calBtn.textContent = 'Calibrando...';

  let lipSamples = [];
  let sumEAR=0,sumEBH=0,n=0;

  await new Promise(resolve => {
    const tick = () => {
      if (n >= framesToUse) return resolve();
      if (lastMetrics) {
        sumEAR += lastMetrics.EAR;
        sumEBH += lastMetrics.EBH;
        lipSamples.push(lastMetrics.LIPAN);
        n++;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  if (n > 10) {
    baseline.EAR = sumEAR / n;
    baseline.EBH = sumEBH / n;
    const mb = mean(lipSamples), s = std(lipSamples);
    baseline.LIPAN = mb; baseline.LIPAN_SIGMA = s;
    const deltaOpen  = Math.max(0.004, 3 * s);
    const deltaClose = Math.max(0.002, 1.5 * s);
    THR_ABS.LIPAN_OPEN  = mb + deltaOpen;
    THR_ABS.LIPAN_CLOSE = mb + deltaClose;
    lipEma = mb;
  }

  calBtn.textContent = 'Calibrar (2–3 s boca cerrada)';
  calBtn.disabled = false;
  autoCalActive = false;
}

function resetCounters() {
  blinkCount = mouthCount = browCount = 0;
  blinkEl.textContent = mouthEl.textContent = browEl.textContent = '0';
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
calBtn.addEventListener('click', () => calibrate(75));
resetBtn.addEventListener('click', resetCounters);

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    s.getTracks().forEach(t => t.stop());
  } catch {}
  await listCams();
});
