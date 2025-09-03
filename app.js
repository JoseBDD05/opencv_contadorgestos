// ---------- Utilidades ----------
// Función de distancia euclidiana entre dos puntos con propiedades {x,y}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
// Punto medio entre dos puntos {x,y}
const mid  = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
// Media de un arreglo numérico (robusta contra arreglo vacío)
const mean = arr => arr.reduce((s,x)=>s+x,0) / Math.max(arr.length,1);
// Desviación estándar muestral de un arreglo numérico
const std  = arr => {
  if (arr.length <= 1) return 0;            // con 0 o 1 muestra no hay dispersión definida
  const m = mean(arr);                      // calculamos media
  const v = arr.reduce((s,x)=>(s+(x-m)*(x-m)),0)/(arr.length-1); // varianza muestral
  return Math.sqrt(v);                       // desviación estándar
};

// Índices de landmarks de MediaPipe para ojos, labios y cejas
const L_EYE = { left: 33, right: 133, top1: 160, top2: 158, bottom1: 144, bottom2: 153 };
const R_EYE = { left: 263, right: 362, top1: 387, top2: 385, bottom1: 373, bottom2: 380 };
const LIPS  = { L:61, T:13, R:291, B:14 };
const BROW  = { leftBrow: 105, rightBrow: 334, leftEyeL: 33, leftEyeR: 133, rightEyeL: 263, rightEyeR: 362 };

// Cálculo del Eye Aspect Ratio (EAR) para un ojo dado
function eyeAspectRatio(pts, eye) {
  const p1 = pts[eye.left],  p4 = pts[eye.right];   // extremos horizontales del ojo
  const p2 = pts[eye.top1],  p6 = pts[eye.bottom1]; // primer par vertical
  const p3 = pts[eye.top2],  p5 = pts[eye.bottom2]; // segundo par vertical
  const vert = (dist(p2, p6) + dist(p3, p5)) / 2;   // altura media (parpadeo reduce esta altura)
  const horiz = dist(p1, p4);                        // ancho del ojo
  return vert / (horiz + 1e-6);                      // EAR = altura/ancho (evita división por 0)
}
// Altura normalizada de las cejas respecto al centro de los ojos
function eyebrowHeightNorm(pts) {
  const lEyeL = pts[BROW.leftEyeL], lEyeR = pts[BROW.leftEyeR];   // extremos del ojo izquierdo
  const rEyeL = pts[BROW.rightEyeL], rEyeR = pts[BROW.rightEyeR]; // extremos del ojo derecho
  const lEyeCenter = mid(lEyeL, lEyeR), rEyeCenter = mid(rEyeL, rEyeR); // centros de ojos
  const lBrow = pts[BROW.leftBrow], rBrow = pts[BROW.rightBrow];  // puntos de ceja
  const lH = Math.abs(lBrow.y - lEyeCenter.y) / (dist(lEyeL, lEyeR) + 1e-6); // altura normalizada izq.
  const rH = Math.abs(rBrow.y - rEyeCenter.y) / (dist(rEyeL, rEyeR) + 1e-6); // altura normalizada der.
  return (lH + rH) / 2;                                           // promedio de ambas cejas
}

// Referencias a DOM para video/canvas y contexto 2D
const video = document.getElementById('video');     // elemento <video> de cámara
const canvas = document.getElementById('overlay');  // canvas overlay de dibujo
const ctx = canvas.getContext('2d');                // contexto 2D para pintar

// Indicadores de estado y métricas en UI
const camInd = document.getElementById('cam-ind');  // punto de estado de cámara
const faceInd = document.getElementById('face-ind');// punto de estado de rostro detectado
const fpsEl = document.getElementById('fps');       // label FPS
const dbgMar = document.getElementById('dbgMar');   // texto de debug de boca

// Contadores visibles
const blinkEl = document.getElementById('blinkCount'); // contador de parpadeos
const mouthEl = document.getElementById('mouthCount'); // contador de bocas abiertas
const browEl  = document.getElementById('browCount');  // contador de cejas levantadas

// Botones y selector de cámara
const startBtn = document.getElementById('startBtn');  // iniciar cámara
const stopBtn  = document.getElementById('stopBtn');   // detener cámara
const calBtn   = document.getElementById('calBtn');    // calibrar
const resetBtn = document.getElementById('resetBtn');  // reiniciar contadores
const deviceSel = document.getElementById('deviceSel');// selector de dispositivos de video

// Estado principal del ciclo de cámara
let running = false;           // si el loop está activo
let stream = null;             // MediaStream de getUserMedia

// Contadores y flags de eventos detectados
let blinkCount = 0, mouthCount = 0, browCount = 0;     // contadores
let eyeClosed = false, mouthOpen = false, browRaised = false; // flags de estado actuales

// Baseline/calibración: valores base y sigma para boca
let baseline = { EAR: null, EBH: null, LIPAN: null, LIPAN_SIGMA: 0 }; // baseline de EAR, EBH y área de labios
const THR_REL = { EAR: 0.75, EBH: 1.25 };           // umbrales relativos a baseline
let THR_ABS = { LIPAN_OPEN: null, LIPAN_CLOSE: null }; // umbrales absolutos de boca (abrir/cerrar)

// Filtro EMA para suavizar área de boca
let lipEma = null;                                   // valor EMA de LIPAN
const EMA_ALPHA = 0.25;                              // factor de suavizado (25%)
const LIP_FRAMES_NEEDED = 3;                         // frames consecutivos requeridos para confirmar evento
let lipAboveFrames = 0, lipBelowFrames = 0;          // contadores de frames por encima/debajo de umbrales

// Cooldowns para no contar eventos muy seguidos (en ms)
const COOLDOWN = { blink: 180, mouth: 250, brow: 350 }; // tiempos mínimos entre eventos
let lastEvt = { blink: 0, mouth: 0, brow: 0 };          // timestamps del último evento

// Cómputo de FPS
let lastTime = performance.now(), frames = 0;        // acumuladores para calcular FPS cada ~500ms
let lastMetrics = null;                               // últimas métricas calculadas (EAR, EBH, LIPAN, EMA)

// Auto calibración (toma muestras con boca cerrada al iniciar)
let autoCalActive = false;                            // si la auto-calibración está activa
const AUTO_CAL_FRAMES = 60;                           // cuántos frames de muestra tomar
let autoLipSamples = [];                              // arreglo de muestras de LIPAN

// FaceMesh de MediaPipe (instancia)
let faceMesh = null;                                  // referencia a la instancia FaceMesh

// Dibuja landmarks de ojos, boca y puntos de cejas
function drawLandmarks(pts, isMouthOpen=false) {
  // Función interna para dibujar líneas entre índices
  const draw = (idxs, close=false) => {
    ctx.beginPath();                                  // inicia path
    for (let i = 0; i < idxs.length; i++) {           // recorre índices
      const p = pts[idxs[i]];                         // punto normalizado {x,y}
      const x = p.x * canvas.width;                   // escala x a px de canvas
      const y = p.y * canvas.height;                  // escala y a px de canvas
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);  // mueve o traza línea
    }
    if (close) ctx.closePath();                       // cierra el polígono si procede
    ctx.lineWidth = 2;                                // grosor de línea
    ctx.strokeStyle = 'rgba(65, 175, 255, 0.9)';      // color de trazo azul
    ctx.stroke();                                     // dibuja
  };
  // Ojo izquierdo (contorno)
  draw([33,160,158,133,153,144], true);
  // Ojo derecho (contorno)
  draw([263,387,385,362,380,373], true);

  // Boca como rombo (61-13-291-14-61)
  const mouth = [LIPS.L, LIPS.T, LIPS.R, LIPS.B, LIPS.L];
  if (isMouthOpen) {                                  // si está abierta, rellenamos levemente
    ctx.beginPath();
    mouth.forEach((i, k) => {
      const p = pts[i]; const x = p.x * canvas.width, y = p.y * canvas.height;
      k ? ctx.lineTo(x,y) : ctx.moveTo(x,y);          // dibuja rombo
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 200, 255, 0.15)';        // relleno cian translúcido
    ctx.fill();
  }
  draw(mouth, true);                                   // trazo del rombo de boca

  // Cejas como puntos (landmarks 105 y 334)
  [105,334].forEach(i => {
    const p = pts[i];                                  // punto de ceja
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2); // círculo pequeño
    ctx.fillStyle = 'rgba(255, 212, 59, 0.9)';         // color amarillo
    ctx.fill();
  });
}

// Render de cada frame: ajusta tamaño, limpia, dibuja frame y landmarks
function paintFrame(image, landmarks, isMouthOpen=false) {
  if (canvas.width !== image.videoWidth)  canvas.width  = image.videoWidth;   // iguala ancho a video
  if (canvas.height !== image.videoHeight) canvas.height = image.videoHeight; // iguala alto a video

  ctx.save();                                           // guarda estado de canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);     // limpia frame anterior
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height); // pinta el frame del video
  if (landmarks && landmarks.length > 0) drawLandmarks(landmarks[0], isMouthOpen); // dibuja si hay rostro
  ctx.restore();                                        // restaura estado
}

// Área de la boca normalizada por el ancho de boca^2 usando OpenCV.js
function mouthAreaNorm(pts) {
  if (!window.cvReady || !window.cv) return 0;          // si OpenCV no está listo, devolvemos 0

  const W = video.videoWidth, H = video.videoHeight;    // dimensiones del video
  const pL = { x: pts[LIPS.L].x * W, y: pts[LIPS.L].y * H }; // punto izquierdo en px
  const pT = { x: pts[LIPS.T].x * W, y: pts[LIPS.T].y * H }; // punto superior en px
  const pR = { x: pts[LIPS.R].x * W, y: pts[LIPS.R].y * H }; // punto derecho en px
  const pB = { x: pts[LIPS.B].x * W, y: pts[LIPS.B].y * H }; // punto inferior en px

  const arr = new Float32Array([pL.x,pL.y, pT.x,pT.y, pR.x,pR.y, pB.x,pB.y]); // contorno 4 puntos
  const cnt = cv.matFromArray(4, 1, cv.CV_32FC2, arr); // Mat de contorno (4x1 de puntos 2D)
  const area = Math.abs(cv.contourArea(cnt, true));    // área firmada del contorno (rombo)
  cnt.delete();                                        // liberamos Mat para evitar leaks

  const mouthWidth = Math.hypot(pR.x - pL.x, pR.y - pL.y) + 1e-6; // ancho de boca (px)
  return area / (mouthWidth * mouthWidth);             // área normalizada por ancho^2
}

// Calcula métricas instantáneas a partir de los landmarks
function processMetrics(pts) {
  const EAR = (eyeAspectRatio(pts, L_EYE) + eyeAspectRatio(pts, R_EYE)) / 2; // EAR promedio
  const EBH = eyebrowHeightNorm(pts);                     // altura de cejas normalizada
  const LIPAN = mouthAreaNorm(pts);                       // área normalizada de boca
  return { EAR, EBH, LIPAN };                             // regresamos objeto métrico
}
// Fija baseline si aún no existe para cada métrica (primera vez)
function ensureBaseline(m) {
  if (baseline.EAR == null && m.EAR > 0) baseline.EAR = m.EAR;          // EAR base
  if (baseline.EBH == null && m.EBH > 0) baseline.EBH = m.EBH;          // EBH base
  if (baseline.LIPAN == null && m.LIPAN >= 0) baseline.LIPAN = m.LIPAN; // LIPAN base
}

// Lógica de detección y conteo de eventos (parpadeo, boca, cejas)
function updateEvents(m) {
  const now = performance.now();                         // timestamp actual (ms)

  // --- Parpadeo ---
  const blinkCond = (baseline.EAR && m.EAR < baseline.EAR * THR_REL.EAR); // EAR por debajo de umbral
  if (blinkCond && !eyeClosed && (now - lastEvt.blink > COOLDOWN.blink)) {
    eyeClosed = true; lastEvt.blink = now; blinkCount++; blinkEl.textContent = blinkCount; // cuenta
  } else if (!blinkCond) { eyeClosed = false; }          // si se abrió, limpiamos flag

  // --- Boca abierta/cerrada con histéresis y EMA ---
  const ANv = (typeof m.LIPAN_EMA === 'number') ? m.LIPAN_EMA : m.LIPAN; // valor filtrado o crudo
  if (THR_ABS.LIPAN_OPEN != null && THR_ABS.LIPAN_CLOSE != null) {       // si tenemos umbrales
    const openCond  = ANv > THR_ABS.LIPAN_OPEN;                           // condición de apertura
    const closeCond = ANv < THR_ABS.LIPAN_CLOSE;                          // condición de cierre

    if (openCond) { lipAboveFrames++; lipBelowFrames = 0; }               // contamos frames arriba
    else if (closeCond) { lipBelowFrames++; lipAboveFrames = 0; }         // contamos frames abajo
    else {                                                                // zona intermedia: decaimiento
      lipAboveFrames = Math.max(0, lipAboveFrames - 1);
      lipBelowFrames = Math.max(0, lipBelowFrames - 1);
    }

    if (!mouthOpen && lipAboveFrames >= LIP_FRAMES_NEEDED && (now - lastEvt.mouth > COOLDOWN.mouth)) {
      mouthOpen = true; lastEvt.mouth = now; mouthCount++; mouthEl.textContent = mouthCount; // cuenta apertura
    }
    if (mouthOpen && lipBelowFrames >= LIP_FRAMES_NEEDED) {
      mouthOpen = false;                                                  // marcamos cierre estable
    }
  }

  // --- Cejas levantadas (comparado con baseline) ---
  const browCond = (baseline.EBH && m.EBH > baseline.EBH * THR_REL.EBH); // EBH mayor al umbral relativo
  if (browCond && !browRaised && (now - lastEvt.brow > COOLDOWN.brow)) {
    browRaised = true; lastEvt.brow = now; browCount++; browEl.textContent = browCount; // cuenta
  } else if (!browCond) { browRaised = false; }                          // resetea flag cuando baja
}

// Callback de MediaPipe FaceMesh en cada resultado
function onResults(results) {
  const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0; // hay rostro
  faceInd.classList.toggle('ok', hasFace); // enciende indicador "ok" si hay rostro
  faceInd.classList.toggle('no', !hasFace);// apaga si no hay

  let isMouthOpenDraw = false;                 // flag para rellenar rombo de boca visualmente

  if (hasFace) {
    const pts = results.multiFaceLandmarks[0]; // primer rostro
    const m = processMetrics(pts);             // calculamos métricas

    if (lipEma == null) lipEma = m.LIPAN;      // inicializa EMA con primer valor
    else lipEma = EMA_ALPHA * m.LIPAN + (1 - EMA_ALPHA) * lipEma; // EMA exponencial

    ensureBaseline(m);                          // asegura baseline inicial
    lastMetrics = { ...m, LIPAN_EMA: lipEma };  // guardamos últimas métricas + EMA
    updateEvents(lastMetrics);                  // actualizamos eventos/contadores

    if (THR_ABS.LIPAN_OPEN != null) {          // si ya hay umbral de apertura
      isMouthOpenDraw = lipEma > THR_ABS.LIPAN_OPEN; // decide relleno visual del rombo
    }

    if (autoCalActive) {                       // si estamos en auto-calibración
      autoLipSamples.push(m.LIPAN);            // acumulamos muestras de LIPAN
      if (autoLipSamples.length >= AUTO_CAL_FRAMES) { // si ya juntamos suficientes
        applyCalibrationFromSamples(autoLipSamples);  // fijamos umbrales con sigma
        autoCalActive = false;                 // desactivamos auto-calibración
      }
    }

    // Texto de depuración de área de boca, EMA y umbrales
    dbgMar.textContent =
      `ÁreaN: ${m.LIPAN.toFixed(4)} | EMA: ${lipEma.toFixed(4)} | base: ${baseline.LIPAN?.toFixed(4) ?? '—'} | ` +
      `thrOpen: ${THR_ABS.LIPAN_OPEN?.toFixed(4) ?? '—'} | thrClose: ${THR_ABS.LIPAN_CLOSE?.toFixed(4) ?? '—'}`;
  } else {
    // Si no hay rostro, mantenemos baseline y mostramos guiones
    dbgMar.textContent =
      `ÁreaN: — | EMA: — | base: ${baseline.LIPAN?.toFixed(4) ?? '—'} | ` +
      `thrOpen: ${THR_ABS.LIPAN_OPEN?.toFixed(4) ?? '—'} | thrClose: ${THR_ABS.LIPAN_CLOSE?.toFixed(4) ?? '—'}`;
  }

  paintFrame(video, results.multiFaceLandmarks, isMouthOpenDraw); // dibuja frame + landmarks

  frames++;                                   // acumulamos un frame para cómputo de FPS
  const now = performance.now();              // timestamp actual
  if (now - lastTime >= 500) {                // cada ~0.5s actualizamos lectura FPS
    fpsEl.textContent = Math.round((frames / (now - lastTime)) * 1000); // FPS = frames/ms * 1000
    frames = 0; lastTime = now;               // reseteamos acumuladores
  }
}

// Aplica calibración a partir de muestras de boca (media+sigma -> umbrales)
function applyCalibrationFromSamples(samples) {
  const mBase = mean(samples);                // media de LIPAN
  const s     = std(samples);                 // sigma muestral
  baseline.LIPAN = mBase;                     // baseline de boca = media
  baseline.LIPAN_SIGMA = s;                   // guardamos sigma

  const deltaOpen  = Math.max(0.004, 3 * s);  // apertura: base + 3σ (mínimo absoluto 0.004)
  const deltaClose = Math.max(0.002, 1.5 * s);// cierre:   base + 1.5σ (mínimo absoluto 0.002)
  THR_ABS.LIPAN_OPEN  = mBase + deltaOpen;    // umbral absoluto de apertura
  THR_ABS.LIPAN_CLOSE = mBase + deltaClose;   // umbral absoluto de cierre (histéresis)

  lipEma = mBase;                             // reinicia EMA a la base
}

// Inicializa MediaPipe FaceMesh (si no existe)
async function initFaceMesh() {
  if (faceMesh) return;                       // evita re-inicialización

  // Compatibilidad con los dos namespaces posibles
  const FaceMeshClass = (window.FaceMesh && window.FaceMesh.FaceMesh)
                      ? window.FaceMesh.FaceMesh
                      : window.FaceMesh;

  if (!FaceMeshClass) {                       // si no está definido, avisamos
    console.error('No se encontró la clase FaceMesh en window.FaceMesh');
    alert('No se pudo cargar MediaPipe FaceMesh. Verifica el <script> del CDN.');
    return;
  }

  faceMesh = new FaceMeshClass({              // creamos instancia
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` // ruta CDN
  });

  faceMesh.setOptions({                       // opciones por defecto
    maxNumFaces: 1,                           // un solo rostro
    refineLandmarks: true,                    // landmarks refinados (labios/ojos)
    minDetectionConfidence: 0.5,              // conf. mínima de detección
    minTrackingConfidence: 0.5                // conf. mínima de seguimiento
  });

  faceMesh.onResults(onResults);              // registramos callback por frame
}

// Llena el selector con las cámaras disponibles
async function listCams() {
  const devices = await navigator.mediaDevices.enumerateDevices(); // lista dispositivos
  const cams = devices.filter(d => d.kind === 'videoinput');       // solo videoinput
  deviceSel.innerHTML = '';                                        // limpia opciones
  cams.forEach((d,i) => {                                          // agrega opción por cámara
    const opt = document.createElement('option');
    opt.value = d.deviceId;                                        // id del dispositivo
    opt.textContent = d.label || `Cámara ${i+1}`;                  // etiqueta visible
    deviceSel.appendChild(opt);
  });
}

// Espera activa hasta que OpenCV (WASM) indique estar listo
function waitForOpenCV() {
  return new Promise(resolve => {
    if (window.cvReady) return resolve();                          // si ya está, resolvemos
    const iv = setInterval(() => {                                 // si no, verificamos cada 50ms
      if (window.cvReady) { clearInterval(iv); resolve(); }        // y resolvemos cuando esté
    }, 50);
  });
}

// Inicia la cámara, FaceMesh y loop principal
async function startCamera() {
  if (running) return;                                             // evita doble inicio
  if (!navigator.mediaDevices?.getUserMedia) {                     // soporte del API
    alert('Tu navegador no soporta getUserMedia. Usa Chrome/Edge actualizados.'); return;
  }

  await waitForOpenCV();                                           // esperamos a OpenCV

  try {
    const deviceId = deviceSel.value || undefined;                 // id seleccionado (o indef.)
    const constraints = deviceId                                   // constraints según selección
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: { facingMode: 'user', width: {ideal: 960}, height: {ideal: 540} }, audio: false };

    stream = await navigator.mediaDevices.getUserMedia(constraints); // pedimos permisos/stream
    video.srcObject = stream; video.muted = true; await video.play(); // conectamos y reproducimos

    camInd.classList.add('ok'); camInd.classList.remove('no');     // indicador de cámara activa

    await initFaceMesh();                                          // inicializa FaceMesh una vez
    running = true;                                                // marcamos loop activo

    autoCalActive = true; autoLipSamples = [];                     // habilitamos auto-calibración
    THR_ABS.LIPAN_OPEN = THR_ABS.LIPAN_CLOSE = null;               // limpiamos umbrales previos

    // Bucle de procesamiento por frame
    const loop = async () => {
      if (!running) return;                                        // si se detiene, salimos
      if (!faceMesh) { requestAnimationFrame(loop); return; }      // si aún no hay faceMesh, espera
      try { await faceMesh.send({ image: video }); } catch(e) { console.error('FaceMesh:', e); } // procesa
      requestAnimationFrame(loop);                                  // siguiente frame
    };
    requestAnimationFrame(loop);                                    // arranca el loop
  } catch (err) {
    console.error('Error al iniciar cámara:', err);                 // log del error
    const name = err?.name || 'Error';                              // nombre de error para UI
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      alert('Permiso de cámara denegado. Revisa el candado del sitio y permite la cámara.');
    } else if (name === 'NotReadableError') {
      alert('La cámara está en uso por otra app. Ciérrala (Zoom/Teams/Meet/OBS) e inténtalo de nuevo.');
    } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      alert('No se encontró cámara o no cumple las restricciones. Prueba elegir otra en el selector.');
    } else {
      alert(`No se pudo iniciar la cámara. (${name})`);             // otros errores genéricos
    }
  }
}

// Detiene la cámara y limpia estado
function stopCamera() {
  running = false;                                                 // apaga loop
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } // detiene tracks
  if (faceMesh?.close) faceMesh.close();                           // cierra faceMesh si soporta
  faceMesh = null;                                                 // limpia referencia
  video.srcObject = null;                                          // desconecta video
  camInd.classList.remove('ok'); camInd.classList.add('no');       // actualiza UI de cámara
  faceInd.classList.remove('ok'); faceInd.classList.add('no');     // actualiza UI de rostro

  lipEma = null; lipAboveFrames = 0; lipBelowFrames = 0; mouthOpen = false; // resetea boca/EMA
  autoCalActive = false; autoLipSamples = [];                      // resetea auto-calibración
  THR_ABS.LIPAN_OPEN = THR_ABS.LIPAN_CLOSE = null;                 // borra umbrales absolutos
}

// Calibración manual (captura N frames con boca cerrada)
async function calibrate(framesToUse = 75) {
  if (!running) return;                                            // requiere cámara corriendo
  calBtn.disabled = true; calBtn.textContent = 'Calibrando...';    // deshabilita botón y cambia texto

  let lipSamples = [];                                             // muestras de LIPAN
  let sumEAR=0,sumEBH=0,n=0;                                       // acumuladores para EAR/EBH

  // Bucle de captura cuadro a cuadro por requestAnimationFrame
  await new Promise(resolve => {
    const tick = () => {
      if (n >= framesToUse) return resolve();                      // alcanzamos N frames
      if (lastMetrics) {                                           // si ya hay métricas
        sumEAR += lastMetrics.EAR;                                 // acumulamos EAR
        sumEBH += lastMetrics.EBH;                                 // acumulamos EBH
        lipSamples.push(lastMetrics.LIPAN);                        // guardamos LIPAN
        n++;                                                       // incrementa contador
      }
      requestAnimationFrame(tick);                                 // siguiente frame
    };
    requestAnimationFrame(tick);                                   // arranca
  });

  if (n > 10) {                                                    // si reunimos suficientes muestras
    baseline.EAR = sumEAR / n;                                     // baseline EAR
    baseline.EBH = sumEBH / n;                                     // baseline EBH
    const mb = mean(lipSamples), s = std(lipSamples);              // media y sigma de LIPAN
    baseline.LIPAN = mb; baseline.LIPAN_SIGMA = s;                 // guardamos baseline
    const deltaOpen  = Math.max(0.004, 3 * s);                     // apertura absoluta = base + 3σ
    const deltaClose = Math.max(0.002, 1.5 * s);                   // cierre absoluto = base + 1.5σ
    THR_ABS.LIPAN_OPEN  = mb + deltaOpen;                          // umbral apertura
    THR_ABS.LIPAN_CLOSE = mb + deltaClose;                         // umbral cierre
    lipEma = mb;                                                   // reinicia EMA
  }

  calBtn.textContent = 'Calibrar (2–3 s boca cerrada)';            // restaura texto
  calBtn.disabled = false;                                         // re-habilita botón
  autoCalActive = false;                                           // cancela auto-calibración
}

// Reinicia los contadores visibles y lógicos
function resetCounters() {
  blinkCount = mouthCount = browCount = 0;                         // zera contadores
  blinkEl.textContent = mouthEl.textContent = browEl.textContent = '0'; // refresca UI
}

// Listeners de botones
startBtn.addEventListener('click', startCamera);                   // iniciar cámara
stopBtn.addEventListener('click', stopCamera);                     // detener cámara
calBtn.addEventListener('click', () => calibrate(75));             // calibrar (75 frames)
resetBtn.addEventListener('click', resetCounters);                 // reiniciar contadores

// Al cargar el DOM: pedimos permiso “rápido” para poblar labels de cámaras y luego listamos
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); // solicita acceso
    s.getTracks().forEach(t => t.stop());                                               // libera inmediatamente
  } catch {}                                                                            // ignoramos errores (p.ej. denegado)
  await listCams();                                                                     // llena el selector
});
