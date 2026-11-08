// What this browser can tell us about the device, which in-browser backends it can run
// (WebGPU, WebNN, WebAssembly), and whether it meets small-sd's needs. No ONNX Runtime
// dependency, so it is cheap to run.

export const MODEL_CACHE = "small-sd-models-v1";
export const MODEL_FILES = {
  vocab: "tokenizer/vocab.json",
  merges: "tokenizer/merges.txt",
  textEncoder: "text_encoder/model.onnx",
  unet: "unet/model.onnx",
  vae: "vae_decoder/model.onnx",
};

const MODEL_BYTES = 1.61e9;   // all files in MODEL_FILES
const GB = 1024 ** 3;
const isWindows = /Windows/i.test(navigator.userAgentData?.platform ?? navigator.userAgent);

export const BACKENDS = {
  webgpu: { label: "GPU (WebGPU)" },
  webnn: { label: isWindows ? "GPU (WebNN / DirectML)" : "GPU (WebNN)" },
  wasm: { label: "CPU (WebAssembly)" },
};

/** Largest width and height the CPU (WebAssembly) backend allows: bigger images overflow its 4 GB heap. */
export const CPU_MAX_SIDE = 512;

/** Threads the WebAssembly (CPU) backend gets; must match browser-sd.js. */
// Capped at 8: on a 6-core/12-thread test CPU, 12 threads were slower per step than 8.
export const cpuThreads = () => self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;

/**
 * @param baseUrl folder the models are served (and cached) from, e.g. "/models/"
 * @returns {{
 *   verdict: "ok"|"warn"|"fail",
 *   checks: {label, value, status, note}[],            status: "ok" | "warn" | "fail" | "info"
 *   backends: { webgpu|webnn|wasm: { available: boolean, reason: string } },
 *   auto: "webgpu"|"webnn"|"wasm"                       what Auto mode will use
 * }}
 */
export async function checkDevice(baseUrl) {
  const [gpu, webnn] = await Promise.all([webGpuInfo(), webNnInfo()]);
  const threads = cpuThreads();
  const backends = {
    webgpu: gpu.backend,
    webnn,
    wasm: { available: true, reason: `${threads} thread${threads > 1 ? "s" : ""}` },
  };
  const auto = backends.webgpu.available ? "webgpu" : backends.webnn.available ? "webnn" : "wasm";

  const checks = [cpuCheck(), ramCheck(), gpu.check, processorCheck(auto, threads), await storageCheck(), await modelCheck(baseUrl)];
  const verdict = checks.some(c => c.status === "fail") ? "fail" : checks.some(c => c.status === "warn") ? "warn" : "ok";
  return { verdict, checks, backends, auto };
}

/**
 * Explains the backend that will run.
 * @param auto what Auto picks on this device; tells a forced CPU choice apart from a CPU fallback
 */
export function processorCheck(backend, auto = backend, threads = cpuThreads()) {
  const label = "Runs on";
  if (backend === "webgpu") return { label, value: "GPU via WebGPU", status: "ok", note: "" };
  if (backend === "webnn")
    return { label, value: `GPU via WebNN${isWindows ? " (DirectML)" : ""}`, status: "ok", note: "Experimental in browsers; the first image for each size compiles the model, which is slow." };
  const why = auto === "wasm" ? "No usable GPU, so the CPU is used." : "The CPU is selected even though a GPU is available.";
  return {
    label, value: `CPU via WebAssembly, ${threads} thread${threads > 1 ? "s" : ""}`, status: "warn",
    note: threads > 1 ? `${why} Expect minutes per image; see the estimate below.`
                      : `${why} This page can't use multiple threads, so expect a very long wait per image.`,
  };
}

// CPU (WebAssembly) timings measured on an AMD Ryzen 5 5600G (6 cores / 12 threads) in Chrome, at 512×512,
// by thread count. Steps scale poorly with threads (part of each step, such as casting the fp16 weights,
// runs on one thread); decoding scales well. Other thread counts are interpolated on 1/threads.
const CPU_STEP_SECONDS_512 = { 1: 46.6, 2: 36.6, 4: 28.5, 8: 24.3 };
const CPU_DECODE_SECONDS_512 = { 1: 77.6, 2: 50.3, 4: 28.3, 8: 16.3 };
const CPU_SETUP_SECONDS = 6.3; // loading the text encoder and UNet from cache, encoding the prompt
// Measured 384×384/512×512 ratios were 0.48–0.53 per step and ~0.54 for decoding (pixel ratio 0.56).
const CPU_STEP_PIXEL_EXPONENT = 1.1, CPU_DECODE_PIXEL_EXPONENT = 1.05;
const CPU_SPEED_KEY = "sd-cpu-speed";

function interpolateThreads(table, threads) {
  const points = Object.entries(table).map(([t, s]) => [+t, s]).sort((a, b) => a[0] - b[0]);
  const lo = [...points].reverse().find(p => p[0] <= threads) ?? points[0];
  const hi = points.find(p => p[0] >= threads) ?? points[points.length - 1];
  if (lo[0] === hi[0]) return lo[1];
  const f = (1 / threads - 1 / lo[0]) / (1 / hi[0] - 1 / lo[0]);
  return lo[1] + f * (hi[1] - lo[1]);
}

/** How much slower or faster this device was than the reference machine on its last CPU image (1 = same). */
function deviceSpeedFactor() {
  try { return JSON.parse(localStorage.getItem(CPU_SPEED_KEY))?.factor ?? null; } catch { return null; }
}

/** Reference-machine estimate in seconds, before any correction for this device. */
function referenceEstimate({ width, height, threads }) {
  const pixels = (width * height) / (512 * 512);
  return {
    perStep: interpolateThreads(CPU_STEP_SECONDS_512, threads) * pixels ** CPU_STEP_PIXEL_EXPONENT,
    decode: interpolateThreads(CPU_DECODE_SECONDS_512, threads) * pixels ** CPU_DECODE_PIXEL_EXPONENT,
    setup: CPU_SETUP_SECONDS,
  };
}

/**
 * Estimated CPU generation time in seconds, from the thread count (and this device's measured speed
 * once it has made a CPU image). Returns { total, perStep, decode, calibrated }.
 */
export function estimateCpu({ width, height, steps, threads = cpuThreads() }) {
  const ref = referenceEstimate({ width, height, threads });
  const factor = deviceSpeedFactor();
  const k = factor ?? 1;
  const perStep = ref.perStep * k, decode = ref.decode * k;
  return { total: ref.setup + steps * perStep + decode, perStep, decode, calibrated: factor !== null };
}

/** Learns this device's CPU speed from a finished run's timings (BrowserDiffusion.lastRun). */
export function recordCpuRun(run) {
  const ref = referenceEstimate(run);
  const factor = run.perStep / ref.perStep;
  if (!isFinite(factor) || factor <= 0) return;
  try { localStorage.setItem(CPU_SPEED_KEY, JSON.stringify({ factor, threads: run.threads, at: Date.now() })); } catch { }
}

/** "45 s", "about 8½ min", "about 1 h 10 min"; exact=true gives "1:05" style for elapsed times. */
export function formatDuration(seconds, exact = false) {
  seconds = Math.max(0, seconds);
  if (exact) {
    const s = Math.round(seconds);
    return s < 60 ? `${s} s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  if (seconds < 90) return `about ${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  if (seconds < 3600) {
    const halves = Math.round(seconds / 30);
    const whole = Math.floor(halves / 2);
    return `about ${whole}${halves % 2 ? "½" : ""} min`;
  }
  const minutes = Math.round(seconds / 60);
  return `about ${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function cpuCheck() {
  const cores = navigator.hardwareConcurrency;
  if (!cores) return { label: "CPU", value: "Not reported", status: "info", note: "" };
  return cores >= 4
    ? { label: "CPU", value: `${cores} logical cores`, status: "ok", note: "" }
    : { label: "CPU", value: `${cores} logical cores`, status: "warn", note: "Model loading and the page itself may be sluggish; 4+ cores recommended." };
}

function ramCheck() {
  // Chrome/Edge only, rounded down to a power of two and capped at 8.
  const gb = navigator.deviceMemory;
  if (gb === undefined)
    return { label: "RAM", value: "Not reported by this browser", status: "warn", note: "Loading needs about 4 GB of free RAM." };
  const value = gb >= 8 ? "8 GB or more" : `about ${gb} GB`;
  if (gb >= 8) return { label: "RAM", value, status: "ok", note: "" };
  if (gb >= 4) return { label: "RAM", value, status: "warn", note: "The 1.2 GB UNet is held in memory while loading; the tab may run out of memory." };
  return { label: "RAM", value, status: "fail", note: "At least 4 GB is needed to load the models." };
}

async function webGpuInfo() {
  const label = "GPU";
  const no = (value, reason, note = reason) =>
    ({ check: { label, value, status: "warn", note }, backend: { available: false, reason } });

  if (!window.isSecureContext) return no(rendererName() ?? "Unknown", "Needs HTTPS or localhost");
  if (!("gpu" in navigator)) return no(rendererName() ?? "Unknown", "Not supported by this browser");

  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter(); } catch { }
  if (!adapter) return no(rendererName() ?? "No GPU found", "No GPU adapter (none present, driver blocklisted, or GPU disabled)");

  const info = adapter.info ?? {};
  const f16 = adapter.features.has("shader-f16");
  const value = `${gpuName(info)} · max buffer ${Math.round(adapter.limits.maxBufferSize / 1048576)} MB${f16 ? " · fp16" : ""}`;
  if (info.isFallbackAdapter) return no(value, "Only a software (CPU) adapter is available");
  if (!f16) return no(value, "GPU lacks shader-f16, which the fp16 text encoder and UNet need");
  return {
    check: { label, value, status: "ok", note: "Browsers don't report GPU memory; about 2 GB free is needed." },
    backend: { available: true, reason: gpuName(info) },
  };
}

async function webNnInfo() {
  if (!("ml" in navigator)) return { available: false, reason: "Not supported by this browser (or not enabled)" };
  try {
    const context = await navigator.ml.createContext({ deviceType: "gpu" });
    context.destroy?.();
    return { available: true, reason: isWindows ? "DirectML" : "GPU" };
  } catch (err) {
    return { available: false, reason: `No WebNN GPU context: ${err.message ?? err}` };
  }
}

async function storageCheck() {
  const label = "Storage";
  try {
    const { quota = 0, usage = 0 } = await navigator.storage.estimate();
    const free = quota - usage;
    const value = `${(free / GB).toFixed(1)} GB available to this site`;
    return free >= MODEL_BYTES * 1.05
      ? { label, value, status: "ok", note: "" }
      : { label, value, status: "warn", note: "Not enough room to cache the models; they will be downloaded again on every visit." };
  } catch {
    return { label, value: "Not reported", status: "warn", note: "Models may not be cached between visits." };
  }
}

async function modelCheck(baseUrl) {
  const label = "Models";
  try {
    const cache = await caches.open(MODEL_CACHE);
    const hits = await Promise.all(Object.values(MODEL_FILES).map(f => cache.match(new URL(baseUrl + f, location.href).href)));
    if (hits.every(Boolean)) return { label, value: "Cached in this browser", status: "ok", note: "" };
  } catch { }
  return { label, value: "Not downloaded yet", status: "info", note: "The first run downloads about 1.6 GB." };
}

/** Human-readable GPU name, preferring WebGL's renderer string when it is the same vendor as the WebGPU adapter. */
function gpuName(info) {
  const renderer = rendererName();
  const vendor = (info.vendor ?? "").toLowerCase();
  if (renderer && (!vendor || renderer.toLowerCase().includes(vendor))) return renderer;
  const parts = [info.vendor, info.architecture, info.description].filter(Boolean);
  return parts.length ? parts.join(" ") : renderer ?? "Unknown GPU";
}

function rendererName() {
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    const raw = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl?.getParameter(gl.RENDERER);
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    if (!raw) return null;
    // "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 (0x00002786) Direct3D11 vs_5_0 ps_5_0, D3D11)" -> "NVIDIA GeForce RTX 4070"
    const angle = /^ANGLE \([^,]+, (.+?)(?: \(0x[0-9a-f]+\))?(?: Direct3D| OpenGL| Vulkan| Metal|,|\)$)/i.exec(raw);
    return angle ? angle[1].trim() : raw;
  } catch {
    return null;
  }
}
