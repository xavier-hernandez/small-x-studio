// In-browser text-to-image for a diffusers-style ONNX export (text_encoder / unet / vae_decoder),
// run with ONNX Runtime Web: WebGPU or WebNN on the GPU, or WebAssembly on the CPU.
import { MODEL_CACHE as CACHE_NAME, MODEL_FILES as FILES, BACKENDS, CPU_MAX_SIDE, cpuThreads } from "./device-check.js";
import * as ort from "../lib/onnxruntime-web/ort.webgpu.min.mjs"; // self-hosted, see lib/onnxruntime-web/README.md

ort.env.wasm.wasmPaths = new URL("../lib/onnxruntime-web/", import.meta.url).href;
ort.env.logLevel = "error"; // hides the expected "some nodes were not assigned to WebGPU" warnings

const MAX_LENGTH = 77;
const VAE_SCALE = 0.18215;

// Multi-threaded WebAssembly needs a cross-origin isolated page (COOP/COEP headers from the server).
ort.env.wasm.numThreads = cpuThreads();

// ORT can run the CPU backend in a worker ("proxy"), so long CPU runs don't freeze the page. The proxy
// supports only the WebAssembly backend and is fixed once ORT initialises, so a page can't mix the CPU
// backend with a GPU one.
let proxied = null;

/** True when this page has already initialised ONNX Runtime for a backend incompatible with `backend`. */
export const needsReloadFor = backend => proxied !== null && proxied !== (backend === "wasm");

/** CLIP byte-level BPE tokenizer (port of openai/CLIP simple_tokenizer), driven by vocab.json + merges.txt. */
class ClipTokenizer {
  static #pattern = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
  static #byteToUnicode = (() => {
    const map = new Array(256), assigned = new Array(256).fill(false);
    const range = (a, b) => { for (let i = a; i <= b; i++) { map[i] = String.fromCharCode(i); assigned[i] = true; } };
    range(0x21, 0x7e); range(0xa1, 0xac); range(0xae, 0xff);
    let n = 0;
    for (let b = 0; b < 256; b++) if (!assigned[b]) map[b] = String.fromCharCode(256 + n++);
    return map;
  })();

  #cache = new Map();

  constructor(vocab, mergesText) {
    this.vocab = vocab;
    this.ranks = new Map();
    let rank = 0;
    for (const line of mergesText.split("\n")) {
      if (!line || line.startsWith("#version")) continue;
      const parts = line.trim().split(" ");
      if (parts.length === 2) { const key = parts[0] + " " + parts[1]; if (!this.ranks.has(key)) this.ranks.set(key, rank++); }
    }
    this.bos = vocab["<|startoftext|>"];
    this.eos = vocab["<|endoftext|>"];
  }

  encode(text) {
    const ids = [this.bos];
    const doc = new DOMParser().parseFromString(text ?? "", "text/html").documentElement.textContent ?? "";
    const clean = doc.replace(/\s+/g, " ").trim().toLowerCase();
    const utf8 = new TextEncoder();

    for (const [token] of clean.matchAll(ClipTokenizer.#pattern)) {
      if (token.startsWith("<|") && token in this.vocab) { ids.push(this.vocab[token]); continue; }
      const mapped = Array.from(utf8.encode(token), b => ClipTokenizer.#byteToUnicode[b]).join("");
      for (const piece of this.#bpe(mapped)) if (piece in this.vocab) ids.push(this.vocab[piece]);
      if (ids.length >= MAX_LENGTH - 1) break;
    }

    ids.length = Math.min(ids.length, MAX_LENGTH - 1);
    while (ids.length < MAX_LENGTH) ids.push(this.eos);
    return ids;
  }

  #bpe(token) {
    const cached = this.#cache.get(token);
    if (cached) return cached;

    let word = [...token];
    word[word.length - 1] += "</w>";
    while (word.length > 1) {
      let bestRank = Infinity, bestIndex = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIndex = i; }
      }
      if (bestIndex < 0) break;
      const [first, second] = [word[bestIndex], word[bestIndex + 1]];
      const merged = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) { merged.push(first + second); i++; }
        else merged.push(word[i]);
      }
      word = merged;
    }
    this.#cache.set(token, word);
    return word;
  }
}

/** DPM-Solver++(2M) over the SD 1.x scaled_linear schedule, in the k-diffusion (sigma) formulation. */
class DpmSolverScheduler {
  #sigmas; #prevDenoised = null; #prevH = 0;

  constructor(steps, trainSteps = 1000, betaStart = 0.00085, betaEnd = 0.012) {
    const alphasCumprod = new Float64Array(trainSteps);
    const s0 = Math.sqrt(betaStart), s1 = Math.sqrt(betaEnd);
    let cumprod = 1;
    for (let i = 0; i < trainSteps; i++) {
      const sqrtBeta = s0 + (s1 - s0) * i / (trainSteps - 1);
      cumprod *= 1 - sqrtBeta * sqrtBeta;
      alphasCumprod[i] = cumprod;
    }

    this.timesteps = new Array(steps);
    this.#sigmas = new Float64Array(steps + 1);
    for (let i = 0; i < steps; i++) {
      const t = steps === 1 ? trainSteps - 1 : Math.round((trainSteps - 1) * (1 - i / (steps - 1)));
      this.timesteps[i] = t;
      this.#sigmas[i] = Math.sqrt((1 - alphasCumprod[t]) / alphasCumprod[t]);
    }
  }

  get initNoiseSigma() { return this.#sigmas[0]; }

  inputScale(i) { return 1 / Math.sqrt(this.#sigmas[i] ** 2 + 1); }

  step(i, noisePred, latents) {
    const sigma = this.#sigmas[i], sigmaNext = this.#sigmas[i + 1];
    const denoised = new Float32Array(latents.length);
    for (let j = 0; j < latents.length; j++) denoised[j] = latents[j] - sigma * noisePred[j];

    if (sigmaNext === 0) {
      latents.set(denoised);
    } else {
      const h = Math.log(sigma) - Math.log(sigmaNext);
      const a = sigmaNext / sigma, b = 1 - Math.exp(-h);
      if (!this.#prevDenoised) {
        for (let j = 0; j < latents.length; j++) latents[j] = a * latents[j] + b * denoised[j];
      } else {
        const r = this.#prevH / h, c1 = 1 + 1 / (2 * r), c2 = 1 / (2 * r);
        const prev = this.#prevDenoised;
        for (let j = 0; j < latents.length; j++) latents[j] = a * latents[j] + b * (c1 * denoised[j] - c2 * prev[j]);
      }
      this.#prevH = h;
    }
    this.#prevDenoised = denoised;
  }
}

/** Fetches a file, serving it from the Cache API when already downloaded. */
async function fetchCached(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { cache = null; }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // Stream a copy straight into the cache; putting a 1.2 GB in-memory Response fails in Chrome.
  const caching = cache?.put(url, res.clone()).catch(err => console.warn("Could not cache", url, err));
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }

  await caching;
  return bytes;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class BrowserDiffusion {
  static #MODELS = {
    textEncoder: { file: FILES.textEncoder, label: "text encoder" },
    unet: { file: FILES.unet, label: "UNet" },
    vae: { file: FILES.vae, label: "VAE decoder" },
  };

  #tokenizer; #sessions = {}; #get; #onStatus; #size = null;

  /** "webgpu" | "webnn" | "wasm" */
  backend;

  /** Timings of the last successful generate(): { backend, width, height, steps, threads, setup, perStep, decode, total }. */
  lastRun = null;

  /**
   * CPU (WebAssembly) has a 4 GB heap: there, only one model is loaded at a time and the UNet runs
   * the two guidance passes separately (batch 1), which halves its working memory. See #needsSingleBatch.
   */
  get #lowMemory() { return this.backend === "wasm"; }

  /**
   * @param baseUrl folder the model files are served from, e.g. "/models/"
   * @param backend one of the keys of BACKENDS
   */
  static async load(baseUrl, backend, onStatus) {
    if (!(backend in BACKENDS)) throw new Error(`Unknown backend ${backend}`);
    if (needsReloadFor(backend)) throw new Error("Switching between the CPU and a GPU processor needs a page reload");
    if (proxied === null) ort.env.wasm.proxy = proxied = backend === "wasm";
    const self = new BrowserDiffusion();
    self.backend = backend;
    self.#onStatus = onStatus;
    self.#get = async (file, label) => {
      try {
        return await fetchCached(baseUrl + file, (got, total) =>
          onStatus?.(total ? `Downloading ${label}: ${Math.floor(got * 100 / total)}% of ${Math.round(total / 1048576)} MB`
                           : `Downloading ${label}: ${Math.round(got / 1048576)} MB`));
      } catch (err) {
        throw new Error(`Couldn't get ${label} from the server (${err.message}). If the server is still downloading the models, try again once it's ready.`);
      }
    };

    onStatus?.("Loading tokenizer");
    const decoder = new TextDecoder();
    self.#tokenizer = new ClipTokenizer(
      JSON.parse(decoder.decode(await self.#get(FILES.vocab, "vocabulary"))),
      decoder.decode(await self.#get(FILES.merges, "merges")));

    try {
      // WebGPU keeps all models loaded. WebNN builds the UNet and VAE on first use, because it needs the
      // image size. CPU loads each model only for its stage of a generation.
      if (backend === "webgpu") for (const name of ["unet", "textEncoder", "vae"]) await self.#acquire(name);
      if (backend === "webnn") await self.#acquire("textEncoder");
    } catch (err) {
      await self.dispose();
      throw err;
    }
    onStatus?.("Ready");
    return self;
  }

  async dispose() {
    for (const name of Object.keys(this.#sessions)) await this.#release(name);
  }

  #options(freeDimensionOverrides) {
    const executionProviders = {
      webgpu: ["webgpu"],
      webnn: [{ name: "webnn", deviceType: "gpu", powerPreference: "high-performance" }, "wasm"],
      wasm: ["wasm"],
    }[this.backend];
    const options = { executionProviders, graphOptimizationLevel: "all", logSeverityLevel: 3 };
    if (this.backend === "webnn") options.freeDimensionOverrides = freeDimensionOverrides;
    if (this.backend === "wasm") {
      // The CPU kernels run in fp32. Optimising would fold the fp16->fp32 weight casts into a permanent
      // fp32 copy (UNet ~2.3 GB) and overflow the 4 GB WebAssembly heap; unoptimised, weights stay fp16
      // and are cast per run. No arena, so those transient fp32 buffers are returned straight away.
      options.graphOptimizationLevel = "disabled";
      options.enableCpuMemArena = false;
    }
    return options;
  }

  /** WebNN needs every dimension fixed. */
  #overrides(name) {
    if (this.backend !== "webnn") return undefined;
    const dims = this.#size ? { height: this.#size.height / 8, width: this.#size.width / 8 } : {};
    return { textEncoder: { batch: 2 }, unet: { batch: 2, ...dims }, vae: { batch: 1, ...dims } }[name];
  }

  /** Returns the named model's session, loading it if needed. */
  async #acquire(name) {
    if (!this.#sessions[name]) {
      const { file, label } = BrowserDiffusion.#MODELS[name];
      const bytes = await this.#get(file, label);
      this.#onStatus?.(`Preparing ${label} for ${BACKENDS[this.backend].label}`);
      this.#sessions[name] = await ort.InferenceSession.create(bytes, this.#options(this.#overrides(name)));
    }
    return this.#sessions[name];
  }

  async #release(name) {
    await this.#sessions[name]?.release().catch(() => { });
    delete this.#sessions[name];
  }

  /** Called when a stage is finished with a model; frees it in low-memory mode. */
  async #done(name) {
    if (this.#lowMemory) await this.#release(name);
  }

  /**
   * Whether the UNet must run the two guidance passes separately: always on the CPU (4 GB heap), and on
   * WebGPU when the first attention layer's scores (tokens² × 8 heads × fp16 × batch 2) would exceed the
   * GPU's largest allowed buffer, e.g. 768×768 needs 2.7 GB where many GPUs allow 2 GB.
   */
  async #needsSingleBatch(lh, lw) {
    if (this.#lowMemory) return true;
    if (this.backend !== "webgpu") return false;
    let maxBuffer = 2 ** 31;
    try { maxBuffer = (await ort.env.webgpu.device)?.limits?.maxBufferSize ?? maxBuffer; } catch { }
    return (lh * lw) ** 2 * 8 * 2 * 2 > maxBuffer;
  }

  async #ensureSize(width, height) {
    if (this.backend !== "webnn" || (this.#size?.width === width && this.#size?.height === height)) return;
    await this.#release("unet");
    await this.#release("vae");
    this.#size = { width, height };
  }

  /** Returns a PNG Blob. onStep(completed, total) is called after each denoising step. */
  /** `signal` cancels between model runs (a run already in progress can't be interrupted). */
  async generate({ prompt, negativePrompt = "", steps = 20, guidance = 7.5, seed, width = 512, height = 512, signal }, onStep) {
    try {
      return await this.#generate({ prompt, negativePrompt, steps, guidance, seed, width, height, signal }, onStep);
    } finally {
      if (this.#lowMemory) await this.dispose(); // don't leave a model loaded after a failure
    }
  }

  async #generate({ prompt, negativePrompt, steps, guidance, seed, width, height, signal }, onStep) {
    if (this.backend === "wasm" && (width > CPU_MAX_SIDE || height > CPU_MAX_SIDE))
      throw new Error("The CPU processor is limited to 512×512: larger images don't fit in the browser's 4 GB WebAssembly memory.");
    const started = performance.now();
    await this.#ensureSize(width, height);

    signal?.throwIfAborted();
    // Text embeddings for [negative, prompt] in one batch.
    const textEncoder = await this.#acquire("textEncoder");
    const ids = Int32Array.from([...this.#tokenizer.encode(negativePrompt), ...this.#tokenizer.encode(prompt)]);
    const te = await textEncoder.run({ [textEncoder.inputNames[0]]: new ort.Tensor("int32", ids, [2, MAX_LENGTH]) });
    const hidden = te[textEncoder.outputNames.find(n => n.includes("last_hidden")) ?? textEncoder.outputNames[0]];
    await this.#done("textEncoder");

    const unet = await this.#acquire("unet");
    const sampleName = unet.inputNames.find(n => n.includes("sample")) ?? unet.inputNames[0];
    const timeName = unet.inputNames.find(n => n.includes("time")) ?? unet.inputNames[1];
    const hiddenName = unet.inputNames.find(n => n.includes("hidden")) ?? unet.inputNames[2];

    const [lh, lw] = [height / 8, width / 8];
    const singleBatch = await this.#needsSingleBatch(lh, lw);
    const size = 4 * lh * lw;
    // Inputs are copied for every run: in worker (proxy) mode ORT transfers the arrays to the worker,
    // which empties the originals.
    const hiddenDims = hidden.dims.slice(1);
    const predict = async (sample, batch, t, hiddenStates) => {
      signal?.throwIfAborted();
      const out = await unet.run({
        [sampleName]: new ort.Tensor("float32", sample.slice(), [batch, 4, lh, lw]),
        [timeName]: new ort.Tensor("float32", new Float32Array(batch).fill(t), [batch]),
        [hiddenName]: new ort.Tensor("float32", hiddenStates.slice(), [batch, ...hiddenDims]),
      });
      return out[unet.outputNames[0]].data;
    };

    // Batch-1 mode feeds the negative and positive embeddings separately.
    const half = hidden.data.length / 2;
    const [uncondHidden, condHidden] = [hidden.data.subarray(0, half), hidden.data.subarray(half)];

    const scheduler = new DpmSolverScheduler(steps);
    const rand = mulberry32(seed);
    const latents = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const u1 = 1 - rand(), u2 = rand();
      latents[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scheduler.initNoiseSigma;
    }

    const stepsStarted = performance.now();
    for (let step = 0; step < steps; step++) {
      const scale = scheduler.inputScale(step);
      const t = scheduler.timesteps[step];
      const scaled = latents.map(v => v * scale);

      let uncond, cond;
      if (singleBatch) {
        uncond = await predict(scaled, 1, t, uncondHidden);
        cond = await predict(scaled, 1, t, condHidden);
      } else {
        const both = new Float32Array(size * 2);
        both.set(scaled);
        both.set(scaled, size);
        const pred = await predict(both, 2, t, hidden.data);
        [uncond, cond] = [pred.subarray(0, size), pred.subarray(size)];
      }

      const noise = new Float32Array(size);
      for (let i = 0; i < size; i++) noise[i] = uncond[i] + guidance * (cond[i] - uncond[i]);
      scheduler.step(step, noise, latents);
      onStep?.(step + 1, steps);
    }
    await this.#done("unet");
    const stepsEnded = performance.now();

    signal?.throwIfAborted();
    const vae = await this.#acquire("vae");
    const scaled = latents.map(v => v / VAE_SCALE);
    const vaeOut = await vae.run({ [vae.inputNames[0]]: new ort.Tensor("float32", scaled, [1, 4, lh, lw]) });
    const image = vaeOut[vae.outputNames[0]];
    await this.#done("vae");
    const png = await toPng(image.data, width, height);

    // Timings of this run, in seconds; the page uses them to refine its time estimates.
    const ended = performance.now();
    this.lastRun = {
      backend: this.backend, width, height, steps,
      threads: this.backend === "wasm" ? ort.env.wasm.numThreads : null,
      setup: (stepsStarted - started) / 1000,
      perStep: (stepsEnded - stepsStarted) / 1000 / steps,
      decode: (ended - stepsEnded) / 1000,
      total: (ended - started) / 1000,
    };
    return png;
  }
}

function toPng(chw, width, height) {
  const plane = width * height;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) rgba[p * 4 + c] = (chw[c * plane + p] / 2 + 0.5) * 255 + 0.5;
    rgba[p * 4 + 3] = 255;
  }
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error("PNG encode failed")), "image/png"));
}
