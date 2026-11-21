// In-browser 16×16 pixel-art sprites with AnticPan/mini-pixel-art: a tiny pixel-space DiT (2–9M parameters)
// sampled with flow matching. Port of the repo's mini_pixel_art_onnx.py. No VAE and no CLIP: captions are
// split into words and looked up in vocab.json, and every pixel is snapped to the 4,096-colour palette.
import { ort, initRuntime, fetchCached, mulberry32 } from "./browser-sd.js";

const PAD = 0, UNK = 1, NULL = 2;
const TEXT_LEN = 12, SIDE = 16, PIXELS = SIDE * SIDE, SIZE = 4 * PIXELS; // [4, 16, 16] RGBA in [-1, 1]
const WORD = /[a-z0-9]+(?:[-'][a-z0-9]+)*/g;
export const UPSCALE = 8;

/** Sizes in the repo; the page uses l-p1, the best quality. */
const PIXEL_VARIANTS = ["s-p1", "m-p1", "l-p1", "m-p2"];

/** Where the pixel-art model runs for the processor the user picked. */
export const pixelBackend = processor => processor === "webgpu" ? "webgpu" : "wasm";

/** Lower-case words, as in the training captions. */
export const captionWords = caption => caption.toLowerCase().match(WORD) ?? [];

export class PixelArt {
  #session; #vocab; #palette; #paletteF; #sampler;

  /** Where it runs: "webgpu" | "wasm" */
  backend;
  /** The processor the user picked when it was loaded */
  processor;
  variant;

  /**
   * @param baseUrl folder the pixel-art files are served from, e.g. "/models/mini-pixel-art/"
   * @param variant one of PIXEL_VARIANTS (the server must have downloaded it)
   * @param processor the processor the user picked ("webgpu" | "webnn" | "wasm"). WebGPU runs on WebGPU;
   *   anything else runs on the CPU, since WebNN gains nothing for a model this small and its int64 support is patchy.
   */
  static async load(baseUrl, variant, processor, onStatus) {
    if (!PIXEL_VARIANTS.includes(variant)) throw new Error(`Unknown pixel-art model ${variant}`);
    initRuntime(processor);
    const backend = pixelBackend(processor);
    const get = async (file, label) => {
      try {
        return await fetchCached(baseUrl + file, (got, total) =>
          onStatus?.(total ? `Downloading ${label}: ${Math.floor(got * 100 / total)}%` : `Downloading ${label}`));
      } catch (err) {
        throw new Error(`Couldn't get ${label} from the server (${err.message}). If the server is still downloading the models, try again once it's ready.`);
      }
    };
    const self = new PixelArt();
    self.backend = backend;
    self.processor = processor;
    self.variant = variant;
    const text = bytes => JSON.parse(new TextDecoder().decode(bytes));

    onStatus?.("Loading vocabulary and palette");
    self.#vocab = text(await get("vocab.json", "vocabulary"));
    self.#palette = await get("palette.bin", "palette"); // 4097 × RGB; index 0 = transparent
    self.#paletteF = Float32Array.from(self.#palette.subarray(3), v => v / 127.5 - 1);
    self.#sampler = text(await get(`${variant}/config.json`, "model settings")).sampler;

    const bytes = await get(`${variant}/model.onnx`, `pixel-art model (${variant})`);
    onStatus?.("Preparing the pixel-art model");
    self.#session = await ort.InferenceSession.create(bytes, {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3,
    });
    return self;
  }

  /** The model's own sampler settings from config.json: { steps, cfg, cfg_alpha, t_stop }. */
  get defaults() { return { ...this.#sampler }; }

  async dispose() { await this.#session?.release().catch(() => { }); this.#session = null; }

  /** Words of the caption the model doesn't know; they're ignored. */
  unknownWords(caption) { return captionWords(caption).filter(w => !(w in this.#vocab)); }

  #encode(caption) {
    const ids = captionWords(caption).map(w => this.#vocab[w] ?? UNK).slice(0, TEXT_LEN);
    while (ids.length < TEXT_LEN) ids.push(PAD);
    return ids;
  }

  /**
   * Generates `count` sprites for one caption. Image i uses seed `seed + i`, so each can be reproduced alone.
   * @returns {Promise<{ seed: number, png: Blob, pngLarge: Blob }[]>} 16×16 PNGs and 8× nearest-neighbour copies
   */
  async generate({ prompt, steps, guidance, count = 1, seed, signal }, onStep) {
    const { cfg_alpha: cfgAlpha, t_stop: tStop } = this.#sampler;
    const n = count, batch = 2 * n;
    const seeds = Array.from({ length: n }, (_, i) => (seed + i) >>> 0);

    // Captions for the guided half, then the "null" caption for the unguided half.
    const ids = new BigInt64Array(batch * TEXT_LEN);
    const caption = this.#encode(prompt).map(BigInt);
    for (let i = 0; i < n; i++) { ids.set(caption, i * TEXT_LEN); ids[(n + i) * TEXT_LEN] = BigInt(NULL); }

    // Starting noise, one generator per image.
    const z = new Float32Array(n * SIZE);
    seeds.forEach((s, i) => {
      const rand = mulberry32(s);
      for (let k = 0; k < SIZE; k++) {
        const u1 = 1 - rand(), u2 = rand();
        z[i * SIZE + k] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
    });

    // Euler from t = 1 to t_stop, then one jump to the predicted image. The models were trained with
    // max(t, 0.05), so the grid must not go below t_stop.
    const ts = steps > 1 ? [...Array.from({ length: steps }, (_, i) => 1 + (tStop - 1) * i / (steps - 1)), 0] : [1, 0];
    const s = this.#session;
    for (let k = 0; k < ts.length - 1; k++) {
      signal?.throwIfAborted();
      const [t, tNext] = [ts[k], ts[k + 1]];
      const zz = new Float32Array(batch * SIZE);
      zz.set(z); zz.set(z, n * SIZE);
      // Inputs are new arrays each run: in worker (proxy) mode ORT transfers them, which empties the originals.
      const out = await s.run({
        z: new ort.Tensor("float32", zz, [batch, 4, SIDE, SIDE]),
        t: new ort.Tensor("float32", new Float32Array(batch).fill(t), [batch]),
        cap_ids: new ort.Tensor("int64", ids.slice(), [batch, TEXT_LEN]),
      });
      const x = out.x.data, r = tNext / t;
      for (let i = 0; i < n * SIZE; i++) {
        const scale = Math.floor((i % SIZE) / PIXELS) === 3 ? cfgAlpha : guidance; // channel 3 is alpha
        const xu = x[n * SIZE + i];
        z[i] = r * z[i] + (1 - r) * (xu + scale * (x[i] - xu));
      }
      onStep?.(k + 1, ts.length - 1);
    }

    return Promise.all(seeds.map(async (seed, i) => {
      const rgba = this.#toRgba(z.subarray(i * SIZE, (i + 1) * SIZE));
      return { seed, png: await toPng(rgba, 1), pngLarge: await toPng(rgba, UPSCALE) };
    }));
  }

  /** [4,16,16] in [-1,1] → 16×16 RGBA: each pixel snapped to the nearest palette colour; alpha < 0 is transparent. */
  #toRgba(x) {
    const pal = this.#paletteF, colours = pal.length / 3;
    const rgba = new Uint8ClampedArray(PIXELS * 4);
    for (let p = 0; p < PIXELS; p++) {
      if (x[3 * PIXELS + p] < 0) continue; // stays transparent black
      const r = x[p], g = x[PIXELS + p], b = x[2 * PIXELS + p];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < colours; c++) {
        const dr = r - pal[3 * c], dg = g - pal[3 * c + 1], db = b - pal[3 * c + 2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = c; }
      }
      const src = 3 * (best + 1); // palette.bin index 0 is the transparent entry
      rgba.set([this.#palette[src], this.#palette[src + 1], this.#palette[src + 2], 255], 4 * p);
    }
    return rgba;
  }
}

function toPng(rgba, scale) {
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = SIDE;
  sprite.getContext("2d").putImageData(new ImageData(rgba, SIDE, SIDE), 0, 0);
  let canvas = sprite;
  if (scale > 1) {
    canvas = document.createElement("canvas");
    canvas.width = canvas.height = SIDE * scale;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(sprite, 0, 0, canvas.width, canvas.height);
  }
  return new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error("PNG encode failed")), "image/png"));
}
