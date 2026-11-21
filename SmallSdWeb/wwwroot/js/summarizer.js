// In-browser summarization with DistilBART-CNN (sshleifer/distilbart-cnn-6-6, ONNX export Xenova/distilbart-cnn-6-6),
// run with ONNX Runtime Web on the CPU, with int8 ("fast") or fp32 ("best") weights. The int8 weights use
// MatMulInteger, which only ORT's WebAssembly backend runs well, so there's no GPU path. Meant to run in a Web Worker
// (see summarize-worker.js), where blocking is harmless.
import { ort, fetchCached } from "./browser-sd.js";

const BOS = 0, PAD = 1, EOS = 2, UNK = 3, MASK = 50264;
const SPECIAL = new Set([BOS, PAD, EOS, UNK, MASK]);
const DECODER_START = EOS;           // BART starts decoding from </s>
const MAX_INPUT = 1024;              // position embeddings
const CHUNK_TOKENS = MAX_INPUT - 2;  // room for <s> and </s>
const LAYERS = 6;
const LENGTH_PENALTY = 2.0, NO_REPEAT_NGRAM = 3;

/** Summary lengths in tokens (about ¾ of a word each). Medium is the model's own setting. */
export const LENGTHS = {
  short: { min: 30, max: 80 },
  medium: { min: 56, max: 142 },
  long: { min: 100, max: 250 },
};

const TOKENIZER = { vocab: "vocab.json", merges: "merges.txt" };

/**
 * The two versions of the model. The ids match the server's /api/status keys. Both run at about the same speed on
 * the CPU; "best" (fp32) writes better summaries but is a 1.1 GB download and takes about twice as long to load.
 */
export const VARIANTS = {
  fast: { status: "summarizer", label: "Fast", size: "290 MB",
    encoder: "onnx/encoder_model_quantized.onnx", decoder: "onnx/decoder_model_merged_quantized.onnx" },
  best: { status: "summarizer-full", label: "Best", size: "1.1 GB",
    encoder: "onnx/encoder_model.onnx", decoder: "onnx/decoder_model_merged.onnx" },
};

/** GPT-2 / RoBERTa byte-level BPE, driven by vocab.json + merges.txt (as BART uses). */
export class Gpt2Tokenizer {
  static #pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  #encoder; #decoder = []; #ranks = new Map(); #cache = new Map(); #byteToChar = []; #charToByte = new Map();

  constructor(vocab, merges) {
    this.#encoder = vocab;
    for (const [token, id] of Object.entries(vocab)) this.#decoder[id] = token;
    merges.split("\n").slice(1).forEach((line, rank) => { if (line) this.#ranks.set(line, rank); });
    // Printable stand-ins for every byte, so BPE never sees whitespace or control characters.
    const bytes = [];
    for (let i = 0x21; i <= 0x7e; i++) bytes.push(i);
    for (let i = 0xa1; i <= 0xac; i++) bytes.push(i);
    for (let i = 0xae; i <= 0xff; i++) bytes.push(i);
    const chars = [...bytes];
    for (let b = 0, n = 0; b < 256; b++) if (!bytes.includes(b)) { bytes.push(b); chars.push(256 + n++); }
    bytes.forEach((b, i) => { this.#byteToChar[b] = String.fromCharCode(chars[i]); this.#charToByte.set(String.fromCharCode(chars[i]), b); });
  }

  #bpe(word) {
    const cached = this.#cache.get(word);
    if (cached) return cached;
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.#ranks.get(parts[i] + " " + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; best = i; }
      }
      if (best < 0) break;
      const [a, b] = [parts[best], parts[best + 1]], merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; }
        else merged.push(parts[i]);
      }
      parts = merged;
    }
    const ids = parts.map(p => this.#encoder[p] ?? UNK);
    this.#cache.set(word, ids);
    return ids;
  }

  /** Token ids without <s> / </s>. */
  encode(text) {
    const utf8 = new TextEncoder(), ids = [];
    for (const piece of text.match(Gpt2Tokenizer.#pattern) ?? [])
      ids.push(...this.#bpe(Array.from(utf8.encode(piece), b => this.#byteToChar[b]).join("")));
    return ids;
  }

  decode(ids) {
    const text = ids.filter(id => !SPECIAL.has(id)).map(id => this.#decoder[id] ?? "").join("");
    return new TextDecoder().decode(Uint8Array.from([...text], c => this.#charToByte.get(c) ?? 0x3f));
  }
}

/**
 * Splits text into pieces of at most CHUNK_TOKENS tokens, at sentence or paragraph ends where possible.
 * @returns {number[][]} token ids per chunk
 */
function chunk(tokenizer, text) {
  // Each piece keeps its leading whitespace, so the pieces' tokens join up like the whole text's would.
  const sentences = text.match(/\s*[^.!?\n]+(?:[.!?]+["')\]]*|\n+|$)|\s*[.!?\n]+/g) ?? [];
  const chunks = [];
  let current = [];
  for (const sentence of sentences) {
    let ids = tokenizer.encode(sentence);
    if (!ids.length) continue;
    if (current.length + ids.length > CHUNK_TOKENS && current.length) { chunks.push(current); current = []; }
    while (ids.length > CHUNK_TOKENS) { chunks.push(ids.slice(0, CHUNK_TOKENS)); ids = ids.slice(CHUNK_TOKENS); }
    current = current.concat(ids);
  }
  if (current.length) chunks.push(current);
  return chunks;
}

/**
 * Cleans up the model's output: CNN/DailyMail-trained models write "centre ." (from the dataset's tokenized text),
 * and beam search with the model's length penalty often stops mid-sentence, so a trailing fragment is dropped
 * when at least one whole sentence comes before it.
 */
export function tidy(text) {
  text = text.trim().replace(/\s+([.,!?;:%)])/g, "$1").replace(/([($])\s+/g, "$1").replace(/\s{2,}/g, " ");
  if (/[.!?]["')\]]*$/.test(text)) return text;
  const end = Math.max(...[".", "!", "?"].map(p => text.lastIndexOf(p + " ")));
  return end > 0 ? text.slice(0, end + 1) : text + "…";
}

/**
 * Waits one turn of the event loop. ORT's CPU run() finishes without yielding to it, so without this a message
 * (such as "cancel") sent during a summary would only be handled after the whole summary. A MessageChannel
 * round trip avoids setTimeout's minimum delay.
 */
function yieldToEvents() {
  return new Promise(resolve => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => { port1.close(); resolve(); };
    port2.postMessage(null);
  });
}

function cancelledError() { return new DOMException("Cancelled", "AbortError"); }

export class Summarizer {
  #tokenizer; #encoder; #decoder;

  /** A key of VARIANTS */
  variant;

  /**
   * @param baseUrl folder the model files are served from, e.g. "/models/distilbart-cnn-6-6/"
   * @param variant a key of VARIANTS
   * @param onStatus called with a status line while files download and sessions are created
   */
  static async load(baseUrl, variant, onStatus) {
    const files = VARIANTS[variant];
    if (!files) throw new Error(`Unknown summarizer version ${variant}`);
    const get = async (file, label) => {
      try {
        return await fetchCached(baseUrl + file, (got, total) =>
          onStatus?.(total ? `Downloading ${label}: ${Math.floor(got * 100 / total)}% of ${Math.round(total / 1048576)} MB`
                           : `Downloading ${label}: ${Math.round(got / 1048576)} MB`));
      } catch (err) {
        throw new Error(`Couldn't get ${label} from the server (${err.message}). If the server is still downloading the models, try again once it's ready.`);
      }
    };
    const self = new Summarizer();
    self.variant = variant;
    const decoder = new TextDecoder();
    onStatus?.("Loading tokenizer");
    self.#tokenizer = new Gpt2Tokenizer(
      JSON.parse(decoder.decode(await get(TOKENIZER.vocab, "vocabulary"))),
      decoder.decode(await get(TOKENIZER.merges, "merges")));
    const options = { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 };
    try {
      for (const name of ["encoder", "decoder"]) {
        const bytes = await get(files[name], name);
        onStatus?.(`Preparing the ${name}`);
        const session = await ort.InferenceSession.create(bytes, options);
        if (name === "encoder") self.#encoder = session; else self.#decoder = session;
      }
    } catch (err) {
      await self.dispose();
      throw err;
    }
    return self;
  }

  /** Frees the sessions (the fp32 version holds about 1.1 GB). */
  async dispose() {
    for (const s of [this.#encoder, this.#decoder]) await s?.release().catch(() => { });
    this.#encoder = this.#decoder = null;
  }

  /** Token count of `text`, for the input limit and chunking. */
  countTokens(text) { return this.#tokenizer.encode(text).length; }

  /**
   * Summarizes `text`. Text longer than the model's 1,024-token window is split into chunks, each summarized
   * separately; the result joins them as paragraphs.
   * @param length a key of LENGTHS
   * @param beams 4 = the model's beam search (best quality), 1 = greedy (about 3× faster)
   * @param isCancelled checked between decoding steps
   * @param onProgress ({ chunk, chunks, step, maxSteps }) after each decoding step
   */
  async summarize(text, { length = "medium", beams = 4, isCancelled = () => false, onProgress } = {}) {
    const { min, max } = LENGTHS[length] ?? LENGTHS.medium;
    const started = performance.now();
    const chunks = chunk(this.#tokenizer, text);
    if (!chunks.length) throw new Error("There's no text to summarize.");
    const parts = [];
    for (let c = 0; c < chunks.length; c++) {
      const ids = await this.#generate([BOS, ...chunks[c], EOS], { min, max, beams, isCancelled,
        onStep: (step, maxSteps) => onProgress?.({ chunk: c + 1, chunks: chunks.length, step, maxSteps }) });
      parts.push(tidy(this.#tokenizer.decode(ids)));
    }
    return {
      summary: parts.join("\n\n"),
      inputTokens: chunks.reduce((n, c) => n + c.length, 0),
      chunks: chunks.length,
      seconds: (performance.now() - started) / 1000,
    };
  }

  /** Beam search with the model's processors: forced <s> first, no repeated 3-grams, min/max length, forced </s>. */
  async #generate(inputIds, { min, max, beams, isCancelled, onStep }) {
    const n = inputIds.length, B = beams;
    const enc = await this.#encoder.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds, BigInt), [1, n]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
    });
    const hidden = enc.last_hidden_state.data, width = hidden.length / n;
    const hiddenB = new Float32Array(B * hidden.length);
    for (let b = 0; b < B; b++) hiddenB.set(hidden, b * hidden.length);
    const encoderHidden = new ort.Tensor("float32", hiddenB, [B, n, width]);
    const encoderMask = new ort.Tensor("int64", new BigInt64Array(B * n).fill(1n), [B, n]);

    const heads = this.#decoder.inputMetadata?.find?.(m => m.name === "past_key_values.0.decoder.key")?.shape?.[1] ?? 16;
    const headDim = width / heads;
    const kinds = ["decoder.key", "decoder.value", "encoder.key", "encoder.value"];
    const empty = new ort.Tensor("float32", new Float32Array(0), [B, heads, 0, headDim]);
    let past = Object.fromEntries(Array.from({ length: LAYERS }, (_, l) => kinds.map(k => [`past_key_values.${l}.${k}`, empty])).flat());
    // After the first step only the logits and the decoder's growing cache are needed; the encoder's cache never changes.
    const fetches = ["logits", ...Array.from({ length: LAYERS }, (_, l) => [`present.${l}.decoder.key`, `present.${l}.decoder.value`]).flat()];

    let seqs = Array.from({ length: B }, () => [DECODER_START]);
    let scores = Array.from({ length: B }, (_, b) => b === 0 ? 0 : -1e9); // beams start identical; keep one
    const finished = []; // { ids, score }
    const addFinished = (ids, sum) => {
      finished.push({ ids, score: sum / ids.length ** LENGTH_PENALTY });
      finished.sort((a, b) => b.score - a.score);
      if (finished.length > B) finished.pop();
    };

    for (let len = 1; len < max; len++) {
      await yieldToEvents(); // lets a pending "cancel" message in
      if (isCancelled()) throw cancelledError();
      const first = len === 1;
      const out = await this.#decoder.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(seqs, s => BigInt(s[s.length - 1])), [B, 1]),
        encoder_hidden_states: encoderHidden,
        encoder_attention_mask: encoderMask,
        use_cache_branch: new ort.Tensor("bool", [!first], [1]),
        ...past,
      }, first ? undefined : fetches);
      const logits = out.logits.data, vocab = logits.length / B;

      // Best 2B candidates over every beam and token, from log-probabilities after the processors.
      const cand = [];
      for (let b = 0; b < B; b++) {
        if (scores[b] <= -1e9 && !first) continue;
        const lp = logSoftmax(logits.subarray(b * vocab, (b + 1) * vocab));
        this.#process(lp, seqs[b], len, min, max);
        for (let t = 0; t < vocab; t++) {
          const s = scores[b] + lp[t];
          if (cand.length < 2 * B || s > cand[cand.length - 1].s) {
            let i = cand.length < 2 * B ? cand.length : cand.length - 1;
            while (i > 0 && cand[i - 1].s < s) { cand[i] = cand[i - 1]; i--; }
            cand[i] = { b, t, s };
            if (cand.length > 2 * B) cand.length = 2 * B;
          }
        }
      }

      const next = [];
      for (let rank = 0; rank < cand.length && next.length < B; rank++) {
        const { b, t, s } = cand[rank];
        if (!isFinite(s)) break;
        if (t === EOS) { if (rank < B) addFinished(seqs[b], s); }
        else next.push({ b, t, s });
      }
      onStep?.(len, max - 1);
      // early_stopping: done once B finished summaries exist.
      if (finished.length >= B || !next.length) break;

      // Reorder the decoder cache to follow the surviving beams; the encoder cache is the same for every beam.
      const order = next.map(c => c.b);
      past = { ...past };
      for (let l = 0; l < LAYERS; l++) {
        for (const kv of ["key", "value"]) {
          const present = out[`present.${l}.decoder.${kv}`];
          past[`past_key_values.${l}.decoder.${kv}`] = reorder(present, order);
          if (first) past[`past_key_values.${l}.encoder.${kv}`] = out[`present.${l}.encoder.${kv}`];
        }
      }
      seqs = next.map(c => [...seqs[c.b], c.t]);
      scores = next.map(c => c.s);
      while (seqs.length < B) { seqs.push([...seqs[0]]); scores.push(-1e9); } // only if candidates ran out
    }
    if (finished.length < B) seqs.forEach((s, b) => { if (scores[b] > -1e9) addFinished(s, scores[b]); });
    return finished[0].ids;
  }

  /** Logits processors, applied in place to one beam's log-probabilities. */
  #process(lp, seq, len, min, max) {
    const only = token => { lp.fill(-Infinity); lp[token] = 0; };
    if (len === 1) return only(BOS);         // forced_bos_token_id
    if (len === max - 1) return only(EOS);   // forced_eos_token_id
    if (len < min) lp[EOS] = -Infinity;      // min_length
    // no_repeat_ngram_size: ban any token that would repeat a 3-gram already in this beam.
    const k = NO_REPEAT_NGRAM - 1;
    if (seq.length >= k) {
      const tail = seq.slice(-k);
      for (let i = 0; i + k < seq.length; i++) {
        let match = true;
        for (let j = 0; j < k; j++) if (seq[i + j] !== tail[j]) { match = false; break; }
        if (match) lp[seq[i + k]] = -Infinity;
      }
    }
  }
}

function logSoftmax(logits) {
  let maxV = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > maxV) maxV = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - maxV);
  const lse = maxV + Math.log(sum), out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] - lse;
  return out;
}

/** Copies a [B, heads, seq, dim] cache so beam i holds what beam order[i] held. */
function reorder(tensor, order) {
  const data = tensor.data, block = data.length / order.length;
  if (order.every((b, i) => b === i)) return tensor;
  const outData = new Float32Array(data.length);
  order.forEach((b, i) => outData.set(data.subarray(b * block, (b + 1) * block), i * block));
  return new ort.Tensor("float32", outData, tensor.dims);
}
