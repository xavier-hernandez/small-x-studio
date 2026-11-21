// Runs the summarizer in a Web Worker: the model and its ~200 MB decoding cache stay on this thread, and the page
// stays responsive. ONNX Runtime still uses multiple CPU threads from here (the page is cross-origin isolated).
//
// Messages in:  { type: "summarize", text, variant, length, beams }   { type: "cancel" }
// Messages out: { type: "status", text }   { type: "progress", chunk, chunks, step, maxSteps }
//               { type: "result", summary, inputTokens, chunks, seconds }   { type: "error", message, cancelled }
import { Summarizer } from "./summarizer.js";

const BASE = "/models/distilbart-cnn-6-6/";
let engine = null, cancelled = false;

self.onmessage = async ({ data }) => {
  if (data.type === "cancel") { cancelled = true; return; }
  if (data.type !== "summarize") return;
  cancelled = false;
  try {
    // One version is loaded at a time: switching frees the other first, since the fp32 one alone is 1.1 GB.
    if (engine && engine.variant !== data.variant) {
      self.postMessage({ type: "status", text: "Switching model…" });
      await engine.dispose();
      engine = null;
    }
    engine ??= await Summarizer.load(BASE, data.variant, text => self.postMessage({ type: "status", text }));
    const result = await engine.summarize(data.text, {
      length: data.length, beams: data.beams,
      isCancelled: () => cancelled,
      onProgress: p => self.postMessage({ type: "progress", ...p }),
    });
    self.postMessage({ type: "result", ...result });
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message ?? String(err), cancelled: err?.name === "AbortError" });
  }
};
