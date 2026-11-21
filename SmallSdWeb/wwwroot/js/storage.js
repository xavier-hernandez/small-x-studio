// What this site keeps in the browser, and deleting it: the model files in the Cache API, and the settings in
// localStorage. No ONNX Runtime dependency, so the About page can use it cheaply.
import { MODEL_CACHE } from "./device-check.js";

const SUM = "/models/distilbart-cnn-6-6/";

/** Cached files are grouped by their URL; the first matching group wins. */
const GROUPS = [
  { id: "mini-pixel-art", label: "Mini Pixel Art", tool: "Generate images", match: p => p.startsWith("/models/mini-pixel-art/") },
  { id: "summarizer-fast", label: "Summarizer · Fast", tool: "Summarize text", match: p => p.startsWith(SUM + "onnx/") && p.endsWith("_quantized.onnx") },
  { id: "summarizer-best", label: "Summarizer · Best", tool: "Summarize text", match: p => p.startsWith(SUM + "onnx/") },
  // The tokenizer both summarizer versions use; shown with one of them (see listModels).
  { id: "summarizer-shared", label: "Summarizer tokenizer", tool: "Summarize text", match: p => p.startsWith(SUM) },
  { id: "small-sd", label: "Small SD", tool: "Generate images", match: p => p.startsWith("/models/") },
  { id: "other", label: "Other files", tool: "", match: () => true },
];

async function entries() {
  if (!("caches" in self)) return [];
  const cache = await caches.open(MODEL_CACHE);
  const out = [];
  for (const request of await cache.keys()) {
    const path = new URL(request.url).pathname;
    const response = await cache.match(request);
    let size = Number(response?.headers.get("Content-Length")) || 0;
    if (!size && response) size = (await response.blob()).size;
    out.push({ request, path, size, group: GROUPS.find(g => g.match(path)).id });
  }
  return out;
}

/**
 * The models stored in this browser.
 * @returns {Promise<{ id, label, tool, files: number, bytes: number }[]>} only groups with files, in GROUPS order
 */
export async function listModels() {
  const all = await entries();
  const rows = new Map(GROUPS.map(g => [g.id, { id: g.id, label: g.label, tool: g.tool, files: 0, bytes: 0 }]));
  for (const e of all) { const r = rows.get(e.group); r.files++; r.bytes += e.size; }
  // Fold the shared tokenizer into a summarizer version that's present, so it doesn't look like a separate model.
  const shared = rows.get("summarizer-shared");
  const host = [rows.get("summarizer-fast"), rows.get("summarizer-best")].find(r => r.files);
  if (host && shared.files) { host.files += shared.files; host.bytes += shared.bytes; shared.files = shared.bytes = 0; }
  return [...rows.values()].filter(r => r.files);
}

/** Deletes one group's files. The summarizer tokenizer goes once neither summarizer version is left. */
export async function deleteModel(id) {
  const cache = await caches.open(MODEL_CACHE);
  const all = await entries();
  for (const e of all.filter(e => e.group === id)) await cache.delete(e.request);
  const summarizerLeft = all.some(e => e.group !== id && (e.group === "summarizer-fast" || e.group === "summarizer-best"));
  if (id.startsWith("summarizer") && !summarizerLeft)
    for (const e of all.filter(e => e.group === "summarizer-shared")) await cache.delete(e.request);
}

/** Deletes every cache this site made and its saved settings. */
export async function clearAll() {
  if ("caches" in self) for (const name of await caches.keys()) await caches.delete(name);
  try { localStorage.clear(); } catch { }
}

/** What the browser reports for this site: { usage, quota } in bytes, or null if it doesn't say. */
export async function storageEstimate() {
  try { const { usage, quota } = await navigator.storage.estimate(); return { usage, quota }; } catch { return null; }
}

export function formatBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} B`;
}
