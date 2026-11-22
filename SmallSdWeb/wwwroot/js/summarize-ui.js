// The summarizer controls shared by /summarize and /summarize-file: model, length and search options, the
// Summarize and Cancel buttons, progress, result cards and the model status line. Both pages use the same element
// ids. The summarizer itself runs in js/summarize-worker.js, so the page never loads ONNX Runtime.
import { initHelp, watchModel } from "/js/help.js";

const $ = id => document.getElementById(id);
const CHUNK_TOKENS = 1022, TOKENS_PER_WORD = 1.35; // estimate only; the worker counts exact tokens

// Mirrors VARIANTS in js/summarizer.js. `status` is the server's /api/status key.
const VARIANTS = {
  fast: { status: "summarizer", label: "Fast", size: "290 MB" },
  best: { status: "summarizer-full", label: "Best", size: "1.1 GB" },
};
const SEARCH = { 4: "Thorough", 1: "Quick" };
const load = (key, fallback) => { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } };
const save = (key, value) => { try { localStorage.setItem(key, value); } catch { } };

/** Words in the text, and roughly how many parts (one summary paragraph each) it will be split into. */
export function estimate(text) {
  text = text.trim();
  const words = text ? text.split(/\s+/).length : 0;
  return { words, parts: Math.max(1, Math.ceil(words * TOKENS_PER_WORD / CHUNK_TOKENS)) };
}

export function describeLength(text) {
  const { words, parts } = estimate(text);
  return !words ? "0 words"
    : `${words.toLocaleString()} word${words > 1 ? "s" : ""}` + (parts > 1 ? ` · about ${parts} parts, one paragraph each` : "");
}

/**
 * Wires up the shared controls.
 * - getText(): the text to summarize.
 * - lock: ids of the page's own controls to disable while a summary is being written.
 * - canStart(): extra page conditions for enabling Summarize (default: there is text).
 * - source(): an optional label for the result card, such as the file name.
 * - onBusy(busy): called when a run starts and ends.
 * Returns { refresh, setMessage, busy() }.
 */
export function setupSummarizer({ getText, lock = [], canStart = () => !!getText().trim(), source = () => "", onBusy = () => { } }) {
  let ready = false, busy = false, worker = null, job = null;
  $("variant").value = load("summarizer-variant", "fast") in VARIANTS ? load("summarizer-variant", "fast") : "fast";
  $("size").textContent = VARIANTS[$("variant").value].size;

  function refresh() {
    $("go").disabled = busy || !ready || !canStart();
    $("cancel").hidden = !busy;
    for (const id of ["variant", "length", "beams", ...lock]) if ($(id)) $(id).disabled = busy;
  }
  function setMessage(text, isError) { $("message").textContent = text; $("message").className = "message" + (isError ? " error" : ""); }

  /** The summarizer runs in a worker, created on first use and kept, so the model loads once per visit. */
  function getWorker() {
    if (worker) return worker;
    worker = new Worker("/js/summarize-worker.js", { type: "module" });
    worker.onmessage = ({ data }) => job?.[data.type]?.(data);
    worker.onerror = e => { worker = null; job?.error({ message: e.message || "The summarizer stopped unexpectedly." }); };
    return worker;
  }

  $("form").onsubmit = e => {
    e.preventDefault();
    if ($("go").disabled) return;
    const req = { text: getText(), variant: $("variant").value, length: $("length").value, beams: +$("beams").value };
    const label = source();
    busy = true; refresh(); onBusy(true);
    $("bar").style.width = "0"; $("cancel").disabled = false; $("cancel").textContent = "Cancel";
    setMessage("Starting…");
    const done = () => { job = null; busy = false; refresh(); onBusy(false); };
    job = {
      status: ({ text }) => setMessage(text),
      progress: ({ chunk, chunks, step, maxSteps }) => {
        if (job.cancelling) return;
        $("bar").style.width = `${((chunk - 1 + step / maxSteps) / chunks) * 100}%`;
        setMessage(`Writing the summary${chunks > 1 ? ` · part ${chunk} of ${chunks}` : ""} · ${step} tokens`);
      },
      result: r => {
        $("bar").style.width = "100%";
        setMessage(`Done in ${r.seconds < 60 ? r.seconds.toFixed(1) + " s" : Math.round(r.seconds / 60) + " min"}`);
        addResult(r, req, label);
        done();
      },
      error: ({ message, cancelled }) => {
        $("bar").style.width = "0";
        setMessage(cancelled ? "Cancelled." : message, !cancelled);
        done();
      },
    };
    getWorker().postMessage({ type: "summarize", ...req });
  };

  // Decoding can't stop mid-step, so cancelling takes effect after the current token.
  $("cancel").onclick = () => {
    if (!job) return;
    job.cancelling = true;
    worker?.postMessage({ type: "cancel" });
    $("cancel").disabled = true; $("cancel").textContent = "Cancelling…";
    setMessage("Cancelling…");
  };

  initHelp();
  const status = watchModel(() => VARIANTS[$("variant").value].status, $("status"), s => { ready = s?.phase === "ready"; refresh(); });
  $("variant").onchange = () => {
    save("summarizer-variant", $("variant").value);
    $("size").textContent = VARIANTS[$("variant").value].size;
    status.show();
  };
  refresh();
  return { refresh, setMessage, busy: () => busy };
}

function addResult({ summary, inputTokens, chunks, seconds }, req, label) {
  $("empty")?.remove();
  const card = document.createElement("article");
  card.className = "result panel";
  card.innerHTML = `<p></p><div class="meta"><span class="tag"></span><span></span><span></span><button type="button" class="secondary">Copy</button></div>`;
  card.querySelector("p").textContent = summary;
  if (label) {
    const title = document.createElement("h2");
    title.className = "source";
    title.textContent = label;
    card.prepend(title);
  }
  const words = summary.split(/\s+/).length;
  card.querySelector(".tag").textContent =
    `${VARIANTS[req.variant].label} model · ${req.length[0].toUpperCase()}${req.length.slice(1)} · ${SEARCH[req.beams]}`;
  const [a, b] = card.querySelectorAll(".meta > span:not(.tag)");
  a.textContent = `${words} words from ${inputTokens.toLocaleString()} tokens${chunks > 1 ? ` in ${chunks} parts` : ""}`;
  b.textContent = `${seconds < 60 ? seconds.toFixed(1) + " s" : Math.round(seconds / 60) + " min"}`;
  const copy = card.querySelector("button");
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(summary); copy.textContent = "Copied"; }
    catch { copy.textContent = "Copy failed"; }
    setTimeout(() => copy.textContent = "Copy", 1500);
  };
  $("results").prepend(card);
}
