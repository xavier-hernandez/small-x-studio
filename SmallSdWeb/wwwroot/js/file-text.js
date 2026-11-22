// Reads the text out of a file the user picked: plain text, Word (.docx) or PDF.
// Everything happens in this tab: the file is read with the File API and never sent anywhere.
// PDFs use pdf.js (served by this site from /lib/pdfjs/, loaded only when a PDF is picked);
// .docx files are zip archives of XML, read here with the browser's own DecompressionStream.

export const ACCEPT = ".pdf,.docx,.txt,.md,.markdown,.text,.csv,application/pdf,text/plain,text/markdown,"
  + "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
export const MAX_BYTES = 100 * 1024 * 1024;

/** A problem with the file itself, shown to the user as is. */
export class FileTextError extends Error { }

/**
 * Returns { text, kind, pages? }. `kind` is "PDF", "Word" or "Text".
 * `onProgress(done, total)` reports PDF pages as they're read.
 */
export async function extractText(file, onProgress) {
  if (file.size > MAX_BYTES) throw new FileTextError(`This file is ${formatSize(file.size)}; the limit is ${formatSize(MAX_BYTES)}.`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const kind = detectKind(file.name, bytes);
  let result;
  if (kind === "PDF") result = await pdfText(bytes, onProgress);
  else if (kind === "Word") result = { text: await docxText(bytes) };
  else if (kind === "Text") result = { text: decodeText(bytes) };
  else if (kind === "not-docx") throw new FileTextError("This file is named .docx but isn't a Word document.");
  else if (kind === "not-pdf") throw new FileTextError("This file is named .pdf but isn't a PDF.");
  else if (kind === "doc") throw new FileTextError("This is an older Word file (.doc). Open it in Word and save it as .docx, then pick that file.");
  else throw new FileTextError("This file type isn't supported. Pick a PDF, a Word document (.docx) or a text file.");
  result.text = tidyText(result.text);
  if (!result.text) {
    throw new FileTextError(kind === "PDF"
      ? "No text found in this PDF. It's probably scanned pages (images of text), and reading text from images isn't supported."
      : "No text found in this file.");
  }
  return { kind, ...result };
}

/** Decides by content first (magic bytes), then by extension. */
function detectKind(name, b) {
  const ext = (name.match(/\.([^.]+)$/)?.[1] ?? "").toLowerCase();
  if (b[0] === 0x25 && b[1] === 0x50 && b[2] === 0x44 && b[3] === 0x46) return "PDF";                 // %PDF
  if (b[0] === 0x50 && b[1] === 0x4b && b[2] === 3 && b[3] === 4) return ext === "docx" || ext === "docm" || !ext ? "Word" : "zip";
  if (b[0] === 0xd0 && b[1] === 0xcf && b[2] === 0x11 && b[3] === 0xe0) return "doc";                 // OLE2: .doc, .xls…
  if (ext === "docx" || ext === "pdf") return "not-" + ext;
  if (["txt", "text", "md", "markdown", "csv", "log", ""].includes(ext) || looksLikeText(b)) return "Text";
  return "unknown";
}

function looksLikeText(b) {
  const n = Math.min(b.length, 4096);
  if (b[0] === 0xff && b[1] === 0xfe || b[0] === 0xfe && b[1] === 0xff) return true;                // UTF-16 BOM
  for (let i = 0; i < n; i++) if (b[i] === 0) return false;
  return true;
}

/** UTF-8 (with or without BOM) or UTF-16 with a BOM; anything that isn't valid UTF-8 is read as Windows-1252. */
function decodeText(b) {
  if (b[0] === 0xff && b[1] === 0xfe) return new TextDecoder("utf-16le").decode(b);
  if (b[0] === 0xfe && b[1] === 0xff) return new TextDecoder("utf-16be").decode(b);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(b); }
  catch { return new TextDecoder("windows-1252").decode(b); }
}

/** Normalizes line endings and spacing, and keeps at most one blank line between paragraphs. */
function tidyText(text) {
  return text.replace(/\r\n?/g, "\n").replace(/[ \t\f\v]+/g, " ").replace(/[ ]*\n[ ]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n").replace(/[ ]{2,}/g, " ").trim();
}

// ---------- PDF ----------

let pdfjs = null;
async function pdfText(bytes, onProgress) {
  if (!pdfjs) {
    pdfjs = await import("/lib/pdfjs/pdf.min.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = "/lib/pdfjs/pdf.worker.min.mjs";
  }
  // isEvalSupported: false keeps pdf.js from compiling code from the file; fonts and images aren't needed for text.
  const task = pdfjs.getDocument({ data: bytes, isEvalSupported: false, disableFontFace: true, useSystemFonts: false, enableXfa: false });
  let doc;
  try {
    doc = await task.promise;
  } catch (e) {
    task.destroy();
    if (e?.name === "PasswordException") throw new FileTextError("This PDF is password-protected. Remove the password and try again.");
    if (e?.name === "InvalidPDFException") throw new FileTextError("This PDF is damaged or isn't really a PDF.");
    throw e;
  }
  try {
    const pages = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      pages.push(pageText(await page.getTextContent()));
      page.cleanup();
      onProgress?.(i, doc.numPages);
    }
    return { text: pages.filter(Boolean).join("\n\n"), pages: doc.numPages };
  } finally {
    task.destroy();
  }
}

/**
 * Rebuilds a page's text from pdf.js text items. Lines in the same paragraph are joined with a space (or with
 * nothing, for a word hyphenated across lines); a gap clearly taller than the usual line spacing starts a new
 * paragraph. Lines that are only a page number are dropped.
 */
function pageText({ items }) {
  const lines = [];
  let line = { text: "", y: null, h: 0 };
  for (const it of items) {
    if (!("str" in it)) continue;
    const y = it.transform?.[5];
    if (line.y === null && it.str.trim()) { line.y = y; line.h = it.height || Math.hypot(it.transform[2], it.transform[3]); }
    line.text += it.str;
    if (it.hasEOL) { lines.push(line); line = { text: "", y: null, h: 0 }; }
  }
  lines.push(line);

  const kept = lines.filter(l => l.text.trim() && !/^\s*(page\s+)?\d{1,4}(\s*(of|\/)\s*\d{1,4})?\s*$/i.test(l.text));
  const gaps = [];
  for (let i = 1; i < kept.length; i++) {
    const gap = kept[i - 1].y - kept[i].y;
    if (gap > 0) gaps.push(gap);
  }
  gaps.sort((a, b) => a - b);
  const usual = gaps.length ? gaps[Math.floor((gaps.length - 1) / 2)] : 0;

  let out = "";
  kept.forEach((l, i) => {
    const text = l.text.replace(/\s+/g, " ").trim();
    if (i === 0) { out = text; return; }
    const gap = kept[i - 1].y - l.y;
    const newParagraph = usual > 0 && (gap > usual * 1.4 || gap < 0);
    if (newParagraph) out += "\n\n" + text;
    else if (/\p{L}-$/u.test(out) && /^\p{Ll}/u.test(text)) out = out.slice(0, -1) + text;
    else out += " " + text;
  });
  return out;
}

// ---------- Word (.docx) ----------

const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";

async function docxText(bytes) {
  const zip = readZip(bytes);
  const main = zip.get("word/document.xml");
  if (!main) throw new FileTextError("This file isn't a Word document (.docx).");
  const xml = new DOMParser().parseFromString(await inflate(bytes, main), "application/xml");
  if (xml.getElementsByTagName("parsererror").length) throw new FileTextError("This Word document is damaged.");
  return wordText(xml.documentElement);
}

/**
 * Walks the document in order: each paragraph (w:p) ends with a line break, w:t is text, w:tab and w:br are
 * spacing. Deleted tracked changes (w:delText) and field codes (w:instrText) are skipped by not matching them.
 */
function wordText(root) {
  const parts = [];
  const walk = node => {
    for (let n = node.firstChild; n; n = n.nextSibling) {
      if (n.nodeType !== 1) continue;
      if (n.namespaceURI === W) {
        switch (n.localName) {
          case "t": parts.push(n.textContent); continue;
          case "tab": parts.push(" "); continue;
          case "br": case "cr": parts.push("\n"); continue;
          case "noBreakHyphen": parts.push("-"); continue;
          case "p": walk(n); parts.push("\n\n"); continue;
          case "tc": walk(n); parts.push(" "); continue;     // table cell
          case "del": case "instrText": case "rPr": case "pPr": continue;
        }
      }
      walk(n);
    }
  };
  walk(root);
  return parts.join("");
}

/** Lists a zip's files from its central directory: name → { offset, method, size }. */
function readZip(b) {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let eocd = -1;
  for (let i = b.length - 22; i >= Math.max(0, b.length - 22 - 65535); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new FileTextError("This Word document is damaged (it isn't a complete zip archive).");
  const count = view.getUint16(eocd + 10, true);
  let p = view.getUint32(eocd + 16, true);
  const files = new Map();
  for (let i = 0; i < count && p + 46 <= b.length; i++) {
    if (view.getUint32(p, true) !== 0x02014b50) break;
    const method = view.getUint16(p + 10, true);
    const size = view.getUint32(p + 20, true);
    const nameLen = view.getUint16(p + 28, true), extraLen = view.getUint16(p + 30, true), commentLen = view.getUint16(p + 32, true);
    const offset = view.getUint32(p + 42, true);
    const name = new TextDecoder().decode(b.subarray(p + 46, p + 46 + nameLen));
    files.set(name, { offset, method, size });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return files;
}

/** Decompresses one zip entry to text. Word uses "deflate" (method 8) or no compression (method 0). */
async function inflate(b, { offset, method, size }) {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (view.getUint32(offset, true) !== 0x04034b50) throw new FileTextError("This Word document is damaged.");
  const start = offset + 30 + view.getUint16(offset + 26, true) + view.getUint16(offset + 28, true);
  const data = b.subarray(start, start + size);
  if (method === 0) return new TextDecoder().decode(data);
  if (method !== 8) throw new FileTextError("This Word document uses a compression method that isn't supported.");
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return await new Response(stream).text();
}

export function formatSize(n) {
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}
