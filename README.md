# Small X Studio

> [!NOTE]
THIS IS A LITTLE VIBE CODING PROJECT OF MINE. SEEING WHAT I CAN DO TOTALLY IN THE BROWSER. I WAS INSPIRED BY https://webllm.mlc.ai/.



Small AI models that run entirely in the browser: text-to-image generation (512px images and 16×16 pixel-art
sprites) and text summarization. Every image and summary is computed on the user's device with
[ONNX Runtime Web](https://onnxruntime.ai/), on the GPU (WebGPU or WebNN) or the CPU (multi-threaded WebAssembly).
Prompts, text and results never leave the browser.

The .NET app is only a host: it downloads the models from Hugging Face once, then serves the pages, the runtime
files and the model files. Source: [github.com/xavier-hernandez/small-x-studio](https://github.com/xavier-hernandez/small-x-studio).

<img width="557" height="869" alt="image" src="https://github.com/user-attachments/assets/3212f437-5592-4503-abc8-a3b40a11cd49" />

## Pages

| Page | URL | What it does |
|---|---|---|
| Home | `/` | Introduction, and whether each tool's models are ready on the server |
| Generate images | `/generate` | Small SD (512px images) or Mini Pixel Art (16×16 sprites), with a device check and processor choice |
| Summarize text | `/summarize` | DistilBART summaries, Fast (290 MB) or Best (1.1 GB) model, with length and search options |
| Summarize a file | `/summarize-file` | The same summarizer for a PDF, Word (`.docx`) or text file, read in the browser and never uploaded |
| About | `/about` | How it works for users, the models and licenses, what's stored in the browser, and a Storage section to delete stored models |

## Run it

### With Docker

```sh
docker compose up --build
```

Then open http://localhost:8080. The models are downloaded into the `models` volume on first start, so later
starts skip the download.

### With the .NET SDK

```sh
cd SmallSdWeb
dotnet run --launch-profile http
```

Then open http://localhost:5245. The models are downloaded into `SmallSdWeb/models/`, which is git-ignored.

Either way, use `localhost` or HTTPS in the address: browsers only enable WebGPU in a secure context.

### First start

The host downloads the models from Hugging Face in the background, mostly smallest first, and each tool unlocks
as soon as its model is ready:

| Order | Model | Size |
|---|---|---|
| 1 | Mini Pixel Art | 34 MB |
| 2 | Summarizer, Fast | 290 MB |
| 3 | Small SD | 1.6 GB |
| 4 | Summarizer, Best | 1.1 GB |

About 3 GB in all. Files that already exist are skipped. Each file downloads to a `.part` file first and is renamed
when complete; an interrupted download starts that file again on the next start. If a model fails to download, its
tool shows the error and the other models carry on.

When a user first opens a tool, the browser downloads that tool's model from the host and stores it with the Cache
API, so later visits load it from disk.

## TODO

- try to load a smaller model
- ability to create larger images without increasing time to generate
- making sure this works under Apple M-series chips

## Models

### Small SD (512px images)

[danyalmalik/small-sd-ONNX-WebGPU](https://huggingface.co/danyalmalik/small-sd-ONNX-WebGPU), the ONNX export of
[segmind/small-sd](https://huggingface.co/segmind/small-sd). CreativeML OpenRAIL-M.

| Item | File (in `models/`) | Size |
|---|---|---|
| Text encoder (fp16 weights) | `text_encoder/model.onnx` | 246 MB |
| UNet (fp16 weights) | `unet/model.onnx` | 1.16 GB |
| VAE decoder (fp32) | `vae_decoder/model.onnx` | 198 MB |
| Tokenizer | `tokenizer/vocab.json`, `tokenizer/merges.txt` | 1.6 MB |

Runs on WebGPU (needs `shader-f16`), WebNN or the CPU. The CPU takes minutes per image and is limited to 512×512,
since larger images don't fit in WebAssembly's 4 GB memory. Switching between a GPU backend and the CPU needs a page
reload, because ONNX Runtime fixes once per page whether CPU work runs in its proxy worker.

### Mini Pixel Art (16×16 sprites)

[AnticPan/mini-pixel-art](https://huggingface.co/AnticPan/mini-pixel-art): tiny pixel-space DiTs that make 16×16
sprites with transparency. There's no VAE or CLIP: captions are looked up word by word in a fixed vocabulary, so
short noun phrases like the training captions work best ("red potion bottle", "side-view knight with sword").
Unknown words are ignored and listed after each run. Sprites can be downloaded at 16×16 or 128×128.

| Item | File (in `models/mini-pixel-art/`) | Size |
|---|---|---|
| Model: `l-p1`, the large size (8.8M parameters, best quality) | `l-p1/model.onnx` | 33.7 MB |
| Sampler settings | `l-p1/config.json` | |
| Vocabulary and 4,096-colour palette | `vocab.json`, `palette.bin` | 0.2 MB |

The repo also has smaller, faster sizes (`s-p1`, `m-p1`, `m-p2`); the page always uses `l-p1`. It runs on WebGPU
when that's the selected processor, otherwise on the CPU, which is fast enough for a model this small.

License: MIT for the code and weights. They were trained on a CC0 dataset that was itself generated with Krea 2,
so the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing) may also apply; see the repo's `NOTICE`.

### Summarizer (DistilBART-CNN)

[Xenova/distilbart-cnn-6-6](https://huggingface.co/Xenova/distilbart-cnn-6-6), the ONNX export of
[sshleifer/distilbart-cnn-6-6](https://huggingface.co/sshleifer/distilbart-cnn-6-6), a BART model distilled for
summarizing news (CNN/DailyMail). Apache 2.0. The page offers two versions, and remembers the choice per browser:

| Version | Files (in `models/distilbart-cnn-6-6/`) | Size |
|---|---|---|
| **Fast** (default): int8 weights | `onnx/encoder_model_quantized.onnx`, `onnx/decoder_model_merged_quantized.onnx` | 271 MB |
| **Best**: full precision (fp32) | `onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx` | 1.07 GB |
| Tokenizer, shared | `vocab.json`, `merges.txt` | 1.2 MB |

Best writes noticeably better summaries. Once loaded, both summarize at about the same speed on the CPU; Best takes
about twice as long to load and uses more memory. The worker keeps one version loaded at a time.

Summarization runs on the CPU only, in a Web Worker: the int8 weights use `MatMulInteger`, which ONNX Runtime runs
well only on its WebAssembly backend. Generation follows the model's own settings: beam search with 4 beams (the
"Thorough" search setting) or greedy ("Quick"), no repeated 3-grams, a minimum and maximum length, and a length
penalty of 2.0. Text longer than the model's 1,024-token window is split at sentence ends, and each part gets its
own summary paragraph. With these settings the model often stops mid-sentence, so a trailing fragment is dropped
when a whole sentence comes before it.

#### Files

The file page reads the file with the browser's File API; it is never uploaded, not even to the host. The text is
shown in an editable box before summarizing.

| Type | How the text is read |
|---|---|
| PDF | [pdf.js](https://mozilla.github.io/pdf.js/) (`getTextContent` per page). Lines are joined into paragraphs by their spacing, words hyphenated across lines are rejoined, and lines that are only a page number are dropped. Scanned PDFs have no text layer and show an error; password-protected PDFs too |
| Word (`.docx`) | Unzipped with the browser's `DecompressionStream` and read from `word/document.xml`: paragraphs, tables, tabs and breaks; deleted tracked changes and field codes are skipped. Older `.doc` files aren't supported |
| Text (`.txt`, `.md`, `.csv`) | UTF-8, or UTF-16 with a BOM; anything else is read as Windows-1252 |

Files up to 100 MB are accepted. The type is detected from the file's first bytes, then its extension.

## Configuration

`SmallSdWeb/appsettings.json` sets where each model comes from and where it's stored:

| Section | Settings |
|---|---|
| `StableDiffusion` | `ModelDirectory`: the folder for all models (default `models`, relative to the app). `Repository`: the Small SD repo |
| `PixelArt` | `Repository`, `Folder` (a subfolder of the model directory) and `Variants`: which sizes to download (default `l-p1`, the one the page uses) |
| `Summarizer` | `Repository` and `Folder`. Both versions download from this repo |

Files download from the repo's `main` branch. A different repository must have the same file layout.

`GET /api/status` reports each model's download state, which the pages poll:

```json
{ "models": { "mini-pixel-art": { "phase": "ready", "detail": null },
              "summarizer": { "phase": "downloading", "detail": "onnx/encoder_model_quantized.onnx: 40% of 123 MB" },
              "small-sd": { "phase": "waiting", "detail": null },
              "summarizer-full": { "phase": "waiting", "detail": null } } }
```

`phase` is `waiting`, `downloading`, `ready` or `error`.

## Dependencies

### .NET SDK (`net10.0`)

The project has **no NuGet package references**. Everything it uses is part of the .NET shared framework.

### JavaScript (browser)

| Dependency | Version | License | Location |
|---|---|---|---|
| [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) | 1.29.0 | MIT | `wwwroot/lib/onnxruntime-web/` (self-hosted, no CDN) |
| [pdfjs-dist](https://www.npmjs.com/package/pdfjs-dist) | 6.5.438 | Apache 2.0 | `wwwroot/lib/pdfjs/` (self-hosted, no CDN) |

Files used from onnxruntime-web:

- `ort.webgpu.min.mjs` (66 KB): the runtime ES module.
- `ort-wasm-simd-threaded.asyncify.mjs` (51 KB) and `.wasm` (25 MB): the WebAssembly runtime, used by every backend.

Files used from pdfjs-dist: `build/pdf.min.mjs` (474 KB) and `build/pdf.worker.min.mjs` (1.3 MB), plus its
`LICENSE`. The file page loads them only when a PDF is picked.

There are no other JavaScript libraries, bundlers or npm install steps.

## Project layout

```
compose.yaml, Dockerfile          container build and run
SmallSdWeb/
  Program.cs                      host: COOP/COEP headers, page URLs, static files, /models, /api/status
  ModelDownloader.cs              downloads the models from Hugging Face on start
  appsettings.json                model repositories and folders
  wwwroot/
    index.html                    home page
    generate.html                 image generation
    summarize.html                summarize pasted text
    summarize-file.html           summarize a PDF, Word or text file
    about.html                    about the project, and the Storage section
    css/site.css                  shared styles: colours, navigation, form controls
    js/browser-sd.js              Small SD: CLIP tokenizer, DPM-Solver++ scheduler, diffusion pipeline; shared ORT setup and model cache
    js/device-check.js            device detection, requirement checks and CPU time estimates
    js/pixel-art.js               Mini Pixel Art: word vocabulary, flow-matching sampler, palette snapping
    js/summarizer.js              summarizer: GPT-2 byte-level BPE tokenizer, encoder, beam search with a KV cache
    js/summarize-worker.js        runs the summarizer in a Web Worker
    js/summarize-ui.js            summarizer controls shared by both summarize pages
    js/file-text.js               reads text from PDF (pdf.js), .docx (zip + XML) and text files
    js/storage.js                 lists and deletes the models stored in the browser
    js/help.js                    "?" help bubbles and the model status line
    lib/onnxruntime-web/          ONNX Runtime Web
    lib/pdfjs/                    pdf.js
```

The host serves `/generate`, `/summarize`, `/summarize-file` and `/about` without the `.html`.

## Browser features

| Feature | Needed for |
|---|---|
| WebGPU with `shader-f16` | GPU backend for Small SD (fastest); Mini Pixel Art also uses WebGPU when it's selected |
| WebNN (`navigator.ml`, GPU device) | DirectML backend on Windows. Experimental: in Chrome, enable `--enable-features=WebMachineLearningNeuralNetwork` |
| WebAssembly SIMD + threads (`SharedArrayBuffer`) | CPU backend, and all summarization. Threads need cross-origin isolation, which the host turns on with COOP/COEP headers |
| Web Workers | Run CPU work off the page's thread: ONNX Runtime's proxy worker for Small SD, and the summarizer's own worker |
| Cache API, Storage API | Store the model files, check free disk space, and list and delete stored models |
| Canvas | Turn the output into PNG |
| File API, `DecompressionStream`, `DOMParser` | Read the picked file locally: unzip `.docx` files and parse their XML |

The image page detects which of these are available and shows whether the device meets the requirements.
