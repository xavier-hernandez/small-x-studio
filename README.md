# Small X Studio

Text-to-image generation and text summarization that run entirely in the browser.

The site has four pages: **Home** (`/`), **Generate images** (`/generate`), **Summarize text** (`/summarize`) and
**About** (`/about`).

Uses [danyalmalik/small-sd-ONNX-WebGPU](https://huggingface.co/danyalmalik/small-sd-ONNX-WebGPU) exported to ONNX, based off of [segmind/small-sd](https://huggingface.co/segmind/small-sd).
The .NET app is only a static host. It serves the page, the runtime files and the model files. Everything happens locally in your browser.

On first start the host downloads the models into `models/` from Hugging Face, mostly smallest first: Mini Pixel
Art (34 MB), the Fast summarizer (290 MB), Small SD (1.6 GB), then the Best summarizer (1.1 GB). About 3 GB in all. The browser then caches them with the Cache API.


<img width="557" height="869" alt="image" src="https://github.com/user-attachments/assets/3212f437-5592-4503-abc8-a3b40a11cd49" />

## Run with Docker

```sh
docker compose up --build
```

Then open http://localhost:8080. The models are downloaded into the `models` volume on first start, so later
starts skip the download. Use `localhost` (or HTTPS) in the address: browsers only enable WebGPU in a secure context.
**TODO**
-----
- try to load a smaller model
- ability to create larger images without increasing time to generate
- making sure this works under Apple M-series chips


## Dependencies

### .NET SDK (`net10.0`)

The project has **no NuGet package references**. Everything above is part of the .NET shared framework.

### JavaScript (browser)

| Dependency | Version | License | Location |
|---|---|---|---|
| [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) | 1.29.0 | MIT | `wwwroot/lib/onnxruntime-web/` (self-hosted, no CDN) |

Files used from onnxruntime-web:

- `ort.webgpu.min.mjs` (66 KB): the runtime ES module.
- `ort-wasm-simd-threaded.asyncify.mjs` (51 KB) and `.wasm` (25 MB): the WebAssembly runtime, used by every backend.

There are no other JavaScript libraries, bundlers or npm install steps. The app code is plain ES
modules:

- `wwwroot/js/browser-sd.js`: the CLIP tokenizer, DPM-Solver++ scheduler and diffusion pipeline.
- `wwwroot/js/device-check.js`: device detection, requirement checks and CPU time estimates.
- `wwwroot/js/pixel-art.js`: the Mini Pixel Art pipeline (word vocabulary, flow-matching sampler, palette snapping).
- `wwwroot/js/summarizer.js`: the summarization pipeline (GPT-2 byte-level BPE tokenizer, encoder, beam search with a KV cache).
- `wwwroot/js/summarize-worker.js`: runs the summarizer in a Web Worker so the page stays responsive.
- `wwwroot/js/help.js`: the "?" help bubbles and the model status line, shared by the pages.
- `wwwroot/js/storage.js`: lists the models stored in the browser's Cache API and deletes them (the About page's
  Storage section, with a Delete button per model and "Clear everything", which also resets the saved settings).

The pages are `wwwroot/index.html` (home), `generate.html`, `summarize.html` and `about.html`, with shared styles in
`wwwroot/css/site.css`. The host serves `/generate`, `/summarize` and `/about` without the `.html`.

### Models

Pick the model at the top of the form. The host downloads Mini Pixel Art first because it's small, so it's
usable within seconds of the first start, while Small SD is still downloading.

#### Small SD (512px images)

| Item | Source | Size |
|---|---|---|
| Text encoder (fp16 weights) | [danyalmalik/small-sd-ONNX-WebGPU](https://huggingface.co/danyalmalik/small-sd-ONNX-WebGPU) `text_encoder/model.onnx` | 246 MB |
| UNet (fp16 weights) | same repo, `unet/model.onnx` | 1.16 GB |
| VAE decoder (fp32) | same repo, `vae_decoder/model.onnx` | 198 MB |
| Tokenizer | same repo, `tokenizer/vocab.json`, `tokenizer/merges.txt` | 1.6 MB |

To change the repository or the download folder, edit the `StableDiffusion` section in
`appsettings.json`. For the model's license, see the model cards on Hugging Face.

#### Mini Pixel Art (16×16 sprites)

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
when that's the selected processor, otherwise on the CPU, which is fast enough for a model this small. The
`PixelArt` section of `appsettings.json` sets the repository and folder.

License: MIT for the code and weights. They were trained on a CC0 dataset that was itself generated with Krea 2,
so the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing) may also apply; see the repo's `NOTICE`.

#### Summarizer (DistilBART-CNN)

[Xenova/distilbart-cnn-6-6](https://huggingface.co/Xenova/distilbart-cnn-6-6), the ONNX export of
[sshleifer/distilbart-cnn-6-6](https://huggingface.co/sshleifer/distilbart-cnn-6-6), a BART model distilled for
summarizing news (CNN/DailyMail). Apache 2.0.

| Item | File (in `models/distilbart-cnn-6-6/`) | Size |
|---|---|---|
The page offers two versions of it (the choice is remembered per browser):

| Version | Files (in `models/distilbart-cnn-6-6/`) | Size |
|---|---|---|
| **Fast** (default): int8 weights | `onnx/encoder_model_quantized.onnx`, `onnx/decoder_model_merged_quantized.onnx` | 271 MB |
| **Best**: full precision (fp32) | `onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx` | 1.07 GB |
| Tokenizer, shared | `vocab.json`, `merges.txt` | 1.2 MB |

Best writes noticeably better summaries. Once loaded, both summarize at about the same speed on the CPU; Best takes
about twice as long to load and uses more memory. The worker keeps one version loaded at a time.

Summarization runs on the CPU only: the int8 weights use `MatMulInteger`, which ONNX Runtime runs well only on its
WebAssembly backend. Generation follows the model's own settings: beam search with 4 beams (the "Thorough" search setting) or greedy
("Quick"), no repeated 3-grams, a minimum and maximum length, and a length penalty of 2.0. Text longer than
the model's 1,024-token window is split at sentence ends, and each part gets its own summary paragraph. With these
settings the model often stops mid-sentence, so a trailing fragment is dropped when a whole sentence comes before it.
The `Summarizer` section of `appsettings.json` sets the repository and folder.

### Browser features

| Feature | Needed for |
|---|---|
| WebGPU with `shader-f16` | GPU backend (fastest) |
| WebNN (`navigator.ml`, GPU device) | DirectML backend on Windows. Experimental: in Chrome, enable `--enable-features=WebMachineLearningNeuralNetwork` |
| WebAssembly SIMD + threads (`SharedArrayBuffer`) | CPU backend. Threads need cross-origin isolation, which the host turns on with COOP/COEP headers |
| Web Workers | Run CPU inference off the main thread (ONNX Runtime proxy worker) |
| Cache API, Storage API | Cache the model files and check free disk space |
| Canvas | Turn the output into PNG |

The page detects which of these are available and shows whether the device meets the requirements.
