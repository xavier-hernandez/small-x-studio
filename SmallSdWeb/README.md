# Small X Studio

Text-to-image generation that runs entirely in the browser. 

Uses [danyalmalik/small-sd-ONNX-WebGPU](https://huggingface.co/danyalmalik/small-sd-ONNX-WebGPU) exported to ONNX, based off of [segmind/small-sd](https://huggingface.co/segmind/small-sd).
The .NET app is only a static host. It serves the page, the runtime files and the model files. Everything happens locally in your browser.

On first start the host downloads the models (about 1.6 GB) into `models/` from Hugging Face. The browser then caches
them with the Cache API.

**TODO**
-----
- try to load a smaller model
- ability to create larger images without increasing time to generate
- making sure this works under Apple M-series chips


## Dependencies

### .NET (server)

| Dependency | Version
|---|---|---|
| .NET SDK | 10.0 (`net10.0`)
| ASP.NET Core (`Microsoft.NET.Sdk.Web`) | ships with .NET 10
| `Microsoft.Extensions.Http` (`IHttpClientFactory`) | ships with .NET 10 | Downloading the models from Hugging Face |

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

The page itself is `wwwroot/index.html`.

### Model

| Item | Source | Size |
|---|---|---|
| Text encoder (fp16 weights) | [danyalmalik/small-sd-ONNX-WebGPU](https://huggingface.co/danyalmalik/small-sd-ONNX-WebGPU) `text_encoder/model.onnx` | 246 MB |
| UNet (fp16 weights) | same repo, `unet/model.onnx` | 1.16 GB |
| VAE decoder (fp32) | same repo, `vae_decoder/model.onnx` | 198 MB |
| Tokenizer | same repo, `tokenizer/vocab.json`, `tokenizer/merges.txt` | 1.6 MB |

To change the repository or the download folder, edit the `StableDiffusion` section in
`appsettings.json`. For the model's license, see the model cards on Hugging Face.

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
