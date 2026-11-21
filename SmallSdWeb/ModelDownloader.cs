namespace SmallSdWeb;

public sealed class ModelOptions
{
    public string ModelDirectory { get; set; } = "models";
    public string Repository { get; set; } = "danyalmalik/small-sd-ONNX-WebGPU";
}

public sealed class PixelArtOptions
{
    /// <summary>The best-quality size, which the page uses. The repo also has s-p1, m-p1 and m-p2.</summary>
    public static readonly string[] DefaultVariants = ["l-p1"];

    public string Repository { get; set; } = "AnticPan/mini-pixel-art";
    /// <summary>Subfolder of the model directory, so the files are served at /models/{Folder}/.</summary>
    public string Folder { get; set; } = "mini-pixel-art";
    /// <summary>Model sizes to download; empty means all of <see cref="DefaultVariants"/>.</summary>
    public string[] Variants { get; set; } = [];
}

public sealed class SummarizerOptions
{
    public string Repository { get; set; } = "Xenova/distilbart-cnn-6-6";
    /// <summary>Subfolder of the model directory, so the files are served at /models/{Folder}/.</summary>
    public string Folder { get; set; } = "distilbart-cnn-6-6";
}

/// <summary>
/// Downloads any missing model files from Hugging Face once, so the app can serve them to browsers
/// at /models. All image generation happens in the browser.
/// </summary>
public sealed class ModelDownloader(IConfiguration config, IWebHostEnvironment env, IHttpClientFactory http, ILogger<ModelDownloader> logger)
    : BackgroundService
{
    private sealed class ModelSet(string repository, string directory, string[] files)
    {
        public string Repository { get; } = repository;
        public string Directory { get; } = directory;
        public string[] Files { get; } = files;
        public string Phase { get; set; } = "waiting";
        public string? Detail { get; set; }
    }

    // Insertion order is download order, mostly smallest first: pixel art (34 MB) is usable within seconds, the fast
    // summarizer (about 290 MB) next, then Stable Diffusion (1.6 GB), and the full-precision summarizer (1.1 GB) last.
    private readonly Dictionary<string, ModelSet> _sets = BuildSets(config, env);

    /// <summary>Per model: phase ("waiting", "downloading", "ready" or "error") and a progress or error detail.</summary>
    public IReadOnlyDictionary<string, object> Status =>
        _sets.ToDictionary(s => s.Key, s => (object)new { phase = s.Value.Phase, detail = s.Value.Detail });

    private static Dictionary<string, ModelSet> BuildSets(IConfiguration config, IWebHostEnvironment env)
    {
        var sd = config.GetSection("StableDiffusion").Get<ModelOptions>() ?? new ModelOptions();
        var pixel = config.GetSection("PixelArt").Get<PixelArtOptions>() ?? new PixelArtOptions();
        var summarizer = config.GetSection("Summarizer").Get<SummarizerOptions>() ?? new SummarizerOptions();
        var root = Path.GetFullPath(sd.ModelDirectory, env.ContentRootPath);
        var variants = pixel.Variants is { Length: > 0 } v ? v : PixelArtOptions.DefaultVariants;

        return new()
        {
            ["mini-pixel-art"] = new(pixel.Repository, Path.Combine(root, pixel.Folder),
                ["vocab.json", "palette.bin", .. variants.SelectMany(name => new[] { $"{name}/config.json", $"{name}/model.onnx" })]),
            // "Fast": int8 weights, a quick download. They run well on the CPU, which is the only backend summarization uses.
            ["summarizer"] = new(summarizer.Repository, Path.Combine(root, summarizer.Folder),
                ["vocab.json", "merges.txt", "onnx/encoder_model_quantized.onnx", "onnx/decoder_model_merged_quantized.onnx"]),
            ["small-sd"] = new(sd.Repository, root,
                ["tokenizer/vocab.json", "tokenizer/merges.txt", "text_encoder/model.onnx", "vae_decoder/model.onnx", "unet/model.onnx"]),
            // "Best": the same model at full precision (fp32), better summaries for a 1.1 GB download. Shares the tokenizer files above.
            ["summarizer-full"] = new(summarizer.Repository, Path.Combine(root, summarizer.Folder),
                ["vocab.json", "merges.txt", "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
        };
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        var client = http.CreateClient();
        foreach (var set in _sets.Values)
        {
            try
            {
                set.Phase = "downloading";
                foreach (var file in set.Files)
                {
                    var target = Path.Combine(set.Directory, file);
                    if (File.Exists(target)) continue;
                    await DownloadAsync(client, $"https://huggingface.co/{set.Repository}/resolve/main/{file}", target, file, set, ct);
                }
                set.Phase = "ready";
                set.Detail = null;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { return; }
            catch (Exception ex)
            {
                // One model failing doesn't stop the others.
                logger.LogError(ex, "Model download failed for {Repository}", set.Repository);
                set.Phase = "error";
                set.Detail = ex.Message;
            }
        }
    }

    private async Task DownloadAsync(HttpClient client, string url, string target, string label, ModelSet set, CancellationToken ct)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        var partial = target + ".part";
        logger.LogInformation("Downloading {Url}", url);

        using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();
        var total = response.Content.Headers.ContentLength;

        await using (var source = await response.Content.ReadAsStreamAsync(ct))
        await using (var dest = File.Create(partial))
        {
            var buffer = new byte[1 << 20];
            long done = 0;
            int read;
            while ((read = await source.ReadAsync(buffer, ct)) > 0)
            {
                await dest.WriteAsync(buffer.AsMemory(0, read), ct);
                done += read;
                set.Detail = total is > 0 ? $"{label}: {done * 100 / total}% of {Math.Max(1, total.Value / 1_048_576)} MB" : $"{label}: {done / 1_048_576} MB";
            }
        }
        File.Move(partial, target, overwrite: true);
    }
}
