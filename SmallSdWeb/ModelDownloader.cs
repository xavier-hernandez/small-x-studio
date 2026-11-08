namespace SmallSdWeb;

public sealed class ModelOptions
{
    public string ModelDirectory { get; set; } = "models";
    public string Repository { get; set; } = "danyalmalik/small-sd-ONNX-WebGPU";
}

/// <summary>
/// Downloads any missing model files from Hugging Face once, so the app can serve them to browsers
/// at /models. All image generation happens in the browser.
/// </summary>
public sealed class ModelDownloader(IConfiguration config, IWebHostEnvironment env, IHttpClientFactory http, ILogger<ModelDownloader> logger)
    : BackgroundService
{
    private static readonly string[] Files =
    [
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "text_encoder/model.onnx",
        "vae_decoder/model.onnx",
        "unet/model.onnx",
    ];

    public string Phase { get; private set; } = "starting";
    public string? Detail { get; private set; }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        var options = config.GetSection("StableDiffusion").Get<ModelOptions>() ?? new ModelOptions();
        var dir = Path.GetFullPath(options.ModelDirectory, env.ContentRootPath);

        try
        {
            Phase = "downloading";
            var client = http.CreateClient();
            foreach (var file in Files)
            {
                var target = Path.Combine(dir, file);
                if (File.Exists(target)) continue;
                await DownloadAsync(client, $"https://huggingface.co/{options.Repository}/resolve/main/{file}", target, file, ct);
            }
            Phase = "ready";
            Detail = null;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
        catch (Exception ex)
        {
            logger.LogError(ex, "Model download failed");
            Phase = "error";
            Detail = ex.Message;
        }
    }

    private async Task DownloadAsync(HttpClient client, string url, string target, string label, CancellationToken ct)
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
                Detail = total is > 0 ? $"{label}: {done * 100 / total}% of {total / 1_048_576} MB" : $"{label}: {done / 1_048_576} MB";
            }
        }
        File.Move(partial, target, overwrite: true);
    }
}
