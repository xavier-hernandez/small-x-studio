using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.FileProviders;
using SmallSdWeb;

// Static host for in-browser image generation: serves the page, ONNX Runtime Web and the model files.
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddHttpClient();
builder.Services.AddSingleton<ModelDownloader>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<ModelDownloader>());

var app = builder.Build();

// Cross-origin isolation lets in-browser CPU (WebAssembly) generation use multiple threads.
app.Use(async (context, next) =>
{
    context.Response.Headers["Cross-Origin-Opener-Policy"] = "same-origin";
    context.Response.Headers["Cross-Origin-Embedder-Policy"] = "require-corp";
    await next();
});

app.UseDefaultFiles();
// Revalidate app files on every load (a 304 when unchanged), so a browser never mixes an old page with new scripts.
app.UseStaticFiles(new StaticFileOptions
{
    OnPrepareResponse = ctx => ctx.Context.Response.Headers.CacheControl = "no-cache",
});

// Model files. Only the known model file types are served, so partial downloads (*.part) stay hidden.
var modelOptions = app.Configuration.GetSection("StableDiffusion").Get<ModelOptions>() ?? new ModelOptions();
var modelDir = Path.GetFullPath(modelOptions.ModelDirectory, app.Environment.ContentRootPath);
Directory.CreateDirectory(modelDir);
var modelTypes = new FileExtensionContentTypeProvider(new Dictionary<string, string>
{
    [".onnx"] = "application/octet-stream",
    [".json"] = "application/json",
    [".txt"] = "text/plain",
});
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(modelDir),
    RequestPath = "/models",
    ContentTypeProvider = modelTypes,
});

// Whether the model files are available yet (they're downloaded on first start).
app.MapGet("/api/status", (ModelDownloader models) => new { phase = models.Phase, detail = models.Detail });

app.Run();
