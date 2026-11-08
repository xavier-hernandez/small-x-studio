# syntax=docker/dockerfile:1

# ---- build ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY SmallSdWeb/SmallSdWeb.csproj SmallSdWeb/
RUN dotnet restore SmallSdWeb/SmallSdWeb.csproj

COPY SmallSdWeb/ SmallSdWeb/
RUN dotnet publish SmallSdWeb/SmallSdWeb.csproj -c Release -o /app/publish --no-restore /p:UseAppHost=false

# ---- runtime ----
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app

COPY --from=build /app/publish .

# Models (~1.6 GB) are downloaded here on first start. Mount a volume to keep them between runs.
# The directory is created owned by the non-root app user so a fresh named volume inherits that ownership.
RUN mkdir -p /app/models && chown "$APP_UID" /app/models
VOLUME ["/app/models"]

USER $APP_UID
ENV ASPNETCORE_HTTP_PORTS=8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "SmallSdWeb.dll"]
