# Hearth hub. Built and run on the target host (x86-64 without AVX is fine:
# every native dependency here was checked on a Celeron G1610T, see README).
FROM python:3.12-slim-bookworm AS models
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*
COPY tools/fetch_models.sh /fetch_models.sh
RUN sh /fetch_models.sh /models

FROM python:3.12-slim-bookworm
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends alsa-utils opus-tools curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY hearth ./hearth
RUN pip install --no-cache-dir --root-user-action=ignore ".[models]"
COPY --from=models /models /models
ENV HEARTH_CONFIG=/config/hearth.toml PYTHONUNBUFFERED=1
# uid 1000 with the host's audio group is set by compose (group ids differ per host)
USER 1000:1000
HEALTHCHECK --interval=1m --timeout=5s --start-period=40s \
    CMD curl -fs "http://${HEARTH_HEALTH_HOST:-127.0.0.1}:${HEARTH_HEALTH_PORT:-8377}/healthz" || exit 1
ENTRYPOINT ["python", "-m", "hearth.app"]
