# syntax=docker/dockerfile:1
# Serving image: a verified model bundle in, predictions out.
# No scikit-learn, no pickle-based model library, no compiler, no root.
#
# Run (see `make docker-run`):
#   docker run --read-only --cap-drop ALL --security-opt no-new-privileges \
#     -v ./build/bundle:/models:ro -v ./runtime/keys/signing.pub:/keys/signing.pub:ro \
#     -p 8080:8080 mlseal:local

# python:3.12-slim, pinned by digest (multi-arch index); bump it deliberately.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
COPY requirements/serve.txt /tmp/serve.txt
# Every package is pinned and hash-checked; wheels only, so no build script ever runs.
# The virtualenv gets no pip of its own (and the runtime stage drops the base image's).
RUN python -m venv --without-pip /opt/venv \
    && python -m pip --python /opt/venv/bin/python install \
        --require-hashes --only-binary=:all: --no-deps -r /tmp/serve.txt
COPY src/mlseal /opt/app/mlseal
RUN /opt/venv/bin/python -m compileall -q /opt/app

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/app \
    MLSEAL_BUNDLE=/models \
    MLSEAL_PUBLIC_KEY=/keys/signing.pub
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/app /opt/app
# The base image's pip goes too: no package installer is left in the image (and no pip
# release for image scanners to flag).
RUN /usr/local/bin/python -m pip uninstall --yes --no-input --disable-pip-version-check \
        --root-user-action=ignore pip \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin mlseal
USER 10001:10001
EXPOSE 8080
# Liveness only; readiness (/readyz) is true once the bundle is verified and loaded.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
ENTRYPOINT ["python", "-m", "mlseal", "serve", "--host", "0.0.0.0", "--port", "8080"]
