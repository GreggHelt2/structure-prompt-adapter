# syntax=docker/dockerfile:1.7
# SPA cloud image — ENV ONLY (W5.1). Serves Phase-1 ESM3 cache-gen + Phase-2 H100 training.
# Built locally on CPU with BuildKit SSH forwarding (the deps are private repos), pushed to
# Artifact Registry; the H100 only pulls the finished image. SPA code is pulled via git at
# runtime and data/cache/weights come from GCS/NGC/HF — none are baked here. (dev 07 W5.1, 08 step 2)
#
# Parity with local spa-dev: torch 2.5.1+cu124 (valid on Hopper/H100), Python 3.12, all deps
# pinned to the validated commits + constraints.txt (Q5.4 monkeypatch parity).
#
# Build: see docker/build_and_push.sh  (needs an ssh-agent with your GitHub key loaded).

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

# git + openssh-client: clone private deps at build, pull the SPA repo at runtime.
# libgomp1: OpenMP runtime for torch/numpy. (No build-essential — all deps ship manylinux wheels;
# add it back only if the CPU build surfaces an sdist that needs compiling.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch from the cu124 wheel index (bundles the CUDA 12.4 runtime; the H100 host provides the driver).
RUN pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Pin the ~160 transitive PyPI deps to the validated spa-dev versions.
COPY constraints.txt /tmp/constraints.txt

# Trust github.com for the SSH clones below.
RUN mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null

# esm is a Biohub fork pinned to a commit; it pulls its OWN transformers (Biohub/transformers @ main,
# a public repo — clones over https, no auth). We must NOT also pin transformers here: esm hard-pins
# transformers to its @main URL, and a second direct-URL spec for the same package makes pip's resolver
# fail (ResolutionImpossible). @main currently == ef32577, the commit the local spa-dev env was built from.
RUN --mount=type=ssh pip install -c /tmp/constraints.txt \
        "esm @ git+ssh://git@github.com/Biohub/esm.git@82ee35553d39169d678f784c8d3f8712ffd7d2c4"

# Core SPA deps at pinned commits, with the extras the project uses (foundry[rfd3] avoids cuequivariance).
RUN --mount=type=ssh pip install -c /tmp/constraints.txt \
        "atomworks[ml] @ git+ssh://git@github.com/RosettaCommons/atomworks.git@54f17711d70244a3be53ec9c8267edf8a5e8c0d6" \
        "rc-foundry[rfd3] @ git+ssh://git@github.com/RosettaCommons/foundry.git@e412591cd651093badd68ca78b93591f2d40f46f" \
        "clss-model @ git+ssh://git@github.com/guyyanai/CLSS.git@b0a42ff41028665c9418bad5afe2413b137b2ac1"

# Cloud tooling: experiment tracking (W5.4 = W&B) + GCS I/O. The SPA repo itself is cloned at runtime.
RUN pip install -c /tmp/constraints.txt wandb google-cloud-storage gcsfs

# Cloud orchestration CLIs (trailing layer, so the torch/deps layers above stay cached):
#   gcloud -> GCS rsync + Secret Manager access (via the job's spa-worker SA / ADC)
#   ngc    -> pull the CDDB dataset (nvidia/clara/proteina-atomistica_data)
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz -o /tmp/gcloud.tgz \
    && tar -xzf /tmp/gcloud.tgz -C /opt && rm /tmp/gcloud.tgz \
    && /opt/google-cloud-sdk/install.sh --quiet --usage-reporting false --path-update false \
    && ln -s /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud \
    && ln -s /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil \
    && curl -fsSL https://ngc.nvidia.com/downloads/ngccli_linux.zip -o /tmp/ngccli.zip \
    && unzip -q /tmp/ngccli.zip -d /opt && rm /tmp/ngccli.zip \
    && ln -s /opt/ngc-cli/ngc /usr/local/bin/ngc \
    && rm -rf /var/lib/apt/lists/*

# GPU exposure for the nvidia-container runtime (Vertex GPU jobs / docker --gpus). The slim base lacks
# these; without NVIDIA_DRIVER_CAPABILITIES=compute the CUDA *driver* libs aren't mounted into the
# container -> torch.cuda.is_available()=False (the 0.2.0 benchmark failed exactly here).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace
CMD ["/bin/bash"]
