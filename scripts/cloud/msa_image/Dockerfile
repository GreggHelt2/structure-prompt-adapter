# spa-msa: MSA-generation image = spa-cloud + ColabFold + MMseqs2-GPU + NGC CLI (dev plan/96 sec 5.9).
# Bakes the tooling the ESM3/RFD3 image lacks so cloud runs start fast and never fail on a runtime
# install (the spa-cloud image has no wget; MMseqs2-GPU is a CUDA static binary, not pip).
FROM us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-cloud:0.3.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl wget unzip ca-certificates tar \
 && rm -rf /var/lib/apt/lists/*

# MMseqs2-GPU static binary (release >= 16; full speed on Hopper H100)
RUN curl -fsSL https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz | tar xz -C /opt \
 && ln -sf /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs

# NGC CLI (pulls the pre-indexed GPU databases)
RUN cd /opt \
 && curl -fsSL -o ngccli.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip \
 && unzip -q ngccli.zip && ln -sf /opt/ngc-cli/ngc /usr/local/bin/ngc && rm -f ngccli.zip

# ColabFold (provides colabfold_search). Torch conflicts are irrelevant: this image only runs MSA.
RUN pip install --no-cache-dir colabfold

# Build-time sanity: fail the BUILD (not a 25-min cloud provision) if any tool is missing.
RUN command -v mmseqs && command -v ngc && command -v colabfold_search && mmseqs version
