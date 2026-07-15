[![SPA ISMB 2026 poster](docs/poster/SPA_ISMB2026_poster_FINAL_thumbnail.png)](docs/poster/SPA_ISMB2026_poster_FINAL.pdf)

The initial research in this repo was presented as a poster at the ISMB 2026 conference, July 12-16 in Washington DC.
Structure Prompt Adapter source code and trained models are both included in this repo.

---

## Structure Prompt Adapter (SPA): Flexible Structural Conditioning for RFdiffusion3
All-atom diffusion models like RFdiffusion3 excel at de novo protein design. However, RFdiffusion3 enforces 3D structural conditioning by explicitly keeping specified atomic coordinates fixed within the iterative diffusion loop. While precise for motif scaffolding, this rigid geometric constraint can limit higher level topological flexibility. Here we introduce the Structure Prompt Adapter (SPA), a parameter-efficient method enabling additional, more flexible structural conditioning when coupled with RFDiffusion3. Inspired by the Image Prompt Adapter (IP-Adapter) used in image diffusion workflows, SPA is a lightweight sidecar model that injects decoupled cross-attention into the RFdiffusion3 token track. It uses the Contrastive Learning Sequence-Structure (CLSS) encoder for encoding its additional structural prompts. This effectively guides coarser fold topology, while the native RFDiffusion3 cross-attention between atom level and token level tracks translates this guidance to the atom level track for positioning. Because RFdiffusion3 natively processes absolute Cartesian coordinates and CLSS encodes geometric topologies as invariant latent features, SPA can integrate these representations without computationally heavy geometric transformation layers. Trained on diverse 3D structures, SPA uses extensive data augmentation to learn spatial symmetries. We also train alternate versions of SPA that use only the outputs of CLSS or deeper layers within CLSS connected to trainable layers in SPA. To validate SPA-guided output, we utilize ProteinMPNN for inverse-folding sequence design, followed by in silico validation using OpenFold3. We compare and contrast specific examples of protein design using RFdiffusion3 with or without SPA. Source code and models are available at: github.com/GreggHelt2/structure-prompt-adapter.

---

## Installation

Get a working SPA + RFdiffusion3 environment with one script:

    bash scripts/setup/install_env.sh

This creates a conda environment (`spa`, Python 3.12), installs the validated combination of PyTorch
(2.5.1+cu124), RFdiffusion3 (via the `foundry` package), `atomworks`, and ESM3 as editable installs from
pinned commits, installs this package itself, and downloads the RFdiffusion3 base checkpoint. Add
`--with-clss` if you want the 1×32 CLSS-framed variant too. Requires conda and an NVIDIA GPU with a
CUDA 12.4-compatible driver.

SPA's own trained adapter weights are already included in this repo, under `models/` (see
[`models/README.md`](models/README.md) for which one to use).

**Note:** ESM3 weights auto-download from Hugging Face on first use. Even though they're MIT-licensed
and ungated, you still need a free Hugging Face account and a Read token (`huggingface-cli login`).

Once installed:

```bash
conda activate spa
python scripts/eval/generate.py \
    variant=C_n_by_1536 eval.ckpt=models/spa-Nx1536-uncond.pt \
    eval.prompt_pdb=/path/to/prompt.pdb 'eval.conditions=[baseline,spa]' \
    'eval.lambda_scale=[0.5,1.0]' eval.num_designs=8 eval.length=100
```
