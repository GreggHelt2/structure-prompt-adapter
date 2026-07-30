#!/usr/bin/env bash
# CFG Phase-0 sweep driver: 7 prompts x lambda {1,2} at fixed L=150 (dev docs/results/27 section 3b).
# Fixed L isolates the prompt, but note it also introduces a prompt-length confound (results/27 3b).
# Spec: dev docs/plan/56 section 2.6a. Aggregate with dev scripts/analysis/aggregate_cfg_phase0.py.
# Phase-0 extension: more prompts for the spread, plus lambda=2 which addresses a
# pre-registered reason the first result was not decisive ("lambda=1 is the floor of the range").
set -u
cd /home/user1/projects/spa/structure-prompt-adapter
PDB=/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb
D=/tmp/claude-1000/-home-user1-projects-spa-structure-prompt-adapter-dev/f071b2dc-a079-4ca3-ba5e-a99b6ee9fcb9/scratchpad
CK=models/spa-Nx1536-uncond.pt

declare -A P=(
  [A0A522W419]="$PDB/AF-A0A522W419-F1-model_v4_esmfold_v1.pdb"
  [A0A7S3EB45]="$PDB/AF-A0A7S3EB45-F1-model_v4_esmfold_v1.pdb"
  [A0A1X7NTP0]="$PDB/AF-A0A1X7NTP0-F1-model_v4_esmfold_v1.pdb"
  [A0A090ME36]="$PDB/AF-A0A090ME36-F1-model_v4_esmfold_v1.pdb"
  [A0A6A0D1E8]="$PDB/AF-A0A6A0D1E8-F1-model_v4_esmfold_v1.pdb"
  [A0A2X2KHU0]="$PDB/AF-A0A2X2KHU0-F1-model_v4_esmfold_v1.pdb"
  [8SIU]="/home/user1/projects/spa/training_data/eval_external/8SIU.pdb"
)

one() {  # tag lam
  tag=$1; lam=$2
  out="$D/p0_${tag}_lam${lam}.json"
  [ -f "$out" ] && { echo "SKIP $tag lam=$lam (exists)"; return; }
  echo "=== $tag  lambda=$lam ==="
  conda run -n spa-dev python scripts/eval/probe_cfg_delta.py \
    --prompt-pdb "${P[$tag]}" --ckpt "$CK" --length 150 -K 2 --seed 42 \
    --lam "$lam" --out "$out" 2>&1 | grep -E "ratio_median|snr_median|prompt_shift_rms_A_max"
}

# 4 new prompts at lambda=1 -> n=7 for the spread
for t in A0A1X7NTP0 A0A090ME36 A0A6A0D1E8 A0A2X2KHU0; do one "$t" 1.0; done
# lambda=2 on all 7: tests whether the ratio scales with lambda
for t in A0A522W419 A0A7S3EB45 8SIU A0A1X7NTP0 A0A090ME36 A0A6A0D1E8 A0A2X2KHU0; do one "$t" 2.0; done
echo "BATCH DONE"
