#!/usr/bin/env bash
set -euo pipefail

# Evaluate brainprint recognition under a pretrained STFT perturbation.
# Example uses the ShallowConvNet resample-EOT delta as source perturbation.

DATASET=${1:-Rest}
GPU=${2:-0}
TGT_MODEL=${3:-BrainprintNet}

python -u main_transfer.py \
  --dataset "${DATASET}" \
  --gpuid "${GPU}" \
  --src_model ShallowConvNet \
  --tgt_model "${TGT_MODEL}" \
  --eot_resample 0.02 \
  --eot_resample_prob 1.0
