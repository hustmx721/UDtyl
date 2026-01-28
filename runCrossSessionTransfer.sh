#!/usr/bin/env bash
set -euo pipefail

# Train clean brainprint models and evaluate robustness on pretrained STFT perturbations.
# Example uses the ShallowConvNet resample-EOT delta as source perturbation.

DATASET=${1:-Rest}
GPU=${2:-0}
TGT_MODEL=${3:-BrainprintNet}
SRC_MODEL=${4:-ShallowConvNet}

python -u main_UID.py \
  --dataset "${DATASET}" \
  --gpuid "${GPU}" \
  --model "${TGT_MODEL}" \
  --src_model "${SRC_MODEL}" \
  --eot_resample 0.02 \
  --eot_resample_prob 1.0
