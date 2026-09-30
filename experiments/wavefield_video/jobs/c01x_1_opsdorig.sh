#!/bin/bash
# c01x_1: OPSD-V stack A/B — ORIGINAL LongLive-1.3B (released lora), street prompt.
set -e
cd /root/opsd-v
exec > /workspace/slava/logs/gpuq_c01x_1_opsdorig.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable⟪HERMES-CONTEXT-COMPRESSION: 353 of 553 chars omitted here by Hermes's context compressor. This is NOT part of the original tool call and must never be reproduced in new output — always write full, untruncated content.⟫