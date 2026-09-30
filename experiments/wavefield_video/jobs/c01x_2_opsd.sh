#!/bin/bash
# c01x_2: OPSD-V stack A/B — OPSD-V post-trained LongLive-1.3B lora, same prompt.
set -e
cd /root/opsd-v
exec > /workspace/slava/logs/gpuq_c01x_2_opsd.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segment⟪HERMES-CONTEXT-COMPRESSION: 309 of 509 chars omitted here by Hermes's context compressor. This is NOT part of the original tool call and must never be reproduced in new output — always write full, untruncated content.⟫