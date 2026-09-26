
## 2. FramePack (Lvmin Zhang et al., NeurIPS 2025, arXiv 2504.12626, github.com/lllyasviel/FramePack)
- Next-frame-section prediction; constant-length packed context (workload invariant to length).
- Anti-drift toolkit: EARLY-ESTABLISHED ENDPOINTS, ADJUSTED SAMPLING ORDERS, DISCRETE HISTORY REPRESENTATION (v1: Planned Anti-Drifting + History Discretization). 13B model at 6GB VRAM.

## 3. Collapse taxonomy (JEPA/world-model literature)
- Loss can be low while representations are degenerate: entire collapse, DIMENSION collapse, MEAN-LEARNING DEFICIENCY. Practitioners judge with probes, not loss. Candidates for us: variance-covariance regularization (VICReg-style) on frame deltas.
- SurgVista (surgical world models, 2026): fixes temporal fidelity collapse via trajectory-based contrastive regularization + DRIFT-PERTURBED TRAINING.

## 4. Hybrid SSM/attention (our Rank-1 hybrid change)
- Production consensus 2026: pure SSM fails exact recall; HYBRIDS win (Zamba2, Hymba, Falcon-H1 patterns). SSM-Scope eval suite (sapmitra.github.io/ssm-scope).
- Tiny-scale reference: github.com/Karan-Anchan/mamba-hybrid-lm (50M, attention:SSM ratios 1:3/1:7/1:15, same-budget comparisons) - our kind of experiment design.

## 5. What we adopt (in priority order)
1) Self-rollout training w/ detached own-forwards (we have it) - escalate per decision tree; Self-Forcing = recipe validation for long horizons.
2) Variance regularizer on deltas (VICReg-style) - new anti-collapse arm.
3) Rolling-window mutual refinement + attention-sink style anchor - ALREADY our plugin shape (spectral state anchor); their attention sink validates the design.
4) Discrete history representation - quantized state experiment for the plugin.
5) Drift-perturbed training - inject drift into training clips as augmentation.
