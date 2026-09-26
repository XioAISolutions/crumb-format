TITLE: Escaping copy-last collapse in autoregressive video training - what worked for you?

Body:
We are training a small autoregressive video model (not a diffusion model): a recurrent spectral/wave-field mixer (attention-free, FFT-based, ~4M params) that predicts next frames on a synthetic multi-ball physics domain. Code is public: github.com/XioAISolutions/crumb-format (wavefield_video experiments).

The pathology: at 16x16 patches the model learns real motion. At 32x32 (4x tokens, same steps) it converges to COPY-LAST - it freezes: prediction equals last input frame, copy_ratio ~0.005-0.04 of the required movement magnitude. A deterministic-dynamics control (no collisions, where freezing is strictly suboptimal vs loss) also froze, so this is a TRAINABILITY failure, not a loss optimum. Controls that did NOT unfreeze it at 2k steps: constant LR vs OneCycle, fp32 vs bf16, no weight decay, motion-weighted loss, larger batches.

Question for builders who have trained AR video/world models: what actually got you out of collapse/freeze at higher token counts? We are evaluating: self-forced/self-rollout training (Self-Forcing family), variance-covariance regularizers (VICReg-style on frame deltas), discrete history representation + anti-drift sampling (FramePack), rolling-window mutual refinement + attention sink memory (Rolling Forcing), and joint long-budget scaling studies. Concrete recipes, failure reports, and repo pointers welcome.
