"""Colab/Jupyter quickstart — paste this into a cell.

Gets a working Crumb LLM generating coherent Shakespeare in one cell.
Requires GPU runtime (T4 is fine, ~10 min).
"""

# Cell 1: Install
# !pip install -q crumb-format[llm]
# !pip install -q crumb-format  # if [llm] fails, install separately: pip install torch

# Cell 2: Train + Demo
import subprocess, sys, os

# Download Shakespeare
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "/tmp/shakespeare.txt"
)

# Clone repo for full source
if not os.path.exists("crumb-format"):
    subprocess.run(["git", "clone", "https://github.com/XioAISolutions/crumb-format.git"], check=True)

os.chdir("crumb-format")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[llm]"], check=True)

# Train (small config: 256-dim, 6 layers, ~3.7M params)
subprocess.run([
    sys.executable, "-m", "crumb_llm.train",
    "--config", "small",
    "--steps", "10000",
    "--data", "/tmp/shakespeare.txt",
    "--out", "/tmp/crumb_model",
    "--log-every", "500",
    "--eval-every", "2000",
], check=True)

# Generate
import torch
from crumb_wavelm.sample import load_checkpoint

model, tok = load_checkpoint("/tmp/crumb_model")
model.eval()
if torch.cuda.is_available():
    model.cuda()

prompts = [
    "First Citizen:\nBefore we proceed any further,",
    "ROMEO:\nBut, soft! what light through yonder window breaks?",
    "To be, or not to be, that is the question:",
]

for p in prompts:
    ids = torch.tensor(tok.encode(p), dtype=torch.long).unsqueeze(0)
    if torch.cuda.is_available():
        ids = ids.cuda()
    out = model.generate(ids, max_new_tokens=300, temperature=0.8, top_k=40)
    text = tok.decode(out[0].tolist())
    print(f"\n{'='*60}")
    print(f"PROMPT: {p}")
    print(f"{'='*60}")
    print(text[:500])
