import torch
import diffusers
import transformers
import peft
import lmdb
import imageio

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| diffusers", diffusers.__version__,
      "| transformers", transformers.__version__, "| peft", peft.__version__, "| lmdb ok | imageio", imageio.__version__)
