import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
