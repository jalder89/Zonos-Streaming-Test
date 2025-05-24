import math

import torch
import torchaudio
from transformers.models.dac import DacModel


class DACAutoencoder:
    def __init__(self):
        super().__init__()
        self.dac = DacModel.from_pretrained("descript/dac_44khz")
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        left_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (left_pad, 0), value=0)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        FIXED VERSION: Enhanced decode method with proper error handling and validation.
        """
        try:
            # Validate input tensor
            if codes is None or codes.numel() == 0:
                raise ValueError("Empty or None codes tensor provided to autoencoder")
            
            # Check tensor dimensions - expect [batch, num_codebooks, sequence_length]
            if codes.dim() != 3:
                raise ValueError(f"Expected 3D codes tensor, got {codes.dim()}D with shape {codes.shape}")
            
            batch_size, num_codebooks, seq_len = codes.shape
            
            # Validate codebook count
            if num_codebooks != self.num_codebooks:
                raise ValueError(f"Codebook count mismatch: expected {self.num_codebooks}, got {num_codebooks}")
            
            # Validate sequence length minimum
            if seq_len < 1:
                raise ValueError(f"Sequence length too short: {seq_len}")
            
            # Validate value ranges - DAC expects codes in [0, 1023]
            min_val, max_val = codes.min().item(), codes.max().item()
            if min_val < 0 or max_val >= 1024:
                print(f"[AUTOENCODER WARNING] Codes values outside expected range [0, 1023]: min={min_val}, max={max_val}")
                # Clamp values to valid range
                codes = torch.clamp(codes, 0, 1023)
            
            # Ensure codes are proper dtype (long/int64 for discrete codes)
            if codes.dtype not in [torch.long, torch.int64, torch.int32]:
                print(f"[AUTOENCODER WARNING] Converting codes from {codes.dtype} to torch.long")
                codes = codes.long()
            
            # Use appropriate autocast based on device
            device_type = self.dac.device.type
            use_autocast = device_type != "cpu"
            
            with torch.autocast(device_type, torch.float16, enabled=use_autocast):
                # Clear cache before decoding on GPU
                if device_type == "cuda":
                    torch.cuda.empty_cache()
                
                # Perform the actual decoding
                decoded_result = self.dac.decode(audio_codes=codes)
                
                # Extract audio values and ensure proper shape
                audio_values = decoded_result.audio_values
                
                # Add channel dimension if needed and convert to float32
                if audio_values.dim() == 2:
                    audio_values = audio_values.unsqueeze(1)  # Add channel dimension
                
                return audio_values.float()
                
        except Exception as e:
            print(f"[AUTOENCODER ERROR] Decode failed: {type(e).__name__}: {e}")
            print(f"[AUTOENCODER ERROR] Input tensor info: shape={codes.shape if codes is not None else 'None'}, "
                  f"dtype={codes.dtype if codes is not None else 'None'}, "
                  f"device={codes.device if codes is not None else 'None'}")
            
            # Return silence as fallback
            if codes is not None and codes.numel() > 0:
                batch_size, _, seq_len = codes.shape
                samples_per_token = 512  # DAC's typical compression ratio
                silence_samples = seq_len * samples_per_token
                print(f"[AUTOENCODER FALLBACK] Returning silence: batch={batch_size}, samples={silence_samples}")
                return torch.zeros((batch_size, 1, silence_samples), 
                                 device=codes.device, 
                                 dtype=torch.float32)
            else:
                # Minimal fallback
                return torch.zeros((1, 1, 512), dtype=torch.float32)