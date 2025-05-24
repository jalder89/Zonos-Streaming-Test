import json
from typing import Callable, Generator

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner, make_cond_dict
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))
UNKNOWN_TOKEN = -1


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls,
        config_path: str,
        model_path: str,
        device: str = DEFAULT_DEVICE,
        backbone: str | None = None,
        speaker_models_path: str | None = None,
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)

        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        model.load_state_dict(sd)

        if speaker_models_path:
            model.spk_clone_model = SpeakerEmbeddingLDA(device, speaker_models_path)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # TODO: support cfg_scale==1
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    def _preprocess_codes_for_decoding(self, codes: torch.Tensor, context: str = "") -> torch.Tensor:
        """
        Preprocess codes to handle unknown tokens and ensure valid range for DAC autoencoder.
        Returns the processed codes tensor.
        """
        try:
            # Check for unknown tokens and invalid values
            min_val, max_val = codes.min().item(), codes.max().item()
            
            if min_val < 0 or max_val >= 1024:
                unknown_count = (codes < 0).sum().item() if min_val < 0 else 0
                invalid_count = (codes >= 1024).sum().item() if max_val >= 1024 else 0
                
                if unknown_count > 0:
                    print(f"[PREPROCESS] {context}: Found {unknown_count} unknown tokens (-1), clamping to 0")
                if invalid_count > 0:
                    print(f"[PREPROCESS] {context}: Found {invalid_count} invalid tokens (>=1024), clamping to 1023")
                
                # Clamp to valid range [0, 1023]
                codes = torch.clamp(codes, min=0, max=1023)
                print(f"[PREPROCESS] {context}: Preprocessed codes range: [{codes.min().item()}, {codes.max().item()}]")
            
            return codes
            
        except Exception as e:
            print(f"[PREPROCESS ERROR] {context}: {e}")
            return codes

    def _safe_autoencoder_decode(self, codes: torch.Tensor, context: str = "") -> torch.Tensor:
        """
        Safely decode codes with proper error handling and preprocessing.
        """
        try:
            # Ensure codes are on the correct device and dtype
            codes = codes.to(self.device)
            
            # Preprocess codes to handle unknown tokens
            codes = self._preprocess_codes_for_decoding(codes, context)
            
            # Validate basic tensor properties
            if codes is None or codes.numel() == 0:
                raise ValueError("Empty or None codes tensor")
            
            if codes.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {codes.dim()}D with shape {codes.shape}")
            
            # Clear GPU cache before decoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Attempt decoding with autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                decoded_audio = self.autoencoder.decode(codes)
            
            # Validate output
            if decoded_audio is None or len(decoded_audio) == 0:
                raise RuntimeError("Autoencoder returned empty result")
            
            return decoded_audio[0]
            
        except Exception as e:
            print(f"[DECODE ERROR] {context}: {type(e).__name__}: {e}")
            
            # Try CPU fallback if CUDA failed
            if codes.device.type == "cuda":
                try:
                    print(f"[DECODE FALLBACK] {context}: Attempting CPU decode")
                    codes_cpu = self._preprocess_codes_for_decoding(codes.cpu(), f"{context}_cpu")
                    autoencoder_cpu = self.autoencoder.dac.cpu()
                    
                    with torch.no_grad():
                        decoded_cpu = autoencoder_cpu.decode(audio_codes=codes_cpu).audio_values.unsqueeze(1).float()
                    
                    # Move back to GPU
                    self.autoencoder.dac.to(self.device)
                    return decoded_cpu.to(self.device)
                    
                except Exception as cpu_e:
                    print(f"[DECODE FALLBACK FAILED] {context}: {cpu_e}")
            
            # Return silence as last resort
            batch_size, _, seq_len = codes.shape
            samples_per_token = 512  # Approximate DAC ratio
            silence_length = seq_len * samples_per_token
            print(f"[DECODE FALLBACK] {context}: Returning silence of length {silence_length}")
            return torch.zeros((batch_size, 1, silence_length), device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        with torch.device(device):
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            codes = torch.full((batch_size, 9, audio_seq_len), UNKNOWN_TOKEN)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS
        # Make EOS less likely because audio often is cut off
        logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device))

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)

        step = 0
        while torch.max(remaining_steps) > 0:
            offset += 1
            input_ids = delayed_codes[..., offset - 1 : offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits += logit_bias

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
            stopping |= eos_in_cb0[:, 0]

            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            remaining_steps -= 1

            progress.update()
            step += 1

            if callback is not None and not callback(frame, step, max_steps):
                break

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        self._cg_graph = None  # reset cuda graph to avoid cache changes

        return out_codes

    @torch.inference_mode()
    def stream(
        self,
        cond_dicts_generator: Generator[dict, None, None],
        audio_prefix_codes: torch.Tensor | None = None,
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        sampling_params: dict = dict(min_p=0.1),
        disable_torch_compile: bool = False,
        chunk_schedule: list[int] = [16, *range(9, 100)],
        chunk_overlap: int = 2,
        whitespace: str = " ",
        mark_boundaries: bool = False,
    ) -> Generator[torch.Tensor | str, None, None]:
        """
        Stream audio generation in chunks with smooth transitions between chunks.
        FIXED VERSION v2 with improved audio cutoff handling.

        Args:
            cond_dicts_generator: Generator of conditioning dictionaries
            audio_prefix_codes: Optional audio prefix codes
            max_new_tokens: Maximum number of new tokens to generate
            cfg_scale: Classifier-free guidance scale
            sampling_params: Parameters for sampling from logits
            disable_torch_compile: Whether to disable torch.compile
            chunk_schedule: List of chunk sizes to use in sequence (will use the last size for remaining chunks)
            chunk_overlap: Number of tokens to overlap between chunks (also determines audio crossfade size)
            whitespace: Whitespace to use between sentences
            mark_boundaries: Whether to yield sentence strings as indicators of the sentence end

        Yields:
            Audio chunks as torch tensors [and sentence strings as indicators of the sentence end if mark_boundaries is True]
        """
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        assert len(chunk_schedule) > 0, "chunk_schedule must not be empty"
        assert all(chunk_overlap * 2 < size for size in chunk_schedule), "overlap must be less than a half of a chunk"

        batch_size = 1  # Streaming generation is single-sample only
        device = self.device

        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        # Calculate window size based on overlap tokens (approx. samples per token)
        samples_per_token = 512  # Approximate value based on DAC model
        overlap = chunk_overlap * samples_per_token

        # Create cosine fade for smooth transition
        cosfade = 0.5 * (1 + torch.cos(torch.linspace(torch.pi, 0, overlap, device=device)))

        # Set up first text and codes to use as a prefix for all generations
        audio_prefix_text = ""
        previous_audio = None
        generator_index = 0

        # Main loop: iterate over sentences in the cond_dicts_generator. For each sentence, we'll be streaming audio chunks out.
        # Once the first sentence is ready, we'll use it's codes as audio_prefix_codes for all the next sentences
        for cond_dict in cond_dicts_generator:
            # Prepend the conditioning dictionary text with the previous sentence text
            curr_text = cond_dict["text"]
            updated_cond_dict = {**cond_dict, "text": audio_prefix_text + curr_text + whitespace}

            prefix_conditioning = self.prepare_conditioning(make_cond_dict(**updated_cond_dict))

            prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
            audio_seq_len = prefix_audio_len + max_new_tokens
            seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

            with torch.device(device):
                inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
                codes = torch.full((batch_size, 9, audio_seq_len), UNKNOWN_TOKEN)

            if audio_prefix_codes is not None:
                codes[..., :prefix_audio_len] = audio_prefix_codes

            delayed_codes = apply_delay_pattern(codes, self.masked_token_id)
            delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

            logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
            next_token = sample_from_logits(logits, **sampling_params)

            offset = delayed_prefix_audio_codes.shape[2]
            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)

            prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
            inference_params.seqlen_offset += prefix_length
            inference_params.lengths_per_sample[:] += prefix_length

            logit_bias = torch.zeros_like(logits)
            logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS
            # Make EOS less likely because audio often is cut off
            logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device))

            # --- Autoregressive loop ---
            stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
            max_steps = delayed_codes.shape[2] - offset
            remaining_steps = torch.full((batch_size,), max_steps, device=device)
            step = 0
            # This variable will let us yield only the new audio since the last yield.
            yielded_len = prefix_audio_len
            chunk_counter = 0

            # For chunk scheduling
            schedule_index = 0

            while torch.max(remaining_steps) > 0:
                offset += 1
                input_ids = delayed_codes[..., offset - 1 : offset]
                logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
                logits += logit_bias

                next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)

                # Update stopping for finished samples.
                eos_in_cb0 = next_token[:, 0] == self.eos_token_id
                remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
                stopping |= eos_in_cb0[:, 0]

                eos_codebook_idx = 9 - remaining_steps
                eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
                for i in range(next_token.shape[0]):
                    if stopping[i]:
                        idx = eos_codebook_idx[i].item()
                        next_token[i, :idx] = self.masked_token_id
                        next_token[i, idx] = self.eos_token_id

                frame = delayed_codes[..., offset : offset + 1]
                frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)
                inference_params.seqlen_offset += 1
                inference_params.lengths_per_sample[:] += 1
                remaining_steps -= 1
                step += 1
                chunk_counter += 1

                # --- Every 'chunk_size' tokens (or when finished), decode and yield the new audio ---
                if (chunk_counter + chunk_overlap + 9 >= chunk_schedule[schedule_index]) or (
                    torch.all(remaining_steps == 0)
                ):
                    try:
                        # In Zonos, the final output codes are produced by reverting the delay pattern.
                        # Only tokens up to (offset - 9) are valid.
                        full_codes = revert_delay_pattern(delayed_codes)
                        full_codes.masked_fill_(full_codes >= 1024, 0)

                        # Get the valid portion of the latent sequence.
                        valid_length = offset - 9
                        
                        # FIXED: Add bounds checking and minimum chunk size validation
                        chunk_start = max(0, yielded_len)
                        chunk_end = max(chunk_start + 1, valid_length)  # Ensure at least 1 token
                        
                        if chunk_start >= chunk_end:
                            print(f"[STREAM WARNING] Skipping invalid chunk: start={chunk_start}, end={chunk_end}")
                            continue
                        
                        partial_codes = full_codes[..., chunk_start:chunk_end]
                        
                        # FIXED: Use safe decoding with automatic unknown token handling
                        context = f"streaming_chunk_{step}_len_{chunk_end - chunk_start}"
                        current_audio = self._safe_autoencoder_decode(partial_codes, context)
                        
                        # Apply crossfading if we have previous audio
                        if current_audio.shape[-1] > 0:
                            size = min(overlap, current_audio.shape[-1])
                            if size > 0:
                                current_audio[..., :size] *= cosfade[-size:]
                                if previous_audio is not None and previous_audio.shape[-1] >= size:
                                    current_audio[..., :size] += previous_audio[..., -size:] * (1 - cosfade[:size])

                            # FIXED: Improved fade-in for the first chunk to prevent audio cutoff
                            if schedule_index == 0 and generator_index == 0:  # Only for very first chunk of first sentence
                                # Gentle fade-in over more samples to preserve initial audio
                                fade_samples = min(overlap * 3, current_audio.shape[-1])  # Longer fade-in
                                if fade_samples > 0:
                                    # Use cosine fade instead of log fade for smoother transition
                                    gentle_fade = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, fade_samples, device=device)))
                                    current_audio[..., :fade_samples] *= gentle_fade
                                    print(f"[STREAM] {context}: Applied gentle fade-in over {fade_samples} samples")

                            # Yield the chunk (keeping overlap for next iteration)
                            yield_length = max(0, current_audio.shape[-1] - overlap)
                            if yield_length > 0:
                                yield current_audio[..., :yield_length]

                            # Store current audio for next iteration and update counters
                            previous_audio = current_audio
                            yielded_len = chunk_end - chunk_overlap
                            chunk_counter = 0

                            # Update chunk size according to schedule
                            if schedule_index < len(chunk_schedule) - 1:
                                schedule_index += 1
                    
                    except Exception as e:
                        print(f"[STREAM ERROR] Failed to process chunk at step {step}: {e}")
                        # Continue to next iteration instead of crashing
                        continue

            if generator_index == 0:
                # Assemble the full codes for this sentence and set the audio_prefix_codes to equal first sentence generated audio
                try:
                    audio_prefix_codes = revert_delay_pattern(delayed_codes)
                    audio_prefix_codes.masked_fill_(audio_prefix_codes >= 1024, 0)
                    audio_prefix_codes = audio_prefix_codes[..., : offset - 9]
                    audio_prefix_text = curr_text + whitespace
                except Exception as e:
                    print(f"[STREAM ERROR] Failed to set audio prefix: {e}")

            # Fade out the final chunk for this sentence
            if previous_audio is not None:
                try:
                    size = min(2 * overlap, previous_audio.shape[-1])
                    if size > 0:
                        logfade = torch.logspace(1, 0, size, base=20, device=device)
                        logfade -= logfade.min()
                        logfade /= logfade.max()
                        previous_audio[..., -size:] *= logfade
                except Exception as e:
                    print(f"[STREAM ERROR] Failed to fade out: {e}")

            self._cg_graph = None  # reset CUDA graph to avoid caching issues
            generator_index += 1

            # Yield the sentence string if mark_boundaries is True
            if mark_boundaries:
                yield curr_text

        # Don't forget to yield the final audio chunk
        if previous_audio is not None:
            yield previous_audio