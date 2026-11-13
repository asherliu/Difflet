"""MiniMax-H3 on one CUDA GPU: text-to-video and first-frame image-to-video, with audio.

Every component stays on the device, about 122 GB of weights, so H3 needs a GPU of the
B200's size. ``load`` brings up the transformer (with compiled blocks under
``torch_compile``) and its companions: the Qwen3-VL text encoder with its vision tower, the
visual VAE's encoder (for the keyframe) and decoder, and the audio VAE's decoder. A request
encodes the prompt (and the keyframe), denoises (``denoise.denoise``) and decodes both
streams; the media are returned on the CPU.

The keyframe's canvas, its presentation to the text encoder and its VAE encode are the
shared FL2VA helpers (``difflet.models.minimax_h3.keyframes``), as on the neuron backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from difflet.backends.cuda.minimax_h3.denoise import denoise
from difflet.backends.gpu.application import GpuApplication
from difflet.models.minimax_h3.contracts import build_t2va_layout
from difflet.models.minimax_h3.keyframes import build_fl2va_presentation, prepare_keyframes
from difflet.models.minimax_h3.pipeline import (
    decode_minimax_h3_audio,
    decode_minimax_h3_video,
    load_minimax_h3_schedulers,
)
from difflet.models.minimax_h3.spec import SPEC

# The visual VAE decodes 256-pixel tiles.
_MIN_SIDE = 256
_MAX_PROMPT_TOKENS = SPEC.prompt_budgets[0].max_tokens


@dataclass(frozen=True)
class MiniMaxH3Output:
    #: ``(1, 3, F, H, W)`` in [0, 1].
    video: torch.Tensor
    #: ``(1, 2, samples)`` in [-1, 1] at ``sample_rate``.
    audio: torch.Tensor
    sample_rate: int


class CudaMiniMaxH3Application(GpuApplication):
    spec = SPEC

    def _configure(self, kwargs) -> None:
        if self.dtype != torch.bfloat16:
            raise ValueError("MiniMax-H3 on CUDA runs in bf16")
        if min(self.shape["height"], self.shape["width"]) < _MIN_SIDE:
            raise ValueError(f"MiniMax-H3 on CUDA needs height and width of at least {_MIN_SIDE}")
        # The reference layout checks the shape before any checkpoint I/O.
        self.num_latent_frames = build_t2va_layout(
            num_text_tokens=1, **self.shape
        ).num_latent_frames

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.minimax_h3.transformer import CudaMiniMaxH3TransformerApplication

        self.transformer = CudaMiniMaxH3TransformerApplication(
            model_path=self.transformer_path, device=self.device, torch_compile=self.torch_compile
        )

    def load_companions(self) -> None:
        from transformers import AutoImageProcessor, AutoTokenizer

        from difflet.backends.cuda.minimax_h3.decoders import (
            load_audio_decoder,
            load_video_decoder,
        )
        from difflet.backends.cuda.minimax_h3.text_encoder import load_text_encoder
        from difflet.models.minimax_h3.host_components import load_host_video_encoder

        root = Path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        # Only the image processor: the full Qwen3-VL processor also builds a video processor
        # that needs torchvision.
        self.image_processor = AutoImageProcessor.from_pretrained(
            root / "processor", local_files_only=True
        )
        self.text_encoder = load_text_encoder(
            root / "text_encoder", dtype=self.dtype, device=self.device
        )
        # The shared tiled keyframe encoder, its FP32 module moved to the device.
        self.video_encoder = load_host_video_encoder(root)
        self.video_encoder.module.to(self.device)
        self.video_decoder = load_video_decoder(str(root / "vae"), device=self.device)
        self.audio_decoder = load_audio_decoder(str(root / "audio_vae"), device=self.device)
        self.video_scheduler, self.audio_scheduler = load_minimax_h3_schedulers(root)

    # ------------------------------------------------------------ encoders

    def encode_prompt(self, prompt: str, keyframes=()) -> tuple[torch.Tensor, torch.Tensor]:
        """The hidden states of the shared FL2VA presentation, ``(1, tokens, 5120)``, and each
        token's row tag. T2VA (no keyframes) is the prompt through the language model; with a
        keyframe, ``"<Picture 1>: "`` and its image block precede the prompt and go through the
        vision tower too, the image block video-tagged."""
        num_prompt_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not 1 <= num_prompt_tokens <= _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"MiniMax-H3 prompts take 1-{_MAX_PROMPT_TOKENS} tokens, got {num_prompt_tokens}"
            )
        token_ids, token_tags, vision = build_fl2va_presentation(
            self.tokenizer, self.image_processor, prompt, keyframes
        )
        ids = torch.tensor([token_ids], device=self.device)
        if not vision:
            hidden = self.text_encoder.language_model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
            ).last_hidden_state
            return hidden, token_tags
        pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        hidden = self.text_encoder(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=vision["pixel_values"].to(self.device),
            image_grid_thw=vision["image_grid_thw"].to(self.device),
            mm_token_type_ids=(ids == pad).to(torch.int32),
            use_cache=False,
        ).last_hidden_state
        return hidden, token_tags

    def encode_keyframe(self, image, *, seed: int) -> torch.Tensor:
        """The keyframe's fixed, noise-augmented anchor rows ``(1, rows_per_frame, 96)``."""
        from difflet.backends.cuda.minimax_h3.keyframe import encode_keyframe, keyframe_anchor_rows

        latent = encode_keyframe(self.video_encoder, image, device=self.device)
        return keyframe_anchor_rows(
            latent, seed=seed, num_target_latent_frames=self.num_latent_frames
        )

    # ------------------------------------------------------------- request

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        num_inference_steps: int = SPEC.default_steps,
        seed: int = 0,
        num_frames: int | None = None,
        first_image=None,
        last_image=None,
        output_type: str = "pt",
    ) -> MiniMaxH3Output:
        """One clip with its audio. ``num_inference_steps`` counts denoising steps (the
        reference's convention); ``first_image`` (a PIL image) conditions the first frame."""
        if self.transformer is None or not self.transformer.is_loaded:
            raise RuntimeError("MiniMax-H3 is not loaded; call load() first")
        if output_type != "pt":
            raise ValueError("MiniMax-H3 returns tensors (output_type='pt')")
        if num_frames not in (None, self.shape["num_frames"]):
            raise ValueError(f"MiniMax-H3 was loaded for {self.shape['num_frames']} frames")
        if last_image is not None:
            raise ValueError("MiniMax-H3 on CUDA conditions on a first frame only")
        keyframes, keyframe_rows = [], None
        if first_image is not None:
            keyframes, _, _, _ = prepare_keyframes(
                first_image, None, height=self.shape["height"], width=self.shape["width"]
            )
        text, text_tags = self.encode_prompt(prompt, keyframes)
        if keyframes:
            keyframe_rows = self.encode_keyframe(keyframes[0], seed=seed)
        latents = denoise(
            self.transformer,
            encoder_hidden_states=text,
            **self.shape,
            num_inference_steps=num_inference_steps,
            seed=seed,
            video_scheduler=self.video_scheduler,
            audio_scheduler=self.audio_scheduler,
            device=self.device,
            text_token_tags=text_tags,
            keyframe_rows=keyframe_rows,
        )
        config = self.video_decoder.config
        # The video is decoded and stitched on the device; the small audio latents are
        # denormalized on the host, as the shared audio decode does.
        video = decode_minimax_h3_video(
            self.video_decoder,
            latents.video_latents,
            latents_mean=config.latents_mean,
            latents_std=config.latents_std,
            clip_length=config.clip_length,
            token_drop=config.token_drop,
            temporal_compression_ratio=config.temporal_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
        ).cpu()
        config = self.audio_decoder.config
        audio = decode_minimax_h3_audio(
            self.audio_decoder,
            latents.audio_latents.cpu(),
            latents_mean=config.latents_mean,
            latents_std=config.latents_std,
        ).cpu()
        return MiniMaxH3Output(video=video, audio=audio, sample_rate=int(config.sampling_rate))


__all__ = ["CudaMiniMaxH3Application", "MiniMaxH3Output"]
