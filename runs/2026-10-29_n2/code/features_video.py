"""Video encoders with temporal modelling (PLAN §5): the counterpart of the keyframe-only `clip_frames`.

Same Strategy contract as features.py: `modality`, `neural`, `dim`, `fingerprint()`, `load()` and
`encode(df, budget_s, log)` -> (len(df), dim) float32, NaN where a window could not be encoded, `coverage` set when a
time budget runs out.

Design. A window is the mean of the embeddings of the non-overlapping *blocks* of `block_s` seconds that it covers. With
window_s=4.8 and hop_s=2.4 (configs/pilot_n1.yaml) a window is two blocks and neighbouring windows share one, so every
block is encoded once. The frames of a block are taken at the centres of `frames` equal sub-intervals, so the frame
spacing is block_s/frames (0.15 s for 16 frames, 0.30 s for 8). Frames are resized to 224 x 224 without cropping, as
`clip_frames` does, so the encoders differ in the model and not in the framing.

Known limits: the frame spacing is wider than the 1/30 s the Kinetics models were trained with, so fast motion is
sampled coarsely; audio and video are aligned only through the shared window grid (no clock-offset correction); each
encoder decodes every video itself (three video encoders decode three times).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .core import register
from .data import _h
from .features import _device, _tensor_out

SIZE = 224
CHUNK = 96  # blocks decoded and held in memory at once: 96 blocks x 16 frames x 150 kB = 230 MB
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def block_times(blocks, block_s: float, frames: int) -> np.ndarray:
    """(n_blocks, frames) sampling times: the centres of `frames` equal sub-intervals of every block."""
    b = np.asarray(blocks, np.float64)[:, None]
    return b * block_s + (np.arange(frames)[None, :] + 0.5) * block_s / frames


def decode_blocks(path: str, blocks, block_s: float, frames: int, size: int = SIZE) -> np.ndarray:
    """uint8 (len(blocks), frames, size, size, 3): for each target time the first decoded frame at or after it.

    `blocks` must be ascending. The container is seeked to just before the first target, so a few blocks from the
    middle of a long video do not cost a full decode."""
    import av
    tt = block_times(blocks, block_s, frames).reshape(-1)
    out = np.empty((len(tt), size, size, 3), np.uint8)
    ptr, last = 0, None
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if tt[0] > 2.0:
            c.seek(int((tt[0] - 1.0) * 1e6))  # lands on a keyframe at or before this time
        for fr in c.decode(st):
            if fr.time is None or float(fr.time) < tt[ptr]:
                continue
            last = fr.reformat(width=size, height=size, format="rgb24").to_ndarray()
            while ptr < len(tt) and tt[ptr] <= float(fr.time):
                out[ptr] = last
                ptr += 1
            if ptr >= len(tt):
                break
    if ptr < len(tt):
        if last is None:
            raise ValueError(f"{path}: no frame at or after t={tt[ptr]:.2f}s")
        out[ptr:] = last  # the clip ends within one frame of the last targets
    return out.reshape(len(blocks), frames, size, size, 3)


class BlockVideoEncoder:
    """Base class: decoding, blocks, window assembly, budget and error handling. Subclasses implement embed_blocks."""
    modality, neural, dim, batch, frames = "video", True, 0, 8, 16

    def __init__(self, block_s: float = 2.4):
        self.block_s = float(block_s)

    def fingerprint(self) -> str:
        return f"{self.registry_name}-b{self.block_s:g}-t{self.frames}-v1"

    def load(self):
        pass

    def embed_blocks(self, clips: np.ndarray) -> np.ndarray:
        """(n, frames, 224, 224, 3) uint8 -> (n, dim) float32."""
        raise NotImplementedError

    def window_blocks(self, t0: float, t1: float) -> list[int]:
        b0, b1 = t0 / self.block_s, t1 / self.block_s
        if abs(b0 - round(b0)) > 1e-3 or abs(b1 - round(b1)) > 1e-3 or round(b1) <= round(b0):
            raise ValueError(f"window [{t0}, {t1}] s does not consist of whole blocks of {self.block_s} s; "
                             f"construct the encoder with block_s equal to the hop (window_s a multiple of it)")
        return list(range(int(round(b0)), int(round(b1))))

    def _embed(self, clips: np.ndarray) -> np.ndarray:
        return np.concatenate([self.embed_blocks(clips[i:i + self.batch]) for i in range(0, len(clips), self.batch)])

    def encode(self, df: pd.DataFrame, budget_s: float | None = None, log=None) -> np.ndarray:
        """Videos are visited in a fixed pseudo-random order; on a budget overrun the rest stay NaN (see AudioEncoder)."""
        self.load()
        t_start = time.time()
        d = df.reset_index(drop=True)
        out = np.full((len(d), self.dim), np.nan, np.float32)
        # a misaligned window is a configuration error, so it raises here and is not skipped like an unreadable video
        d["blocks_"] = pd.Series([self.window_blocks(a, b) for a, b in zip(d.t0.values, d.t1.values)], dtype=object)
        groups = sorted(d.groupby("video_path", sort=False), key=lambda g: _h(g[0], "encode-order"))
        self.coverage = 1.0
        for k, (vid, sub) in enumerate(groups):
            if budget_s and time.time() - t_start > budget_s:
                self.coverage = k / len(groups)
                if log:
                    log.warning(f"  {self.registry_name}: budget of {budget_s:.0f}s used up after {k}/{len(groups)} videos")
                break
            need = sorted({b for bl in sub["blocks_"] for b in bl})
            emb: dict[int, np.ndarray] = {}
            try:
                for s in range(0, len(need), CHUNK):
                    chunk = need[s:s + CHUNK]
                    emb.update(zip(chunk, self._embed(decode_blocks(vid, chunk, self.block_s, self.frames))))
            except Exception as ex:  # noqa: BLE001  (one unreadable video must not stop the encoder)
                if log:
                    log.warning(f"  {self.registry_name}: cannot encode {vid}: {type(ex).__name__}: {ex}")
                continue
            for i, bl in zip(sub.index, sub["blocks_"]):
                out[i] = np.mean([emb[b] for b in bl], 0)
            if log and k % 25 == 0:
                log.info(f"  {self.registry_name}: {k}/{len(groups)} videos")
        return out


# ----------------------------------------------------------------------------- non-neural
@register("encoder", "motion_stats")
class MotionStats(BlockVideoEncoder):
    """Non-neural motion descriptor, the video counterpart of mfcc_stats: absolute differences between consecutive
    frames on a 4 x 4 grid (mean and standard deviation over the block) plus three global percentiles (35-d). It holds
    no appearance information (no colours, no thumbnail), so it also tests how far motion energy alone gets."""
    neural, dim, batch = False, 35, 16

    def embed_blocks(self, clips: np.ndarray) -> np.ndarray:
        n, t = len(clips), clips.shape[1]
        g = clips.astype(np.float32).mean(-1) / 255.0                              # grey (n, t, 224, 224)
        g = g.reshape(n, t, 56, 4, 56, 4).mean((3, 5))                              # (n, t, 56, 56)
        d = np.abs(np.diff(g, axis=1))                                              # (n, t-1, 56, 56)
        cells = d.reshape(n, t - 1, 4, 14, 4, 14).mean((3, 5)).reshape(n, t - 1, 16)
        q = np.percentile(d.reshape(n, -1), [50, 90, 99], axis=1).T
        f = np.concatenate([cells.mean(1), cells.std(1), q], 1)
        return np.log1p(20.0 * f).astype(np.float32)


# ----------------------------------------------------------------------------- neural
def _processor_stats(model_id: str, default: tuple):
    """Normalisation constants from the model's own preprocessor config, the documented defaults if unavailable."""
    try:
        from transformers import AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(model_id)
        return tuple(p.image_mean), tuple(p.image_std)
    except Exception:  # noqa: BLE001
        return default


class _TorchBlockEncoder(BlockVideoEncoder):
    model_id, norm = "", IMAGENET

    def _setup(self):
        self.device = _device()
        self.mean, self.std = _processor_stats(self.model_id, self.norm)

    def _prep(self, clips: np.ndarray):
        import torch
        x = torch.from_numpy(clips).to(self.device).permute(0, 1, 4, 2, 3).float().div_(255.0)  # (n, t, 3, H, W)
        mean = torch.tensor(self.mean, device=self.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 1, 3, 1, 1)
        return (x - mean) / std


@register("encoder", "videomae_k400")
class VideoMAEK400(_TorchBlockEncoder):
    """VideoMAE ViT-B fine-tuned on Kinetics-400 (human actions), 16 frames per block; mean of the token embeddings."""
    dim, batch, frames, model_id, norm = 768, 8, 16, "MCG-NJU/videomae-base-finetuned-kinetics", IMAGENET

    def load(self):
        if not hasattr(self, "model"):
            from transformers import VideoMAEModel
            self._setup()
            self.model = VideoMAEModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed_blocks(self, clips: np.ndarray) -> np.ndarray:
        import torch
        with torch.inference_mode():
            h = self.model(pixel_values=self._prep(clips)).last_hidden_state
        return h.mean(1).float().cpu().numpy()


@register("encoder", "xclip_b32")
class XCLIPB32(_TorchBlockEncoder):
    """X-CLIP ViT-B/32: CLIP with cross-frame attention, trained on Kinetics-400 with text; 8 frames per block. Its
    video embedding lives in the same space as its text embeddings, which allows text-defined criteria later."""
    dim, batch, frames, model_id, norm = 512, 16, 8, "microsoft/xclip-base-patch32", OPENAI_CLIP

    def load(self):
        if not hasattr(self, "model"):
            from transformers import XCLIPModel
            self._setup()
            self.model = XCLIPModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed_blocks(self, clips: np.ndarray) -> np.ndarray:
        import torch
        with torch.inference_mode():
            f = self.model.get_video_features(pixel_values=self._prep(clips))
        return _tensor_out(f).float().cpu().numpy()
