"""Embedding module (PLAN §5): one vector per window. Every encoder is a Strategy registered by name."""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import soundfile as sf

from .core import register
from .data import SR, _h


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _tensor_out(x):
    """transformers may return a tensor or an output object from get_*_features."""
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state.mean(1)
    return x


class _AudioCache:
    def __init__(self, maxsize: int = 8):
        self.d: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize

    def segment(self, path: str, t0: float, t1: float) -> np.ndarray:
        if path not in self.d:
            y, _ = sf.read(path, dtype="float32")
            self.d[path] = y if y.ndim == 1 else y[:, 0]
            if len(self.d) > self.maxsize:
                self.d.popitem(last=False)
        y = self.d[path]
        a, b = int(t0 * SR), int(t1 * SR)
        seg = y[a:b]
        return np.pad(seg, (0, max(0, (b - a) - len(seg))))


class AudioEncoder:
    modality, neural, dim, batch = "audio", True, 0, 16

    def fingerprint(self) -> str:
        return f"{self.registry_name}-v1"

    def load(self):
        pass

    def embed(self, ys: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, df: pd.DataFrame, budget_s: float | None = None, log=None) -> np.ndarray:
        """(n, dim) embeddings. Files are visited in a fixed pseudo-random order; if the budget runs out, the windows
        done so far cover a random subset of files, the rest stay NaN, and `self.coverage` records the share done."""
        self.load()
        cache, t_start = _AudioCache(), time.time()
        file_key = df.audio_path.map(lambda p: _h(p, "encode-order")).values
        order = np.lexsort((df.t0.values, file_key))
        out = np.full((len(df), self.dim), np.nan, np.float32)
        self.coverage = 1.0
        for i in range(0, len(order), self.batch):
            if budget_s and time.time() - t_start > budget_s:
                self.coverage = i / len(order)
                if log:
                    log.warning(f"  {self.registry_name}: budget of {budget_s:.0f}s used up at {i}/{len(order)} windows")
                break
            idx = order[i:i + self.batch]
            ys = [cache.segment(df.audio_path.values[j], df.t0.values[j], df.t1.values[j]) for j in idx]
            out[idx] = self.embed(ys)
            if log and (i // self.batch) % 200 == 0:
                log.info(f"  {self.registry_name}: {i}/{len(order)} windows")
        return out


@register("encoder", "mfcc_stats")
class MFCCStats(AudioEncoder):
    neural, dim, batch = False, 84, 64

    def embed(self, ys):
        import librosa
        feats = []
        for y in ys:
            m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
            rms = librosa.feature.rms(y=y)[0]
            flux = librosa.onset.onset_strength(y=y, sr=SR)
            feats.append(np.concatenate([m.mean(1), m.std(1), [rms.mean(), rms.std(), flux.mean(), flux.std()]]))
        return np.asarray(feats, np.float32)


@register("encoder", "vggish")
class VGGish(AudioEncoder):
    dim, batch = 128, 32

    def load(self):
        if not hasattr(self, "model"):
            import torch
            self.model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True, verbose=False)
            self.model.eval()

    def embed(self, ys):
        import torch
        out = []
        with torch.inference_mode():
            for y in ys:
                e = self.model.forward(y, SR)
                e = e.detach().cpu().numpy().astype(np.float32)
                out.append(e.reshape(-1, 128).mean(0))
        return np.asarray(out, np.float32)


class _HFAudio(AudioEncoder):
    model_id = ""

    def _inputs(self, ys):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError

    def embed(self, ys):
        import torch
        with torch.inference_mode():
            inputs = {k: v.to(self.device) for k, v in self._inputs(ys).items()}
            return self._forward(inputs).float().cpu().numpy()


@register("encoder", "ast")
class AST(_HFAudio):
    dim, model_id = 768, "MIT/ast-finetuned-audioset-10-10-0.4593"

    def load(self):
        if not hasattr(self, "model"):
            from transformers import ASTFeatureExtractor, ASTModel
            self.device = _device()
            self.fe = ASTFeatureExtractor.from_pretrained(self.model_id)
            self.model = ASTModel.from_pretrained(self.model_id).to(self.device).eval()

    def _inputs(self, ys):
        return self.fe(ys, sampling_rate=SR, return_tensors="pt")

    def _forward(self, inputs):
        return _tensor_out(self.model(**inputs))


@register("encoder", "clap")
class CLAP(_HFAudio):
    dim, model_id = 512, "laion/clap-htsat-unfused"

    def load(self):
        if not hasattr(self, "model"):
            from transformers import ClapFeatureExtractor, ClapModel
            self.device = _device()
            self.fe = ClapFeatureExtractor.from_pretrained(self.model_id)
            self.model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()

    def _inputs(self, ys):
        import librosa
        ys48 = [librosa.resample(y, orig_sr=SR, target_sr=48000) for y in ys]
        return self.fe(ys48, sampling_rate=48000, return_tensors="pt")

    def _forward(self, inputs):
        return _tensor_out(self.model.get_audio_features(**inputs))


@register("encoder", "hubert")
class HuBERT(_HFAudio):
    dim, model_id, layer = 768, "facebook/hubert-base-ls960", 9

    def load(self):
        if not hasattr(self, "model"):
            from transformers import AutoFeatureExtractor, HubertModel
            self.device = _device()
            self.fe = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = HubertModel.from_pretrained(self.model_id).to(self.device).eval()

    def fingerprint(self):
        return f"hubert-L{self.layer}-v1"

    def _inputs(self, ys):
        x = self.fe(ys, sampling_rate=SR, return_tensors="pt", padding=True)
        return {"input_values": x["input_values"]}

    def _forward(self, inputs):
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer].mean(1)


@register("encoder", "clip_frames")
class CLIPFrames:
    """Video: CLIP ViT-B/16 on keyframes (decoded with PyAV), averaged within each window."""
    modality, neural, dim, model_id = "video", True, 512, "openai/clip-vit-base-patch16"

    def fingerprint(self):
        return "clip_frames-keyframes-v1"

    def load(self):
        if not hasattr(self, "model"):
            from transformers import CLIPImageProcessor, CLIPModel
            self.device = _device()
            self.proc = CLIPImageProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()

    def _keyframes(self, path: str, max_frames: int = 600):
        import av
        times, imgs = [], []
        with av.open(path) as c:
            st = c.streams.video[0]
            st.codec_context.skip_frame = "NONKEY"
            for fr in c.decode(st):
                if fr.time is None:
                    continue
                times.append(float(fr.time))
                imgs.append(fr.to_image().convert("RGB").resize((224, 224)))
                if len(imgs) >= max_frames:
                    break
        return np.asarray(times), imgs

    def encode(self, df: pd.DataFrame, budget_s: float | None = None, log=None) -> np.ndarray:
        """Videos are visited in a fixed pseudo-random order; on a budget overrun the rest stay NaN (see AudioEncoder)."""
        import torch
        self.load()
        t_start = time.time()
        out = np.full((len(df), self.dim), np.nan, np.float32)
        groups = sorted(df.groupby("video_path", sort=False), key=lambda g: _h(g[0], "encode-order"))
        self.coverage = 1.0
        for k, (vid, sub) in enumerate(groups):
            if budget_s and time.time() - t_start > budget_s:
                self.coverage = k / len(groups)
                if log:
                    log.warning(f"  clip_frames: budget of {budget_s:.0f}s used up after {k}/{len(groups)} videos")
                break
            try:
                times, imgs = self._keyframes(vid)
            except Exception as e:  # noqa: BLE001  (one unreadable video must not stop the encoder)
                if log:
                    log.warning(f"  clip_frames: cannot decode {vid}: {e}")
                continue
            if not imgs:
                continue
            embs = []
            with torch.inference_mode():
                for i in range(0, len(imgs), 32):
                    x = self.proc(images=imgs[i:i + 32], return_tensors="pt")["pixel_values"].to(self.device)
                    embs.append(_tensor_out(self.model.get_image_features(pixel_values=x)).float().cpu().numpy())
            embs = np.concatenate(embs)
            for j, r in zip(sub.index, sub.itertuples()):
                m = (times >= r.t0) & (times < r.t1)
                out[df.index.get_loc(j)] = embs[m].mean(0) if m.any() else embs[np.argmin(np.abs(times - (r.t0 + r.t1) / 2))]
            if log and k % 25 == 0:
                log.info(f"  clip_frames: {k}/{len(groups)} videos")
        return out
