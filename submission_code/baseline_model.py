"""Baseline model used by the MoCha sample submission -- improved classifier head."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from model.t2m_eval_wrapper import EvaluatorModelWrapper
from submission.preprocess import MotionPreprocessor
from utils.get_opt import get_opt


ROOT = Path(__file__).resolve().parents[1]


class ImprovedClassifier(nn.Module):
    """Improved head: 512->256->128->64->4 with LayerNorm, ordinal binary heads."""
    def __init__(self, input_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.head_b1 = nn.Linear(64, 1)
        self.head_b2 = nn.Linear(64, 1)
        self.head_b3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits only (same interface as baseline MotionClassifier)."""
        h = x
        for layer in self.fc[:-1]:
            h = layer(h)
        return self.fc[-1](h)


def load_config() -> dict[str, Any]:
    with (ROOT / "weights" / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("classifier_checkpoint_path","motion_encoder_checkpoint_path",
                "opt_file_path","checkpoints_dir","normalization_dir","smpl_model_path"):
        cfg[key] = (ROOT / cfg[key]).resolve()
    return cfg


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class Model:
    def __init__(self):
        self.cfg = load_config()
        self.device = choose_device()
        self.preprocess = MotionPreprocessor(
            smpl_model_path=self.cfg["smpl_model_path"],
            normalization_dir=self.cfg["normalization_dir"],
            device=self.device,
            sequence_len=int(self.cfg["sequence_len"]),
            target_fps=self.cfg.get("target_fps"),
            apply_slope_correction=bool(self.cfg.get("apply_slope_correction", False)),
        )
        self.eval_wrapper = self._load_motion_encoder()
        self.classifier = self._load_classifier()

    def _load_motion_encoder(self) -> EvaluatorModelWrapper:
        opt = get_opt(Path(self.cfg["opt_file_path"]), self.device)
        opt.checkpoints_dir = str(Path(self.cfg["checkpoints_dir"]))
        wrapper = EvaluatorModelWrapper(opt)
        state = torch_load(Path(self.cfg["motion_encoder_checkpoint_path"]), self.device)
        wrapper.motion_encoder.load_state_dict(state)
        wrapper.motion_encoder.eval()
        wrapper.movement_encoder.eval()
        return wrapper

    def _load_classifier(self) -> ImprovedClassifier:
        model = ImprovedClassifier(512, int(self.cfg.get("num_classes", 4))).to(self.device)
        model.load_state_dict(torch_load(Path(self.cfg["classifier_checkpoint_path"]), self.device))
        model.eval()
        return model

    def predict(self, sample: Mapping[str, Any]) -> int:
        motion, length = self.preprocess(sample)
        motion_t = torch.as_tensor(motion[None], dtype=torch.float32, device=self.device)
        length_t = torch.as_tensor([length], dtype=torch.long, device=self.device)
        with torch.no_grad():
            embedding = self.eval_wrapper.get_motion_embeddings_ordered(motion_t, length_t)
            logits = self.classifier(embedding)
        return int(torch.argmax(logits, dim=1).item())
