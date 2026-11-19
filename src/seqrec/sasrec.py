"""SASRec-style self-attentive sequential recommender (Kang & McAuley, 2018) in PyTorch.

Differences from the original paper, stated plainly: pre-LayerNorm transformer blocks from
``torch.nn`` and full softmax cross-entropy over the catalogue instead of one sampled negative
per position. Item embeddings are shared between input and output.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class SASRecConfig:
    max_len: int = 100
    dim: int = 64
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    lr: float = 2e-3
    batch_size: int = 128
    epochs: int = 120
    patience: int = 3
    eval_every: int = 2
    seed: int = 0
    threads: int | None = None


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(n_items + 1, cfg.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, dim_feedforward=cfg.dim,
                                       dropout=cfg.dropout, batch_first=True, norm_first=True)
            for _ in range(cfg.n_blocks)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        causal = torch.triu(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)
        # small init keeps the shared input/output embedding logits near zero at the start
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) left-padded item ids -> hidden states (B, L, dim)."""
        length = seq.shape[1]
        keep = (seq > 0).unsqueeze(-1).float()
        pos = torch.arange(length, device=seq.device).unsqueeze(0)
        h = self.item_emb(seq) * (self.cfg.dim ** 0.5) + self.pos_emb(pos)
        h = self.drop(h) * keep
        mask = self.causal[:length, :length]
        for blk in self.blocks:
            h = blk(h, src_mask=mask) * keep
        return self.norm(h)


def _left_pad(seqs: list[list[int]], length: int) -> np.ndarray:
    out = np.zeros((len(seqs), length), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-length:]
        if s:
            out[i, length - len(s):] = s
    return out


class SASRec:
    name = "SASRec"

    def __init__(self, cfg: SASRecConfig | None = None):
        self.cfg = cfg or SASRecConfig()
        self.history_: list[dict] = []

    def fit(self, histories: list[list[int]], n_items: int,
            val_fn: Callable[[SASRec], float] | None = None, verbose: bool = False) -> SASRec:
        """Train on next-item prediction at every position of each user's training sequence.

        ``val_fn(model) -> NDCG@10`` drives early stopping; the best checkpoint is restored.
        """
        cfg = self.cfg
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.n_items = n_items
        self.net = SASRecNet(n_items, cfg)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
        usable = [h for h in histories if len(h) >= 2]
        seqs = _left_pad([h[:-1] for h in usable], cfg.max_len)
        tgts = _left_pad([h[1:] for h in usable], cfg.max_len)
        best, best_state, bad = -1.0, None, 0
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            order = rng.permutation(len(seqs))
            total = 0.0
            for i in range(0, len(order), cfg.batch_size):
                b = order[i:i + cfg.batch_size]
                x = torch.from_numpy(seqs[b])
                y = torch.from_numpy(tgts[b])
                h = self.net(x)
                valid = y > 0
                logits = h[valid] @ self.net.item_emb.weight.T
                loss = nn.functional.cross_entropy(logits, y[valid])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            rec = {"epoch": epoch, "loss": total / len(seqs), "seconds": time.time() - t0}
            if val_fn is not None and epoch % cfg.eval_every == 0:
                score = float(val_fn(self))
                rec["val_ndcg10"] = score
                if score > best:
                    best, bad = score, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            self.history_.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_fn is not None and bad >= cfg.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_ = best
        self.train_seconds_ = time.time() - t0
        return self

    @torch.no_grad()
    def score(self, histories: list[list[int]], batch_size: int = 512) -> np.ndarray:
        self.net.eval()
        out = np.empty((len(histories), self.n_items + 1), dtype=np.float32)
        emb = self.net.item_emb.weight
        for i in range(0, len(histories), batch_size):
            x = torch.from_numpy(_left_pad(histories[i:i + batch_size], self.cfg.max_len))
            h = self.net(x)[:, -1, :]
            out[i:i + batch_size] = (h @ emb.T).numpy()
        return out
