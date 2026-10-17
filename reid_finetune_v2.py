"""
reid_finetune_v2.py -- fine-tune a football ReID model on a hand-checked crop corpus.

Input folder (output of build_reid_corpus.py after manual review):
  reid_corpus/
    player_01/*.jpg  -> class 0
    player_02/*.jpg  -> class 1
    ...
    review_queue/    not used for training
    trash/           not used for training
  Files whose name starts with "_" (e.g. _overview.jpg) are skipped. The
  camera (H / L) is read from the file name ("..._H_..." / "..._L_...").

Model
  - Backbone: OSNet x1_0 from torchreid with ImageNet weights
    (downloaded by torchreid)
  - 4 horizontal stripes (PCB) + 1 global branch, each -> 128-d embedding
    (5 x 128 = 640-d concatenated)
  - SupCon loss on every branch + cross-entropy (label smoothing 0.05) on the
    concatenated embedding
  - conv4, conv5 and the new heads are trained; earlier blocks are frozen
  - P x K identity sampler (default P=8 identities x K=4 crops = 32 per batch)

Augmentation: horizontal flip only (see CamAwareAug).

Evaluation: after every epoch, mean cosine similarity of same-identity pairs
minus that of different-identity pairs ("margin") on up to 500 crops. These
crops come from the SAME corpus the model is trained on, so the margin shows
how well the training identities are separated, not how the model
generalises to new players or matches.

Output
  <out>.pth   state_dict + number of classes (best margin so far)
  <out>.json  class names, metric, architecture info

Usage
  python reid_finetune_v2.py --corpus reid_corpus --epochs 40 --out reid_finetuned_v2.pth
"""
from __future__ import annotations
import argparse, json, os, random, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
from PIL import Image


# ════════════════════════════════════════════════════════════════════════
# 1) DATASET
# ════════════════════════════════════════════════════════════════════════
IMG_SIZE = (256, 128)  # (H, W)


def _cam_from_name(fname: str) -> str:
    """Camera from a file name such as 'P-3_f01234_H_safe.jpg' ('H', 'L' or '?')."""
    if "_H_" in fname:
        return "H"
    if "_L_" in fname:
        return "L"
    return "?"


class CorpusDataset(Dataset):
    def __init__(self, corpus_root: Path, transform=None, return_cam: bool = True):
        self.root = Path(corpus_root)
        self.transform = transform
        self.return_cam = return_cam
        self.samples: List[Tuple[Path, int, str]] = []
        self.class_names: List[str] = []

        player_dirs = sorted([d for d in self.root.iterdir()
                              if d.is_dir() and d.name.startswith("player_")])
        for cid, d in enumerate(player_dirs):
            self.class_names.append(d.name)
            for f in sorted(d.glob("*.jpg")):
                if f.name.startswith("_") or "_overview" in f.name:
                    continue  # contact sheets / helper images
                self.samples.append((f, cid, _cam_from_name(f.name)))

        # Crops per camera and class (plain dicts: a defaultdict with a lambda
        # cannot be pickled for DataLoader workers on Windows)
        self.class_cams: Dict[int, Dict[str, int]] = {}
        for _, cid, cam in self.samples:
            d = self.class_cams.setdefault(cid, {"H": 0, "L": 0, "?": 0})
            d[cam] = d.get(cam, 0) + 1

        # Classes seen by only one camera (reported in the log)
        self.is_single_cam_cls: Dict[int, str] = {}
        for cid, c in self.class_cams.items():
            if c.get("H", 0) == 0 and c.get("L", 0) > 0:
                self.is_single_cam_cls[cid] = "L_only"
            elif c.get("L", 0) == 0 and c.get("H", 0) > 0:
                self.is_single_cam_cls[cid] = "H_only"

        print(f"[Corpus] {len(self.class_names)} class, {len(self.samples)} sample")
        for cid, name in enumerate(self.class_names):
            c = self.class_cams[cid]
            flag = f" [{self.is_single_cam_cls.get(cid)}]" if cid in self.is_single_cam_cls else ""
            print(f"  {name}: H={c.get('H',0)} L={c.get('L',0)}{flag}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cid, cam = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        single_cam = cid in self.is_single_cam_cls
        if self.transform is not None:
            img = self.transform(img, cam, single_cam)
        else:
            img = T.ToTensor()(img)
        return img, cid, cam


# ════════════════════════════════════════════════════════════════════════
# 2) MINIMAL AUGMENTATION
# ════════════════════════════════════════════════════════════════════════
# Within one match the lighting, cameras and kits are fixed. Colour jitter
# blurred the colour cues that separate identities, random erasing duplicated
# occlusion that real footage already has, and blur as a fake "other camera"
# added no real cross-camera signal. Only horizontal flip is kept.
# (The cam / single_cam arguments are accepted for this interface but unused.)
class CamAwareAug:
    """Horizontal flip (train only) + ImageNet normalisation."""
    def __init__(self, train: bool = True):
        self.train = train
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img: Image.Image, cam: str, single_cam: bool) -> torch.Tensor:
        if self.train and random.random() < 0.5:
            img = T.functional.hflip(img)
        x = self.to_tensor(img)
        x = self.normalize(x)
        return x


# ════════════════════════════════════════════════════════════════════════
# 3) PK SAMPLER
# ════════════════════════════════════════════════════════════════════════
class PKSampler(Sampler):
    """Each batch: P identities x K crops. Classes with fewer than K crops are
    sampled with replacement."""
    def __init__(self, dataset: CorpusDataset, P: int = 8, K: int = 4, iters_per_epoch: int = 100):
        self.P = P; self.K = K; self.iters = iters_per_epoch
        self.cls_to_idx: Dict[int, List[int]] = defaultdict(list)
        for i, (_, cid, _) in enumerate(dataset.samples):
            self.cls_to_idx[cid].append(i)
        self.classes = list(self.cls_to_idx.keys())

    def __iter__(self):
        for _ in range(self.iters):
            chosen = random.sample(self.classes, min(self.P, len(self.classes)))
            batch = []
            for c in chosen:
                pool = self.cls_to_idx[c]
                if len(pool) >= self.K:
                    batch.extend(random.sample(pool, self.K))
                else:
                    batch.extend(random.choices(pool, k=self.K))
            yield batch

    def __len__(self): return self.iters


# ════════════════════════════════════════════════════════════════════════
# 4) MODEL: OSNet + PCB 4-stripe + global head
# ════════════════════════════════════════════════════════════════════════
class MultiPartReID(nn.Module):
    """OSNet backbone + 4 spatial stripe + global → 5 × 128-d = 640-d."""
    def __init__(self, num_classes: int, part_dim: int = 128):
        super().__init__()
        try:
            import torchreid
            self.backbone = torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, pretrained=True, loss="softmax")
            # Only the feature extractor is needed
            self.backbone.classifier = nn.Identity()
            # OSNet x1_0: 512 channels after the last conv block
            self.feat_ch = 512
        except Exception as e:
            raise RuntimeError(f"could not load torchreid / OSNet: {e}")

        # Use the feature map before global pooling
        self._patch_backbone_return_featmap()

        self.part_dim = part_dim
        self.num_parts = 5  # 4 stripes (roughly head, upper, lower, feet) + global
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feat_ch, part_dim),
                nn.BatchNorm1d(part_dim),
            ) for _ in range(self.num_parts)
        ])
        # Classifier on the concatenated embedding
        self.classifier = nn.Linear(part_dim * self.num_parts, num_classes)

        # Freeze conv1-conv3; train conv4, conv5 (and the heads above)
        self._freeze_early_layers()

    def _patch_backbone_return_featmap(self):
        """Use OSNet.featuremaps (B, 512, h, w) instead of forward()."""
        self._featmap_fn = self.backbone.featuremaps  # (B, 512, h, w)

    def _freeze_early_layers(self):
        names_to_train = ["conv4", "conv5", "classifier", "fc"]
        for n, p in self.backbone.named_parameters():
            if any(nn_ in n for nn_ in names_to_train):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        fm = self._featmap_fn(x)  # (B, 512, Hf, Wf), Hf=16 Wf=8 for 256x128 input
        B, C, Hf, Wf = fm.shape
        # Stripe rows for Hf=16: 0-3 (head), 3-7 (upper body), 7-12 (lower
        # body), 12-16 (feet); scaled proportionally for other heights
        bounds = [int(round(r * Hf / 16)) for r in [0, 3, 7, 12, 16]]
        bounds = sorted(set([max(0, min(Hf, b)) for b in bounds]))
        # Parts: [0,b1], [b1,b2], [b2,b3], [b3,Hf], global
        stripes = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i+1]
            if b <= a: b = a + 1
            s = fm[:, :, a:b, :].mean(dim=[2, 3])  # (B, C)
            stripes.append(s)
        # Very small feature maps: pad with the global mean
        while len(stripes) < 4:
            stripes.append(fm.mean(dim=[2, 3]))
        stripes = stripes[:4]
        g = fm.mean(dim=[2, 3])
        parts = stripes + [g]  # 5 × (B, C)

        embs = []
        for i, p in enumerate(parts):
            e = self.heads[i](p)
            e = F.normalize(e, dim=1)
            embs.append(e)
        concat = torch.cat(embs, dim=1)  # (B, 5*part_dim)
        logits = self.classifier(concat * (1.0 / (self.num_parts ** 0.5)))
        return embs, concat, logits


# ════════════════════════════════════════════════════════════════════════
# 5) SupCon LOSS
# ════════════════════════════════════════════════════════════════════════
def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """features: (B, D) L2-normalized. labels: (B,)."""
    device = features.device
    B = features.size(0)
    sim = features @ features.t() / temp
    # numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    # exclude self-pairs
    mask = mask - torch.eye(B, device=device)
    mask = mask.clamp(min=0.0)
    exp_logits = torch.exp(logits) * (1 - torch.eye(B, device=device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mask_sum = mask.sum(dim=1)
    valid = mask_sum > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / mask_sum[valid]
    return -mean_log_prob_pos.mean()


# ════════════════════════════════════════════════════════════════════════
# 6) TRAIN LOOP
# ════════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Dataset
    ds_train = CorpusDataset(args.corpus, transform=CamAwareAug(train=True))
    ds_val = CorpusDataset(args.corpus, transform=CamAwareAug(train=False))
    num_classes = len(ds_train.class_names)
    sampler = PKSampler(ds_train, P=args.P, K=args.K, iters_per_epoch=args.iters_per_epoch)
    # Windows: worker processes cause pickling problems, so --workers is
    # ignored there and data is loaded in the main process
    _nw = 0 if os.name == "nt" else args.workers
    loader = DataLoader(ds_train, batch_sampler=sampler, num_workers=_nw, pin_memory=True)

    # Model
    model = MultiPartReID(num_classes=num_classes, part_dim=args.part_dim).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] trainable={n_trainable/1e6:.2f}M / total={n_total/1e6:.2f}M")

    # Optim
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_margin = -1.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot_sup, tot_ce, tot_loss, n_iter = 0.0, 0.0, 0.0, 0
        for imgs, labels, cams in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embs, concat, logits = model(imgs)
            # SupCon on every branch
            loss_sup = 0.0
            for e in embs:
                loss_sup = loss_sup + supcon_loss(e, labels, temp=0.1)
            loss_sup = loss_sup / len(embs)
            loss_ce = F.cross_entropy(logits, labels, label_smoothing=0.05)
            loss = loss_sup + args.ce_weight * loss_ce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot_sup += float(loss_sup.item()); tot_ce += float(loss_ce.item())
            tot_loss += float(loss.item()); n_iter += 1
        sched.step()
        dt = time.time() - t0

        # Quick check on (up to 500) crops of the TRAINING corpus:
        # same-identity vs different-identity similarity. Not a held-out score.
        metric = eval_margin(model, ds_val, device, max_samples=500)
        print(f"[Ep {ep+1:02d}/{args.epochs}] loss={tot_loss/n_iter:.3f} "
              f"sup={tot_sup/n_iter:.3f} ce={tot_ce/n_iter:.3f} "
              f"self_sim={metric['self']:.3f} cross_sim={metric['cross']:.3f} "
              f"margin={metric['margin']:.3f} dt={dt:.1f}s lr={opt.param_groups[0]['lr']:.2e}")

        if metric["margin"] > best_margin:
            best_margin = metric["margin"]
            save_ckpt(model, ds_train.class_names, args.out, metric)
            print(f"  [BEST] new best margin={best_margin:.3f} -> {args.out}")


@torch.no_grad()
def eval_margin(model: MultiPartReID, ds: CorpusDataset, device, max_samples: int = 500):
    """Mean cosine similarity of same-class pairs ("self") and of different-class
    pairs ("cross") on a random subset of `ds`; margin = self - cross."""
    model.eval()
    # Random subset
    if len(ds) > max_samples:
        idxs = random.sample(range(len(ds)), max_samples)
    else:
        idxs = list(range(len(ds)))
    feats, labels = [], []
    BS = 64
    buf_imgs, buf_labels = [], []
    def flush():
        nonlocal feats, labels, buf_imgs, buf_labels
        if not buf_imgs: return
        x = torch.stack(buf_imgs).to(device)
        _, concat, _ = model(x)
        concat = F.normalize(concat, dim=1)
        feats.append(concat.cpu()); labels.extend(buf_labels)
        buf_imgs.clear(); buf_labels.clear()
    for i in idxs:
        img, cid, _ = ds[i]
        buf_imgs.append(img); buf_labels.append(cid)
        if len(buf_imgs) >= BS: flush()
    flush()
    if not feats:
        return {"self": 0.0, "cross": 0.0, "margin": 0.0}
    F_ = torch.cat(feats, dim=0)
    L_ = torch.tensor(labels)
    S = F_ @ F_.t()
    eye = torch.eye(S.size(0)).bool()
    same = (L_.unsqueeze(0) == L_.unsqueeze(1)) & (~eye)
    diff = (L_.unsqueeze(0) != L_.unsqueeze(1))
    self_sim = S[same].mean().item() if same.any() else 0.0
    cross_sim = S[diff].mean().item() if diff.any() else 0.0
    return {"self": self_sim, "cross": cross_sim, "margin": self_sim - cross_sim}


def save_ckpt(model: MultiPartReID, class_names: List[str], out_path: str, metric: dict):
    torch.save({"state_dict": model.state_dict(),
                "num_classes": len(class_names)}, out_path)
    meta = {"class_names": class_names, "metric": metric,
            "arch": "osnet_x1_0+pcb4+global", "part_dim": 128, "concat_dim": 640}
    Path(out_path).with_suffix(".json").write_text(json.dumps(meta, indent=2))


# ════════════════════════════════════════════════════════════════════════
# 7) MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fine-tune OSNet + PCB part heads on a ReID crop corpus.")
    ap.add_argument("--corpus", default="reid_corpus", type=str,
                    help="corpus folder with player_*/ subfolders (default: reid_corpus)")
    ap.add_argument("--out", default="reid_finetuned_v2.pth", type=str,
                    help="checkpoint path; metadata goes next to it as .json")
    ap.add_argument("--epochs", default=40, type=int, help="default: 40")
    ap.add_argument("--P", default=8, type=int, help="identities per batch (default: 8)")
    ap.add_argument("--K", default=4, type=int, help="crops per identity (default: 4)")
    ap.add_argument("--iters_per_epoch", default=80, type=int, help="default: 80")
    ap.add_argument("--lr", default=3e-4, type=float, help="AdamW learning rate (default: 3e-4)")
    ap.add_argument("--part_dim", default=128, type=int,
                    help="embedding size per branch (default: 128)")
    ap.add_argument("--ce_weight", default=0.5, type=float,
                    help="weight of the cross-entropy term (default: 0.5)")
    ap.add_argument("--workers", default=0, type=int,
                    help="DataLoader workers; ignored on Windows (default: 0)")
    ap.add_argument("--seed", default=42, type=int, help="default: 42")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    train(args)
