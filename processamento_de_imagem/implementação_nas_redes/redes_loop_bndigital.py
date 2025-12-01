#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treino em lote (várias situações) de uma rede siamesa para verificação de autoria.

- Encontra trincas *_train/_val/_test.txt em --pairs_root (recursivo)
- Um treino por situação (run_slug = caminho relativo do trio)
- Validação por ROC-AUC + Early Stopping
- Saída por run: metrics_epoch.txt/.csv, best_summary.txt, test_results.txt,
  checkpoints/best_epoch...pt, best_model.paeckle
- Agregados globais: _all_runs_metrics.csv e _all_runs_summary.csv
- Resolutor de caminhos robusto p/ não duplicar prefixo de images_root
- Filtragem de pares com arquivos ausentes
- Logs no stdout e em results_root/_run_log.txt

Requisitos: torch, torchvision (opcional), numpy, pillow, scikit-learn, tqdm
"""

from __future__ import annotations
import argparse, os, sys, csv, pickle, random, traceback, time, socket
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, average_precision_score, roc_auc_score
)
from tqdm.auto import tqdm


# =========================
# Config / CLI
# =========================
@dataclass
class Config:
    # caminhos
    pairs_root:   str = "pairs_out"      # padrão robusto
    images_root:  str = ""               # vazio = não prefixa; ideal se .txt já vem com preprocess_out/...
    results_root: str = "results_runs"

    # treino
    epochs:       int   = 50
    batch_size:   int   = 16
    lr:           float = 2e-5
    weight_decay: float = 0.0
    num_workers:  int   = 2
    pin_memory:   bool  = True

    # early stopping (por ROC-AUC de validação)
    es_patience:   int   = 30
    es_min_delta:  float = 1e-3
    es_min_epochs: int   = 5

    # reprodutibilidade
    seed: int = 42

    # amostragem opcional por split (None = usa tudo)
    max_samples_train: Optional[int] = None
    max_samples_val:   Optional[int] = None
    max_samples_test:  Optional[int] = None

    # filtros por nome do run (minúsculo). vazio = pega tudo
    include: List[str] = None
    exclude: List[str] = None

    # comportamento/infra
    no_tqdm: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Treino siamesa (HPC) sobre vários arquivos de pares.")
    p.add_argument("--pairs_root",   type=str, default="pairs_out",
                   help="Pasta raiz com subpastas contendo *_train/_val/_test.txt (default: pairs_out)")
    p.add_argument("--images_root",  type=str, default="",
                   help="Prefixo para caminhos relativos nos .txt ('' = não prefixa)")
    p.add_argument("--results_root", type=str, default="results_runs",
                   help="Pasta de saída para resultados (default: results_runs)")

    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--no_pin_memory", action="store_true")

    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--es_patience",  type=int, default=30)
    p.add_argument("--es_min_delta", type=float, default=1e-3)
    p.add_argument("--es_min_epochs",type=int, default=5)

    p.add_argument("--max_samples_train", type=int, default=None)
    p.add_argument("--max_samples_val",   type=int, default=None)
    p.add_argument("--max_samples_test",  type=int, default=None)

    p.add_argument("--include", nargs="*", default=[], help="Substrings para incluir (minúsculas)")
    p.add_argument("--exclude", nargs="*", default=[], help="Substrings para excluir (minúsculas)")

    p.add_argument("--no_tqdm", action="store_true", help="Desativa barras de progresso (ambientes sem TTY).")

    a = p.parse_args()
    return Config(
        pairs_root=a.pairs_root,
        images_root=a.images_root,
        results_root=a.results_root,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        num_workers=a.num_workers,
        pin_memory=not a.no_pin_memory,
        es_patience=a.es_patience,
        es_min_delta=a.es_min_delta,
        es_min_epochs=a.es_min_epochs,
        seed=a.seed,
        max_samples_train=a.max_samples_train,
        max_samples_val=a.max_samples_val,
        max_samples_test=a.max_samples_test,
        include=[s.lower() for s in (a.include or [])],
        exclude=[s.lower() for s in (a.exclude or [])],
        no_tqdm=a.no_tqdm,
    )


# =========================
# Logging simples (arquivo + stdout)
# =========================
class TeeLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.fh = log_file.open("a", encoding="utf-8")
    def write(self, msg: str):
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
        self.fh.write(msg)
        self.fh.flush()
    def flush(self):  # compat
        sys.__stdout__.flush()
        self.fh.flush()
    def close(self):
        self.fh.close()


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv_header_if_needed(path: Path, header: List[str]):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def write_csv_row(path: Path, row: List):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def find_triplets(pairs_root: Path, include: List[str], exclude: List[str]) -> List[Tuple[Path, Path, Path, str]]:
    """
    Procura recursivamente por *_train.txt e retorna trincas (_train, _val, _test, base_slug).
    base_slug = caminho relativo (a pairs_root) sem sufixo _train/_val/_test e sem .txt.
    """
    trips = []
    for tr in sorted(pairs_root.rglob("*_train.txt")):
        base = str(tr)[:-10]
        va = Path(base + "_val.txt")
        te = Path(base + "_test.txt")
        if va.exists() and te.exists():
            base_rel = str(Path(os.path.relpath(base, pairs_root))).replace("\\", "/")
            name = base_rel.lower()
            if include and not any(s in name for s in include):
                continue
            if exclude and any(s in name for s in exclude):
                continue
            trips.append((tr, va, te, base_rel))
    return trips


# =========================
# Caminho robusto de imagem
# =========================
def resolve_image_path(p: str, images_root: Path) -> Path:
    """
    Resolve p evitando prefixo duplicado e sendo tolerante a separadores.
    Regras:
      - Se p é absoluto e existe, usa p.
      - Se p (normalizado) já começa com images_root (normalizado), tenta como está.
      - Senão, tenta images_root / p; por fim, tenta p como veio.
    """
    p_in = str(p).replace("\\", "/").lstrip("./")
    root = str(images_root).replace("\\", "/").rstrip("/")
    P_abs = Path(p_in)
    try:
        if P_abs.is_absolute() and P_abs.exists():
            return P_abs
    except Exception:
        pass

    if root and (p_in == root or p_in.startswith(root + "/")):
        cand = Path(p_in)
        if cand.exists():
            return cand
        tail = p_in[len(root):].lstrip("/")
        cand2 = Path(root) / Path(tail)
        if cand2.exists():
            return cand2
        return cand

    if str(images_root) != "":
        cand = images_root / Path(p_in)
        if cand.exists():
            return cand

    return Path(p_in)


# =========================
# Dataset
# =========================
def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t.unsqueeze(0)  # [1,H,W]


class SimplePairTxtDataset(Dataset):
    """Lê pares 'path1 path2 label' e só converte para tensor (0..1). Assume HxW já equalizados."""
    def __init__(self, txt_path: Path, images_root: Path, drop_missing: bool = True):
        self.root = images_root
        raw = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    raw.append((parts[0], parts[1], int(parts[2])))

        if drop_missing:
            kept = []
            miss = 0
            for p1, p2, y in raw:
                P1 = resolve_image_path(p1, self.root)
                P2 = resolve_image_path(p2, self.root)
                if P1.exists() and P2.exists():
                    kept.append((p1, p2, y))
                else:
                    miss += 1
            if miss:
                print(f"[AVISO] {txt_path.name}: {miss} pares removidos por arquivos ausentes.", flush=True)
            self.samples = kept
        else:
            self.samples = raw

    def __len__(self): return len(self.samples)

    def _open(self, p: str) -> Image.Image:
        P = resolve_image_path(p, self.root)
        return Image.open(P).convert("L")

    def __getitem__(self, idx):
        p1, p2, y = self.samples[idx]
        im1 = to_tensor(self._open(p1))
        im2 = to_tensor(self._open(p2))
        return im1, im2, torch.tensor(y, dtype=torch.long)


def make_loader(txt_path: Path, images_root: Path, batch_size: int, shuffle: bool,
                num_workers: int, pin_memory: bool, max_keep: Optional[int]):
    ds = SimplePairTxtDataset(txt_path, images_root, drop_missing=True)
    if len(ds) == 0:
        return None, 0
    if (max_keep is not None) and (len(ds) > max_keep):
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        idxs = idxs[:max_keep]
        ds = Subset(ds, idxs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, (len(ds) if isinstance(ds, Subset) else len(ds))


# =========================
# Modelo
# =========================
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class TinyResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class TinyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self._fc_out = num_classes
        self.fc = None  # definido dinamicamente

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride*2),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_fc_if_needed(self, x):
        if self.fc is None:
            flat = x.view(x.size(0), -1).shape[1]
            self.fc = nn.Linear(flat, self._fc_out).to(x.device)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        self._init_fc_if_needed(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResnetSiamese(nn.Module):
    def __init__(self, resnet_layers=(1,1,1,1), resnet_outsize=10, out_layers=2):
        super().__init__()
        self.tinyresnet = TinyResNet(TinyResNetBlock, resnet_layers, num_classes=resnet_outsize)
        self.fc1 = nn.Linear(resnet_outsize * 4, out_layers)
        self.softmax = nn.Softmax(dim=1)

    def forward_once(self, x):
        return self.tinyresnet(x)

    def forward(self, x, y):
        fx, fy = self.forward_once(x), self.forward_once(y)
        sqdiff  = (fx - fy) ** 2
        hadam   = fx * fy
        z = torch.cat([fx, fy, sqdiff, hadam], dim=1)
        z = self.fc1(z)
        return self.softmax(z)


# =========================
# Métricas
# =========================
def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    idx = np.where(fpr <= target)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0

def _eer_from_roc(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
        return float("nan"), float("nan")
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(max(fpr[i], fnr[i]))
    thr = float(thresholds[i])
    return eer, thr

def compute_all_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    out = {
        "roc_auc": float("nan"), "pr_auc": float("nan"),
        "acc": float("nan"), "f1": float("nan"),
        "tpr@0.01": float("nan"), "tpr@0.05": float("nan"), "tpr@0.10": float("nan"),
        "eer": float("nan"), "eer_thr": float("nan"),
        "cm00": "", "cm01": "", "cm10": "", "cm11": "",
    }
    if y_true.size == 0:
        return out
    # ROC related
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p_pos))
        fpr, tpr, thr = roc_curve(y_true, p_pos)
        out["tpr@0.01"] = _tpr_at_fpr(fpr, tpr, 0.01)
        out["tpr@0.05"] = _tpr_at_fpr(fpr, tpr, 0.05)
        out["tpr@0.10"] = _tpr_at_fpr(fpr, tpr, 0.10)
        eer, eer_thr = _eer_from_roc(fpr, tpr, thr)
        out["eer"], out["eer_thr"] = eer, eer_thr
    except Exception:
        pass
    # PR-AUC
    try:
        out["pr_auc"] = float(average_precision_score(y_true, p_pos))
    except Exception:
        pass
    # threshold 0.5 para relatórios
    y_pred = (p_pos >= 0.5).astype(np.int64)
    try:
        out["acc"] = float(accuracy_score(y_true, y_pred))
        out["f1"]  = float(f1_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2,2):
            out["cm00"] = int(cm[0,0]); out["cm01"] = int(cm[0,1])
            out["cm10"] = int(cm[1,0]); out["cm11"] = int(cm[1,1])
    except Exception:
        pass
    return out


# =========================
# Treino/Val/Test helpers
# =========================
def run_train_epoch(model, loader, device, optimizer, scaler, criterion, amp_enabled, desc="train", use_tqdm=True):
    model.train()
    running = 0.0
    iterator = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader
    for X1, X2, Y in iterator:
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                P = model(X1, X2); loss = criterion(P, Y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            P = model(X1, X2); loss = criterion(P, Y)
            loss.backward(); optimizer.step()
        running += loss.item() * X1.size(0)
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(1, len(loader.dataset))

def run_eval_collect(model, loader, device, amp_enabled, desc="eval", use_tqdm=True):
    model.eval()
    ys, ps = [], []
    iterator = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader
    with torch.no_grad():
        ctx = torch.amp.autocast(device_type='cuda', enabled=amp_enabled) if amp_enabled else None
        if ctx is None:
            for X1, X2, Y in iterator:
                X1, X2 = X1.to(device), X2.to(device)
                P = model(X1, X2)
                ys.append(Y.numpy()); ps.append(P[:,1].detach().cpu().numpy())
        else:
            with ctx:
                for X1, X2, Y in iterator:
                    X1, X2 = X1.to(device), X2.to(device)
                    P = model(X1, X2)
                    ys.append(Y.numpy()); ps.append(P[:,1].detach().cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    return y, p


# =========================
# Loop de um run
# =========================
def train_eval_one_run(cfg: Config,
                       train_txt: Path, val_txt: Path, test_txt: Path,
                       run_slug: str,
                       device, amp_enabled: bool,
                       global_metrics_csv: Path, global_summary_csv: Path,
                       use_tqdm: bool):

    images_root = Path(cfg.images_root)
    run_dir = Path(cfg.results_root) / run_slug
    ensure_dir(run_dir); ensure_dir(run_dir / "checkpoints")

    # DataLoaders (com amostragem opcional por split)
    tr_loader, ntr = make_loader(
        train_txt, images_root, cfg.batch_size, True, cfg.num_workers, cfg.pin_memory, cfg.max_samples_train
    )
    va_loader, nva = make_loader(
        val_txt, images_root, 64, False, cfg.num_workers, cfg.pin_memory, cfg.max_samples_val
    )
    te_loader, nte = make_loader(
        test_txt, images_root, 64, False, cfg.num_workers, cfg.pin_memory, cfg.max_samples_test
    )
    if (tr_loader is None) or (va_loader is None) or (te_loader is None):
        print(f"[SKIP] {run_slug}: split vazio/ inválido (ntr={ntr}, nva={nva}, nte={nte}).", flush=True)
        return

    # Modelo / otimizador
    model = ResnetSiamese().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device='cuda', enabled=amp_enabled)

    # Arquivos por época
    ep_txt = (run_dir / "metrics_epoch.txt").open("w", encoding="utf-8")
    ep_csv = run_dir / "metrics_epoch.csv"
    ep_header = ["run_slug","epoch","train_loss","val_roc_auc","val_pr_auc","val_acc@0.5","val_f1@0.5",
                 "val_tpr@0.01","val_tpr@0.05","val_tpr@0.10","val_eer","val_eer_thr"]
    write_csv_header_if_needed(ep_csv, ep_header)
    write_csv_header_if_needed(global_metrics_csv, ep_header)

    # Early stopping por ROC-AUC
    best_auc = -1.0
    best_epoch = -1
    best_acc_at_best = float("nan")
    no_improve = 0

    ep_iter = range(1, cfg.epochs+1)
    if use_tqdm:
        ep_iter = tqdm(ep_iter, desc=f"[{run_slug}] epochs", leave=False)

    for epoch in ep_iter:
        tr_loss = run_train_epoch(model, tr_loader, device, optimizer, scaler, criterion, amp_enabled,
                                  desc=f"train {epoch}/{cfg.epochs}", use_tqdm=use_tqdm)
        yv, pv = run_eval_collect(model, va_loader, device, amp_enabled, desc=f"valid {epoch}/{cfg.epochs}", use_tqdm=use_tqdm)
        vm = compute_all_metrics(yv, pv)

        # log por época (arquivo + csv local + csv global)
        ep_txt.write(
            f"{epoch}\t{tr_loss:.6f}\t{vm['roc_auc']:.6f}\t{vm['pr_auc']:.6f}\t"
            f"{vm['acc']:.6f}\t{vm['f1']:.6f}\t"
            f"{vm['tpr@0.01']:.6f}\t{vm['tpr@0.05']:.6f}\t{vm['tpr@0.10']:.6f}\t"
            f"{vm['eer']:.6f}\t{vm['eer_thr']:.6f}\n"
        ); ep_txt.flush()
        row = [run_slug, epoch, f"{tr_loss:.6f}",
               f"{vm['roc_auc']:.6f}", f"{vm['pr_auc']:.6f}",
               f"{vm['acc']:.6f}", f"{vm['f1']:.6f}",
               f"{vm['tpr@0.01']:.6f}", f"{vm['tpr@0.05']:.6f}", f"{vm['tpr@0.10']:.6f}",
               f"{vm['eer']:.6f}", f"{vm['eer_thr']:.6f}"]
        write_csv_row(ep_csv, row)
        write_csv_row(global_metrics_csv, row)

        val_auc = vm["roc_auc"]
        improved = (not np.isnan(val_auc)) and (val_auc > (best_auc + cfg.es_min_delta))
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            best_acc_at_best = vm["acc"]
            no_improve = 0
            # salva checkpoint + paeckle
            ckpt_pt = run_dir / "checkpoints" / f"best_epoch{epoch}_auc{best_auc:.4f}.pt"
            torch.save({"state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "run_slug": run_slug}, ckpt_pt)
            with (run_dir / "best_model.paeckle").open("wb") as f:
                pickle.dump({
                    "model_class": "ResnetSiamese",
                    "resnet_layers": (1,1,1,1),
                    "resnet_outsize": 10,
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "run_slug": run_slug,
                }, f)
        else:
            no_improve += 1

        if use_tqdm and hasattr(ep_iter, "set_postfix"):
            ep_iter.set_postfix(loss=f"{tr_loss:.4f}",
                                val_auc=f"{0.0 if np.isnan(val_auc) else val_auc:.4f}")

        if (epoch >= cfg.es_min_epochs) and (no_improve >= cfg.es_patience):
            break

    ep_txt.close()

    # resumo do melhor
    with (run_dir / "best_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_epoch\t{best_epoch}\n")
        f.write(f"best_val_roc_auc\t{best_auc:.6f}\n")
        f.write(f"best_val_acc_at_0.5\t{best_acc_at_best:.6f}\n")
        f.write(f"early_stop_patience\t{cfg.es_patience}\n")
        f.write(f"early_stop_min_delta\t{cfg.es_min_delta}\n")
        f.write(f"early_stop_min_epochs\t{cfg.es_min_epochs}\n")

    # carrega melhor antes de testar
    best_ckpts = sorted((run_dir / "checkpoints").glob("best_epoch*_auc*.pt"))
    if best_ckpts:
        try:
            ckpt = torch.load(best_ckpts[-1], map_location=device, weights_only=True)  # PyTorch >= 2.4
        except TypeError:
            ckpt = torch.load(best_ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    # Teste
    yt, pt = run_eval_collect(model, te_loader, device, amp_enabled, desc="test", use_tqdm=use_tqdm)
    tm = compute_all_metrics(yt, pt)
    y_pred = (pt >= 0.5).astype(np.int64)
    cm = confusion_matrix(yt, y_pred) if yt.size else np.zeros((2,2), dtype=int)

    # grava teste
    with (run_dir / "test_results.txt").open("w", encoding="utf-8") as f:
        f.write(f"test_acc@0.5\t{tm['acc']:.6f}\n")
        f.write(f"test_f1@0.5\t{tm['f1']:.6f}\n")
        f.write(f"test_roc_auc\t{tm['roc_auc']:.6f}\n")
        f.write(f"test_pr_auc\t{tm['pr_auc']:.6f}\n")
        f.write(f"test_tpr@0.01\t{tm['tpr@0.01']:.6f}\n")
        f.write(f"test_tpr@0.05\t{tm['tpr@0.05']:.6f}\n")
        f.write(f"test_tpr@0.10\t{tm['tpr@0.10']:.6f}\n")
        f.write(f"test_eer\t{tm['eer']:.6f}\n")
        f.write(f"test_eer_thr\t{tm['eer_thr']:.6f}\n")
        f.write("confusion_matrix@0.5\n")
        if isinstance(cm, np.ndarray) and cm.size == 4:
            f.write(f"{int(cm[0,0])},{int(cm[0,1])}\n{int(cm[1,0])},{int(cm[1,1])}\n")

    # resumo global (uma linha por run)
    summary_header = ["run_slug",
                      "best_epoch","best_val_roc_auc","best_val_acc@0.5",
                      "test_acc@0.5","test_f1@0.5","test_roc_auc","test_pr_auc",
                      "test_tpr@0.01","test_tpr@0.05","test_tpr@0.10",
                      "test_eer","test_eer_thr",
                      "cm00","cm01","cm10","cm11",
                      "es_patience","es_min_delta","es_min_epochs"]
    write_csv_header_if_needed(global_summary_csv, summary_header)
    write_csv_row(global_summary_csv, [
        run_slug,
        best_epoch, f"{best_auc:.6f}", f"{best_acc_at_best:.6f}",
        f"{tm['acc']:.6f}", f"{tm['f1']:.6f}", f"{tm['roc_auc']:.6f}", f"{tm['pr_auc']:.6f}",
        f"{tm['tpr@0.01']:.6f}", f"{tm['tpr@0.05']:.6f}", f"{tm['tpr@0.10']:.6f}",
        f"{tm['eer']:.6f}", f"{tm['eer_thr']:.6f}",
        tm["cm00"], tm["cm01"], tm["cm10"], tm["cm11"],
        cfg.es_patience, cfg.es_min_delta, cfg.es_min_epochs
    ])


# =========================
# Execução
# =========================
def print_env_info():
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    print(f"[ENV] host={host} cwd={os.getcwd()}", flush=True)
    print(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        try:
            print(f"[ENV] device_name={torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            pass


def main():
    cfg = parse_args()
    # prepara resultados e logger
    results_root = Path(cfg.results_root)
    ensure_dir(results_root)
    run_log = results_root / "_run_log.txt"
    logger = TeeLogger(run_log)
    sys.stdout = logger  # duplica stdout p/ arquivo
    sys.stderr = logger

    print_env_info()
    print(f"[CFG] {cfg}", flush=True)

    set_seed(cfg.seed)

    pairs_root   = Path(cfg.pairs_root)
    images_root  = Path(cfg.images_root) if cfg.images_root else Path("")
    ensure_dir(pairs_root)
    ensure_dir(images_root) if str(images_root) != "" else None

    # encontra triplets
    print(f"[INFO] procurando trios em: {pairs_root.resolve()}", flush=True)
    triplets = find_triplets(pairs_root, cfg.include or [], cfg.exclude or [])
    print(f"[INFO] encontrados {len(triplets)} trios.", flush=True)
    if not triplets:
        # Debug: listar *.txt existentes
        any_txt = list(pairs_root.rglob("*.txt"))
        print(f"[DEBUG] qtd .txt no pairs_root: {len(any_txt)}", flush=True)
        for p in any_txt[:20]:
            print(f" - {p}", flush=True)
        print("[ERRO] Nenhum trio *_train/_val/_test encontrado. Verifique nomes/caminhos.", flush=True)
        return 1

    # arquivos agregados globais
    global_metrics_csv = results_root / "_all_runs_metrics.csv"
    write_csv_header_if_needed(global_metrics_csv,
                               ["run_slug","epoch","train_loss",
                                "val_roc_auc","val_pr_auc","val_acc@0.5","val_f1@0.5",
                                "val_tpr@0.01","val_tpr@0.05","val_tpr@0.10",
                                "val_eer","val_eer_thr"])
    global_summary_csv = results_root / "_all_runs_summary.csv"
    write_csv_header_if_needed(global_summary_csv,
                               ["run_slug","best_epoch","best_val_roc_auc","best_val_acc@0.5",
                                "test_acc@0.5","test_f1@0.5","test_roc_auc","test_pr_auc",
                                "test_tpr@0.01","test_tpr@0.05","test_tpr@0.10",
                                "test_eer","test_eer_thr",
                                "cm00","cm01","cm10","cm11",
                                "es_patience","es_min_delta","es_min_epochs"])

    # device / amp
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    amp_enabled = use_cuda
    print(f"[INFO] device={device} amp={amp_enabled}", flush=True)

    # progresso por dataset
    iterator = triplets
    if not cfg.no_tqdm:
        iterator = tqdm(triplets, desc="Datasets", leave=True)

    for train_txt, val_txt, test_txt, base_rel in iterator:
        run_slug = base_rel
        if not cfg.no_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(run=Path(run_slug).name)
        ensure_dir(results_root / run_slug)

        print(f"\n[RUN] {run_slug}", flush=True)
        print(f"      train={train_txt}", flush=True)
        print(f"      valid={val_txt}", flush=True)
        print(f"      test ={test_txt}", flush=True)

        try:
            train_eval_one_run(cfg,
                               train_txt, val_txt, test_txt,
                               run_slug,
                               device, amp_enabled,
                               global_metrics_csv, global_summary_csv,
                               use_tqdm=(not cfg.no_tqdm))
        except Exception as e:
            print(f"[ERRO] Run '{run_slug}' falhou: {e}", flush=True)
            traceback.print_exc()
            # segue para o próximo

    print("\n[OK] Finalizado.", flush=True)
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
