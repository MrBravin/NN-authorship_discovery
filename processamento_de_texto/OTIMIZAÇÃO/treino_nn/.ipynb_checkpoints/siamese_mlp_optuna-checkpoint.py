# -*- coding: utf-8 -*-
"""
Script (CSV-only) para verificação de autoria com Siamese MLP + Optuna (GPU-ready).
- Estrutura e orquestração semelhantes ao seu `treino_optuna_svm_final2.py`.
- Gera/usa df_idx com split por AUTOR (autores desjuntos entre train/test).
- TUNING com Optuna em SUBCONJUNTO balanceado do split de train (sem CV pesada).
- TREINO FINAL no train inteiro com early stopping e avaliação no test.
- Sem dropout (a pedido). Espaço de busca "inteligente mas nem tanto".

Requisitos:
    pip install numpy pandas joblib optuna scikit-learn torch
"""

# =====================
# VARIÁVEIS GLOBAIS
# =====================
SEED                 = 41                 # semente global
DEVICE               = "cuda"             # "cuda" se houver GPU; cai para "cpu" automaticamente se não tiver
AMP                  = True               # mixed precision para acelerar em GPU
CUDNN_BENCHMARK      = True               # kernels mais rápidos (pode variar resultado em 1e-4, ok)

# Optuna / Tuning
METRICA_TUNING       = 'f1'
N_TRIALS             = 50                 # nº de trials (modesto)
USE_TUNING_SUBSAMPLE = True               # usa subconjunto para tuning
N_PAIRS_TUNING       = 10_000             # pares balanceados (até esse máx) no tuning
VAL_FRACTION_TUNING  = 0.15               # fração para validação dentro do train, durante tuning
EPOCHS_TUNING_MAX    = 40                 # ép. máx por trial (curto)
EARLY_STOP_PATIENCE_TUNING = 6            # paciência de early stop no tuning

# Treino final
EPOCHS_FINAL_MAX     = 100                # ép. máx no treino final
EARLY_STOP_PATIENCE_FINAL  = 10           # paciência no final

# I/O
RUN_MEAN             = True               # roda pipeline para a coluna 'vec_mean'
RUN_CLS              = True               # roda pipeline para a coluna 'vec_cls'
RAW_CSV              = "../df_com_embeddings.csv"  # CSV com vetores e coluna de autor
COL_AUTOR            = "autor"            # coluna de autor
VEC_MEAN_COL         = "vec_mean"         # coluna com vetor (mean)
VEC_CLS_COL          = "vec_cls"          # coluna com vetor (cls)

# Parâmetros para GERAR df_idx internamente (quando USE_EXISTING_IDX=False)
USE_EXISTING_IDX     = False
PAIRS_CSV            = "df_idx.csv"
N_PAIRS_TOTAL        = 60_000             # total de pares (balanceado 1:1) a gerar
TEST_SIZE            = 0.20               # fração de autores para 'test'

OUT_DIR              = "resultados_siamese_mlp"     # pasta de saída
SAVE_MODELS          = True                          # salva .pt dos melhores modelos

# =====================
# IMPORTS
# =====================
import os, json, random, math
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, SequentialSampler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# =====================
# UTILS
# =====================
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def write_text(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# =====================
# (A) GERADOR DE PARES BALANCEADOS COM SPLIT POR AUTOR
# =====================
def gerar_pares_balanceados_indices_split(
    df: pd.DataFrame,
    n_pairs_total: int,
    col_autor: str = "autor",
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Gera df_idx com colunas: i1, i2, label (0/1) e split ('train'/'test').
    - Split por AUTOR (autores do train e test NÃO se intersectam).
    - Amostragem balanceada (50/50) dentro de cada split.
    - Trabalha apenas com índices (leve).
    """
    rng = np.random.default_rng(seed)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    autores = df[col_autor].values
    if len(df) < 2:
        return pd.DataFrame(columns=["i1","i2","label","split"])

    autores_unicos = np.array(pd.unique(autores))
    rng.shuffle(autores_unicos)
    n_test_aut = max(1, int(round(len(autores_unicos) * test_size)))
    autores_test = set(autores_unicos[:n_test_aut])
    autores_train = set(autores_unicos[n_test_aut:]) if n_test_aut < len(autores_unicos) else set()

    mask_train = np.array([a in autores_train for a in autores])
    mask_test  = np.array([a in autores_test  for a in autores])

    n_pairs_total = int(n_pairs_total // 2 * 2)
    n_pairs_train = int((1.0 - test_size) * n_pairs_total)
    n_pairs_test  = n_pairs_total - n_pairs_train

    def amostrar_subset(mask_subset: np.ndarray, n_pairs_target: int):
        if n_pairs_target <= 0 or mask_subset.sum() < 2:
            return []

        idx_subset = np.nonzero(mask_subset)[0]
        aut_subset = autores[idx_subset]

        author_to_idx = defaultdict(list)
        for pos_global, a in zip(idx_subset, aut_subset):
            author_to_idx[a].append(pos_global)

        autores_todos = [a for a, L in author_to_idx.items() if len(L) >= 1]
        autores_pos   = [a for a, L in author_to_idx.items() if len(L) >= 2]
        if len(autores_todos) < 2 and not autores_pos:
            return []

        sizes_all = np.array([len(author_to_idx[a]) for a in autores_todos], dtype=np.int64)
        p_all = sizes_all / sizes_all.sum()

        if autores_pos:
            sizes_pos = np.array([len(author_to_idx[a]) for a in autores_pos], dtype=np.int64)
            p_pos = np.maximum(sizes_pos - 1, 1); p_pos = p_pos / p_pos.sum()

        n_pairs_target = int(n_pairs_target // 2 * 2)
        n_pos_target = n_pairs_target // 2
        n_neg_target = n_pairs_target // 2

        rng_local = np.random.default_rng(seed)
        seen = set()
        rows = []

        # positivos
        if n_pos_target > 0 and autores_pos:
            while sum(1 for r in rows if r[2]==1) < n_pos_target:
                a = rng_local.choice(autores_pos, p=p_pos)
                L = author_to_idx[a]
                if len(L) < 2: continue
                i1, i2 = rng_local.choice(L, size=2, replace=False)
                i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
                key = (i_min, i_max, 1)
                if key in seen: continue
                seen.add(key)
                rows.append((i_min, i_max, 1))

        # negativos
        while len(rows) < n_pairs_target:
            a1, a2 = rng_local.choice(autores_todos, size=2, replace=False, p=p_all)
            i1 = rng_local.choice(author_to_idx[a1])
            i2 = rng_local.choice(author_to_idx[a2])
            if i1 == i2: continue
            i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
            key = (i_min, i_max, 0)
            if key in seen: continue
            seen.add(key)
            rows.append((i_min, i_max, 0))

        return rows

    def montar_df(rows, split_name):
        if not rows:
            return pd.DataFrame(columns=["i1","i2","label","split"])
        out = pd.DataFrame(rows, columns=["i1","i2","label"])
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out["split"] = split_name
        return out

    rows_train = amostrar_subset(mask_train, n_pairs_train)
    rows_test  = amostrar_subset(mask_test,  n_pairs_test)

    df_train = montar_df(rows_train, "train")
    df_test  = montar_df(rows_test,  "test")
    out = pd.concat([df_train, df_test], ignore_index=True)

    if not out.empty:
        out = out.astype({"i1":"int32","i2":"int32","label":"uint8"})
        out["split"] = out["split"].astype("category")
    return out

# =====================
# (B) PARES -> DATASETS
# =====================
def _to_float_vector(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        s = x.strip()
        if s.lower().startswith("array("):
            li, ri = s.find("["), s.rfind("]")
            if li != -1 and ri != -1 and ri > li:
                s = s[li:ri+1]
        s = s.replace("\\n", " ").strip()
        arr = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
        if arr.size == 0:
            arr = np.array(eval(s), dtype=np.float32)
        return arr
    raise TypeError(f"Tipo não suportado para vetor: {type(x)}")

def _stack_vectors(df_pairs: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1_list, v2_list, y_list = [], [], []
    c1, c2, y = df_pairs[f"{col}_1"].values, df_pairs[f"{col}_2"].values, df_pairs["label"].values
    for a, b, t in zip(c1, c2, y):
        try:
            va, vb = _to_float_vector(a), _to_float_vector(b)
            if va.shape != vb.shape: continue
            if not (np.isfinite(va).all() and np.isfinite(vb).all()): continue
            v1_list.append(va); v2_list.append(vb); y_list.append(int(t))
        except Exception:
            continue
    if not v1_list:
        raise ValueError("Sem amostras válidas após converter vetores.")
    v1 = np.vstack(v1_list).astype(np.float32, copy=False)
    v2 = np.vstack(v2_list).astype(np.float32, copy=False)
    y  = np.asarray(y_list, dtype=np.int64)
    return v1, v2, y

class PairsDataset(Dataset):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, y: np.ndarray):
        assert v1.shape == v2.shape
        self.v1 = v1
        self.v2 = v2
        self.y  = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.v1[idx]),
            torch.from_numpy(self.v2[idx]),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

def preparar_train_test(df_pairs: pd.DataFrame, col: str):
    tr = df_pairs[df_pairs["split"] == "train"].reset_index(drop=True)
    te = df_pairs[df_pairs["split"] == "test"].reset_index(drop=True)
    v1_tr, v2_tr, y_tr = _stack_vectors(tr, col)
    v1_te, v2_te, y_te = _stack_vectors(te, col)
    return (v1_tr, v2_tr, y_tr), (v1_te, v2_te, y_te)

# =====================
# (C) SIAMESE MLP
# =====================
class TowerMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, proj_dim: int, activation: str = "relu", use_layernorm: bool = True):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU()
        }
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width))
            if use_layernorm:
                layers.append(nn.LayerNorm(width))
            layers.append(acts.get(activation, nn.ReLU()))
            last = width
        layers.append(nn.Linear(last, proj_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(proj_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SiameseMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, proj_dim: int, activation: str = "relu",
                 use_layernorm: bool = True, normalize_embeddings: bool = True, temperature: float = 0.1):
        super().__init__()
        self.tower = TowerMLP(in_dim, width, depth, proj_dim, activation, use_layernorm)
        self.normalize_embeddings = normalize_embeddings
        self.temperature = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)  # fixo no treino (pode virar trainável se quiser)

    def encode(self, x):
        z = self.tower(x)
        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def forward(self, x1, x2):
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        # similaridade de cosseno escalada
        sim = (z1 * z2).sum(dim=-1)  # cos se normalizado; caso contrário é dot
        logits = sim / (self.temperature + 1e-8)
        return logits

# =====================
# (D) TREINO / AVALIAÇÃO
# =====================
def make_loaders(v1, v2, y, batch_size: int, val_fraction: float, seed: int):
    ds = PairsDataset(v1, v2, y)
    n  = len(ds)
    n_val = max(1, int(round(n * val_fraction)))
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]
    ds_trn  = Subset(ds, trn_idx)
    ds_val  = Subset(ds, val_idx)
    dl_trn  = DataLoader(ds_trn, batch_size=batch_size, sampler=RandomSampler(ds_trn),
                         num_workers=2, pin_memory=True, drop_last=False)
    dl_val  = DataLoader(ds_val, batch_size=batch_size, sampler=SequentialSampler(ds_val),
                         num_workers=2, pin_memory=True, drop_last=False)
    return dl_trn, dl_val

@torch.no_grad()
def _predict_logits(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True):
    model.eval()
    all_logits, all_y = [], []
    for a,b,y in loader:
        a = a.to(device, non_blocking=True).float()
        b = b.to(device, non_blocking=True).float()
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16 if "cuda" in device.type else torch.bfloat16):
                logits = model(a,b)
        else:
            logits = model(a,b)
        all_logits.append(logits.detach().float().cpu())
        all_y.append(y.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()

def eval_metrics_from_logits(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    # logit>=0 => classe 1
    y_pred = (logits >= 0.0).astype(np.int32)
    # prob ~ sigmoid(logit) só para métricas de ranking
    prob = 1.0 / (1.0 + np.exp(-logits))
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"]       = float(f1_score(y_true, y_pred))
    try:
        out["roc_auc"]= float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"]= float("nan")
    try:
        out["ap"]     = float(average_precision_score(y_true, prob))
    except Exception:
        out["ap"]     = float("nan")
    return out

def train_one_model(
    v1_tr: np.ndarray, v2_tr: np.ndarray, y_tr: np.ndarray,
    v1_val: np.ndarray, v2_val: np.ndarray, y_val: np.ndarray,
    in_dim: int, width: int, depth: int, proj_dim: int,
    activation: str, use_layernorm: bool,
    lr: float, weight_decay: float, batch_size: int,
    temperature: float,
    max_epochs: int, patience: int,
    device_str: str = DEVICE, amp: bool = AMP, grad_clip: float = 1.0
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = SiameseMLP(in_dim, width, depth, proj_dim, activation, use_layernorm, True, temperature).to(device)
    if CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and torch.cuda.is_available()))
    loss_fn = nn.BCEWithLogitsLoss()

    ds_tr = PairsDataset(v1_tr, v2_tr, y_tr)
    ds_va = PairsDataset(v1_val, v2_val, y_val)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=RandomSampler(ds_tr), num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, sampler=SequentialSampler(ds_va), num_workers=2, pin_memory=True)

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        for a,b,y in dl_tr:
            a = a.to(device, non_blocking=True).float()
            b = b.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            if amp and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(a,b)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(a,b)
                loss = loss_fn(logits, y)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        sched.step()

        # validação
        logits_va, y_va = _predict_logits(model, dl_va, device, amp=amp and torch.cuda.is_available())
        metrics_va = eval_metrics_from_logits(logits_va, y_va)
        val_score = metrics_va.get("ap", 0.0)

        if val_score > best_val:
            best_val = val_score
            best_state = { "model": model.state_dict(), "metrics_val": metrics_va, "epoch": epoch }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # restaura melhor
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    return model, (best_state["metrics_val"] if best_state is not None else {"ap":-1})

# =====================
# (E) TUNING (espaço de busca modesto, sem dropout)
# =====================
def objective_factory(v1_tr_full, v2_tr_full, y_tr_full, in_dim: int, seed: int = SEED):
    def objective(trial: optuna.Trial) -> float:
        # Espaço "inteligente mas nem tanto"
        depth = trial.suggest_categorical("depth", [1,2,3,4,5])
        width = trial.suggest_categorical("width", [256, 512, 768])
        proj_dim = trial.suggest_categorical("proj_dim", [64,128, 256])
        activation = trial.suggest_categorical("activation", ["relu", "gelu"])
        use_layernorm = trial.suggest_categorical("use_layernorm", [True])  # fixado True (estável); mude p/ [True, False] se quiser
        lr = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        temperature = trial.suggest_float("temperature", 0.07, 1.0, log=True)

        # Subamostra balanceada para tuning
        v1_tu, v2_tu, y_tu = v1_tr_full, v2_tr_full, y_tr_full
        if USE_TUNING_SUBSAMPLE:
            v1_tu, v2_tu, y_tu = _amostrar_para_tuning(v1_tr_full, v2_tr_full, y_tr_full, N_PAIRS_TUNING, seed)

        # Split interno train/val
        dl_tr, dl_val = make_loaders(v1_tu, v2_tu, y_tu, batch_size, VAL_FRACTION_TUNING, seed)

        # Constrói arrays para treino/val
        Xtr_a, Xtr_b, Ytr = [], [], []
        for a,b,y in dl_tr:
            Xtr_a.append(a.numpy()); Xtr_b.append(b.numpy()); Ytr.append(y.numpy())
        Xva_a, Xva_b, Yva = [], [], []
        for a,b,y in dl_val:
            Xva_a.append(a.numpy()); Xva_b.append(b.numpy()); Yva.append(y.numpy())
        v1_tr = np.vstack(Xtr_a); v2_tr = np.vstack(Xtr_b); y_tr = np.concatenate(Ytr)
        v1_va = np.vstack(Xva_a); v2_va = np.vstack(Xva_b); y_va = np.concatenate(Yva)

        # Treina um modelo curto com early stopping
        model, val_metrics = train_one_model(
            v1_tr, v2_tr, y_tr, v1_va, v2_va, y_va,
            in_dim=in_dim, width=width, depth=depth, proj_dim=proj_dim,
            activation=activation, use_layernorm=use_layernorm,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            temperature=temperature,
            max_epochs=EPOCHS_TUNING_MAX, patience=EARLY_STOP_PATIENCE_TUNING,
        )
        # Retorna AP de validação
        return float(val_metrics.get(METRICA_TUNING, 0.0))
    return objective

def _amostrar_para_tuning(v1: np.ndarray, v2: np.ndarray, y: np.ndarray, n_pairs: int, seed: int):
    # balanceado 1:1 sempre que possível
    n_pairs = max(2, int(n_pairs // 2 * 2))
    idx_pos = np.nonzero(y == 1)[0]
    idx_neg = np.nonzero(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        rng = np.random.default_rng(seed)
        take = min(n_pairs, len(y))
        sel = rng.choice(len(y), size=take, replace=False)
        return v1[sel], v2[sel], y[sel]
    half = min(n_pairs // 2, len(idx_pos), len(idx_neg))
    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(idx_pos, size=half, replace=False)
    sel_neg = rng.choice(idx_neg, size=half, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return v1[sel], v2[sel], y[sel]

# =====================
# (F) ORQUESTRAÇÃO (mean/cls)
# =====================
def rodar_pipeline(df_pairs: pd.DataFrame, col: str, nome_tag: str):
    # prepara arrays
    (v1_tr, v2_tr, y_tr), (v1_te, v2_te, y_te) = preparar_train_test(df_pairs, col)
    in_dim = v1_tr.shape[1]

    print(f"\n==== {nome_tag.upper()} | TUNING (Optuna, {N_TRIALS} trials) ====")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_factory(v1_tr, v2_tr, y_tr, in_dim, SEED), n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_params

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(OUT_DIR, f"{nome_tag}_best_params.txt")
    write_text(best_path, json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[OK] Melhores hiperparâmetros salvos em: {best_path}")

    # Treino final (train inteiro) com early stopping e avaliação
    print(f"\n==== {nome_tag.upper()} | TREINO FINAL ====")
    # split interno só para monitorar early stopping no final
    dl_tr, dl_val = make_loaders(v1_tr, v2_tr, y_tr, best.get("batch_size", 512), 0.1, SEED)
    # monta arrays
    Xa, Xb, Y = [], [], []
    for a,b,y in dl_tr:
        Xa.append(a.numpy()); Xb.append(b.numpy()); Y.append(y.numpy())
    Va, Vb, Vy = [], [], []
    for a,b,y in dl_val:
        Va.append(a.numpy()); Vb.append(b.numpy()); Vy.append(y.numpy())
    v1_tr_f = np.vstack(Xa); v2_tr_f = np.vstack(Xb); y_tr_f = np.concatenate(Y)
    v1_va_f = np.vstack(Va); v2_va_f = np.vstack(Vb); y_va_f = np.concatenate(Vy)

    model, _ = train_one_model(
        v1_tr_f, v2_tr_f, y_tr_f, v1_va_f, v2_va_f, y_va_f,
        in_dim=in_dim,
        width=best["width"], depth=best["depth"], proj_dim=best["proj_dim"],
        activation=best["activation"], use_layernorm=best.get("use_layernorm", True),
        lr=best["lr"], weight_decay=best["weight_decay"], batch_size=best["batch_size"],
        temperature=best["temperature"],
        max_epochs=EPOCHS_FINAL_MAX, patience=EARLY_STOP_PATIENCE_FINAL
    )

    # avaliação no test
    ds_te = PairsDataset(v1_te, v2_te, y_te)
    dl_te = DataLoader(ds_te, batch_size=best["batch_size"], sampler=SequentialSampler(ds_te), num_workers=2, pin_memory=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logits_te, y_te_np = _predict_logits(model, dl_te, device, amp=AMP and torch.cuda.is_available())
    metrics_test = eval_metrics_from_logits(logits_te, y_te_np)

    # salvar
    write_text(os.path.join(OUT_DIR, f"{nome_tag}_test_metrics.txt"), json.dumps(metrics_test, ensure_ascii=False, indent=2))
    print("MÉTRICAS NO TEST:")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")

    if SAVE_MODELS:
        model_out = os.path.join(OUT_DIR, f"siamese_mlp_{nome_tag}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "best_params": best,
            "config": {
                "activation": best["activation"],
                "normalize_embeddings": True,
            }
        }, model_out)
        print(f"[OK] Modelo salvo em: {model_out}")

# =====================
# (G) MONTAGEM DE DF_PARES PARA CADA COLUNA
# =====================
def pares_com_infos(df_raw: pd.DataFrame, df_pares: pd.DataFrame, coluna: str) -> pd.DataFrame:
    if not isinstance(df_raw.index, pd.RangeIndex):
        df_raw = df_raw.reset_index(drop=True)
    col1 = df_raw.loc[df_pares['i1'].values, coluna].reset_index(drop=True)
    col2 = df_raw.loc[df_pares['i2'].values, coluna].reset_index(drop=True)
    out = pd.DataFrame({
        f"{coluna}_1": col1,
        f"{coluna}_2": col2,
        "label": df_pares["label"].reset_index(drop=True),
        "split": df_pares["split"].reset_index(drop=True),
    })
    return out

# =====================
# MAIN
# =====================
def main():
    set_global_seed(SEED)

    # ler df_raw
    df_raw = read_csv_flexible(RAW_CSV)
    if COL_AUTOR not in df_raw.columns:
        raise ValueError(f"df_raw precisa conter a coluna de autor '{COL_AUTOR}'.")

    # obter df_idx
    if USE_EXISTING_IDX:
        df_idx = read_csv_flexible(PAIRS_CSV)
        for c in ["i1","i2","label","split"]:
            if c not in df_idx.columns:
                raise ValueError(f"df_idx precisa ter a coluna '{c}'.")
        print("[INFO] Usando df_idx existente.")
    else:
        print("[INFO] Gerando df_idx (pares balanceados com split por autor)...")
        df_idx = gerar_pares_balanceados_indices_split(
            df=df_raw,
            n_pairs_total=N_PAIRS_TOTAL,
            col_autor=COL_AUTOR,
            test_size=TEST_SIZE,
            seed=SEED,
        )
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        df_idx_path = os.path.join(OUT_DIR, "df_idx_gerado.csv")
        df_idx.to_csv(df_idx_path, index=False)
        print(f"[OK] df_idx salvo em: {df_idx_path}")

    # mean
    if RUN_MEAN:
        if VEC_MEAN_COL not in df_raw.columns:
            raise ValueError(f"Coluna '{VEC_MEAN_COL}' não encontrada.")
        df_mean = pares_com_infos(df_raw, df_idx, VEC_MEAN_COL)
        rodar_pipeline(df_mean, VEC_MEAN_COL, "mean")

    # cls
    if RUN_CLS:
        if VEC_CLS_COL not in df_raw.columns:
            raise ValueError(f"Coluna '{VEC_CLS_COL}' não encontrada.")
        df_cls = pares_com_infos(df_raw, df_idx, VEC_CLS_COL)
        rodar_pipeline(df_cls, VEC_CLS_COL, "cls")

if __name__ == "__main__":
    main()
