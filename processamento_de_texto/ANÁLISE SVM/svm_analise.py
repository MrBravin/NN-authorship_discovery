#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de avaliação com SVM em pares (autoria) + gráficos salvos em uma pasta.

Saídas (arquivos salvos em --outdir):
- metrics.json
- curva_calibracao.png (se o modelo fornecer probas)
- pca_superficies_svm.png (3 painéis: f(x), P(y=1), P(y=0))
- score_hist_kde.png (histograma + KDE por classe)
- pca_color_score.png (PCA 2D colorido pelo score/proba do modelo real)
- tsne_classe_0_probabilidade.png
- tsne_classe_1_probabilidade.png
- tsne_acertos_erros.png
- tsne_visao_geral.png
"""
import os
import ast
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve

import joblib

# -----------------------------
# CONFIGS EMBUTIDAS (sem CLI)
# -----------------------------
EMBUTIDO_OUTDIR          = "resultados_svm"
EMBUTIDO_CSV             = "df_com_embeddings.csv"
EMBUTIDO_VEC_COL         = "vec_mean"
EMBUTIDO_AUTOR_COL       = "autor"
EMBUTIDO_PAIRS           = 60000
EMBUTIDO_TEST_SIZE       = 0.2
EMBUTIDO_RELATION        = "diff"
EMBUTIDO_SVM_PKL         = "svm_final.pkl"
EMBUTIDO_DO_TSNE         = 1
EMBUTIDO_TSNE_MAX        = 5000
EMBUTIDO_TSNE_PERPLEXITY = 30.0
EMBUTIDO_TSNE_ITERS      = 1000

# -----------------------------
# QoL em HPC/Slurm (limita threads se variável existir)
# -----------------------------
if "SLURM_CPUS_PER_TASK" in os.environ:
    _n = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", _n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _n)
    os.environ.setdefault("MKL_NUM_THREADS", _n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", _n)

# -----------------------------
# CONFIG VISUAL
# -----------------------------
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="whitegrid")

# -----------------------------
# SEEDS
# -----------------------------
SEED = 41
def set_global_seed(seed: int = 41):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(fig: plt.Figure, outdir: Path, filename: str):
    outpath = outdir / filename
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Figura salva: {outpath}")

def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON salvo: {path}")

# -----------------------------
# GERAÇÃO DE PARES (train/test por autor)
# -----------------------------
def gerar_pares_balanceados_indices_split(
    df: pd.DataFrame,
    n_pairs_total: int,
    col_autor: str = "autor",
    test_size: float = 0.2,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    autores = df[col_autor].values
    if len(df) < 2:
        return pd.DataFrame(columns=["i1", "i2", "label", "split"])

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
            p_pos = np.maximum(sizes_pos - 1, 1)
            p_pos = p_pos / p_pos.sum()

        n_pairs_target = int(n_pairs_target // 2 * 2)
        n_pos_target = n_pairs_target // 2
        n_neg_target = n_pairs_target // 2

        rng_local = np.random.default_rng(seed)
        seen = set()
        rows = []

        if n_pos_target > 0 and autores_pos:
            while sum(1 for r in rows if r[2] == 1) < n_pos_target:
                a = rng_local.choice(autores_pos, p=p_pos)
                L = author_to_idx[a]
                if len(L) < 2:
                    continue
                i1, i2 = rng_local.choice(L, size=2, replace=False)
                i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
                key = (i_min, i_max, 1)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((i_min, i_max, 1))

        while len(rows) < n_pairs_target:
            a1, a2 = rng_local.choice(autores_todos, size=2, replace=False, p=p_all)
            i1 = rng_local.choice(author_to_idx[a1])
            i2 = rng_local.choice(author_to_idx[a2])
            if i1 == i2:
                continue
            i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
            key = (i_min, i_max, 0)
            if key in seen:
                continue
            seen.add(key)
            rows.append((i_min, i_max, 0))

        return rows

    rows_train = amostrar_subset(mask_train, n_pairs_train)
    rows_test  = amostrar_subset(mask_test,  n_pairs_test)

    def montar_df(rows, split_name):
        if not rows:
            return pd.DataFrame(columns=["i1", "i2", "label", "split"])
        out = pd.DataFrame(rows, columns=["i1", "i2", "label"])
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out["split"] = split_name
        return out

    df_train = montar_df(rows_train, "train")
    df_test  = montar_df(rows_test,  "test")
    out = pd.concat([df_train, df_test], ignore_index=True)

    if not out.empty:
        out = out.astype({"i1": "int32", "i2": "int32", "label": "uint8"})
        out["split"] = out["split"].astype("category")

    return out

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

# -----------------------------
# STACKING DOS VETORES
# -----------------------------
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
        s = s.replace("\n", " ").strip()
        try:
            arr = np.array(ast.literal_eval(s), dtype=np.float32)
            return arr
        except Exception:
            arr = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
            if arr.size == 0:
                arr = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
            return arr
    raise TypeError(f"Tipo não suportado na coluna de vetor: {type(x)}")

def _stack_vectors(df_pairs: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1_list, v2_list, y_list = [], [], []
    c1, c2, y = df_pairs[f"{col}_1"].values, df_pairs[f"{col}_2"].values, df_pairs["label"].values
    for a, b, t in zip(c1, c2, y):
        try:
            va, vb = _to_float_vector(a), _to_float_vector(b)
            if va.shape != vb.shape:
                continue
            if not (np.isfinite(va).all() and np.isfinite(vb).all()):
                continue
            v1_list.append(va); v2_list.append(vb); y_list.append(int(t))
        except Exception:
            continue
    if not v1_list:
        raise ValueError("Sem amostras válidas após converter os vetores.")
    v1 = np.vstack(v1_list).astype(np.float32, copy=False)
    v2 = np.vstack(v2_list).astype(np.float32, copy=False)
    y  = np.asarray(y_list, dtype=np.int64)
    return v1, v2, y

def preparar_train_test(df_pairs: pd.DataFrame, col: str):
    tr = df_pairs[df_pairs["split"] == "train"].reset_index(drop=True)
    te = df_pairs[df_pairs["split"] == "test"].reset_index(drop=True)
    v1_tr, v2_tr, y_tr = _stack_vectors(tr, col)
    v1_te, v2_te, y_te = _stack_vectors(te, col)
    return (v1_tr, v2_tr, y_tr), (v1_te, v2_te, y_te)

# -----------------------------
# FEATURES ENTRE PARES
# -----------------------------
def build_features(v1: np.ndarray, v2: np.ndarray, relation: str) -> np.ndarray:
    eps = 1e-12
    diff = v1 - v2

    v1n = np.linalg.norm(v1, axis=1, keepdims=True) + eps
    v2n = np.linalg.norm(v2, axis=1, keepdims=True) + eps
    dot = (v1 * v2).sum(axis=1, keepdims=True)
    cos = dot / (v1n * v2n)
    eucl = np.linalg.norm(diff, axis=1, keepdims=True)
    manh = np.abs(diff).sum(axis=1, keepdims=True)
    cheb = np.abs(diff).max(axis=1, keepdims=True)
    proj_v1_em_v2 = dot / v2n
    proj_v2_em_v1 = dot / v1n

    if relation == "cos":      return cos
    if relation == "eucl":     return eucl
    if relation == "manh":     return manh
    if relation == "cheb":     return cheb
    if relation == "dot":      return dot
    if relation == "proj":     return np.concatenate([proj_v1_em_v2, proj_v2_em_v1], axis=1)
    if relation == "scalars":  return np.concatenate([cos, eucl, manh, cheb, dot], axis=1)
    if relation == "diff":     return diff
    if relation == "absdiff":  return np.abs(diff)
    if relation == "hadamard": return v1 * v2
    if relation == "v1v2":     return np.concatenate([v1, v2], axis=1)
    raise ValueError(f"Relação desconhecida: {relation}")

# -----------------------------
# PREDIÇÕES DO MODELO (proba/score)
# -----------------------------
def predict_scores_and_labels(model, X: np.ndarray):
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int) if hasattr(model, "classes_") else model.predict(X)
        return y_proba, y_proba, y_pred, True
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        y_pred = (df >= 0.0).astype(int)
        return df, None, y_pred, False
    y_pred = np.asarray(model.predict(X)).astype(int)
    return y_pred, None, y_pred, False

# -----------------------------
# PLOTS MELHORADOS
# -----------------------------
def plot_calibration(y_true, y_proba, outdir: Path, title_prefix: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_pred, frac_pos, "s-", label="Modelo SVM", linewidth=2, markersize=6)
    ax.plot([0, 1], [0, 1], "k--", label="Calibração perfeita", linewidth=1.5)
    ax.set_xlabel("Probabilidade média prevista", fontsize=12)
    ax.set_ylabel("Fração de positivos reais", fontsize=12)
    ax.set_title("Curva de Calibração do Modelo", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    savefig(fig, outdir, "curva_calibracao.png")

def plot_pca_surfaces(X_test, y_test, svm_model, outdir: Path, title_prefix: str):
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X_test)
    pc1_var, pc2_var = pca.explained_variance_ratio_

    base = svm_model.named_steps["svm"] if hasattr(svm_model, "named_steps") else svm_model
    C_val = getattr(base, "C", 1.0)
    gamma_val = getattr(base, "gamma", "scale")

    svm2 = SVC(kernel="rbf", C=C_val, gamma=gamma_val, probability=True, random_state=SEED)
    svm2.fit(X2, y_test)

    H = 0.03
    x_min, x_max = X2[:, 0].min()-0.5, X2[:, 0].max()+0.5
    y_min, y_max = X2[:, 1].min()-0.5, X2[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))
    grid = np.c_[xx.ravel(), yy.ravel()]

    P1 = svm2.predict_proba(grid)[:, 1].reshape(xx.shape)
    P0 = 1.0 - P1
    DF = svm2.decision_function(grid).reshape(xx.shape)

    m0 = (y_test == 0); m1 = (y_test == 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # Painel 1: Função de decisão
    ax = axes[0]
    im = ax.contourf(xx, yy, DF, levels=50, cmap="RdBu_r", alpha=0.8)
    ax.contour(xx, yy, DF, levels=[0.0], colors="k", linewidths=2.5)
    ax.contour(xx, yy, DF, levels=[-1, 1], colors="k", linestyles="--", linewidths=1.5, alpha=0.7)
    # Pontos com transparência
    scatter0 = ax.scatter(X2[m0, 0], X2[m0, 1], c='blue', alpha=0.6, s=30, label='Classe 0', edgecolors='white', linewidth=0.5)
    scatter1 = ax.scatter(X2[m1, 0], X2[m1, 1], c='red', alpha=0.6, s=30, label='Classe 1', edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f"Componente Principal 1 ({pc1_var:.1%} variância)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({pc2_var:.1%} variância)", fontsize=11)
    ax.set_title("Função de Decisão do SVM\n(Hiperplano e Margens)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    fig.colorbar(im, ax=ax, label="Valor da função de decisão")

    # Painel 2: Probabilidade Classe 1
    ax = axes[1]
    im = ax.contourf(xx, yy, P1, levels=50, cmap="RdYlBu", alpha=0.85)
    ax.contour(xx, yy, P1, levels=[0.5], colors="k", linewidths=2.5)
    scatter0 = ax.scatter(X2[m0, 0], X2[m0, 1], c='blue', alpha=0.5, s=30, label='Classe 0')
    scatter1 = ax.scatter(X2[m1, 0], X2[m1, 1], c='red', alpha=0.5, s=30, label='Classe 1')
    ax.set_xlabel(f"Componente Principal 1 ({pc1_var:.1%} variância)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({pc2_var:.1%} variância)", fontsize=11)
    ax.set_title("Probabilidade - Classe Positiva\n(Fronteira de Decisão P=0.5)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    fig.colorbar(im, ax=ax, label="P(Classe = 1 | x)")

    # Painel 3: Probabilidade Classe 0
    ax = axes[2]
    im = ax.contourf(xx, yy, P0, levels=50, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(xx, yy, P0, levels=[0.5], colors="k", linewidths=2.5)
    scatter0 = ax.scatter(X2[m0, 0], X2[m0, 1], c='blue', alpha=0.5, s=30, label='Classe 0')
    scatter1 = ax.scatter(X2[m1, 0], X2[m1, 1], c='red', alpha=0.5, s=30, label='Classe 1')
    ax.set_xlabel(f"Componente Principal 1 ({pc1_var:.1%} variância)", fontsize=11)
    ax.set_ylabel(f"Componente Principal 2 ({pc2_var:.1%} variância)", fontsize=11)
    ax.set_title("Probabilidade - Classe Negativa\n(Fronteira de Decisão P=0.5)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    fig.colorbar(im, ax=ax, label="P(Classe = 0 | x)")
    
    fig.suptitle(f"Análise de Superfície de Decisão em PCA\nVariância total explicada: {(pc1_var+pc2_var):.1%}",
                 fontsize=14, fontweight='bold', y=1.02)
    savefig(fig, outdir, "pca_superficies_svm.png")

def plot_hist_kde_scores(y_scores, y_true, outdir: Path, title_prefix: str, scores_are_prob: bool):
    df_plot = pd.DataFrame({"score": y_scores, "classe": y_true})
    df_plot['classe'] = df_plot['classe'].map({0: 'Classe Negativa', 1: 'Classe Positiva'})
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histograma com KDE
    sns.histplot(
        data=df_plot, x="score", hue="classe", kde=True,
        element="step", palette={"Classe Negativa": "blue", "Classe Positiva": "red"}, 
        common_norm=False, ax=ax, alpha=0.6, bins=30
    )
    
    # Linha de decisão
    ax.axvline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.8, label="Limiar de decisão (0.5)")
    
    ax.set_title("Distribuição dos Scores por Classe", fontsize=14, fontweight='bold')
    if scores_are_prob:
        ax.set_xlabel("Probabilidade Prevista (Classe Positiva)", fontsize=12)
    else:
        ax.set_xlabel("Score Contínuo do Modelo", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "score_hist_kde.png")

def plot_pca_colored_by_score(X_test, y_true, scores, outdir: Path, title_prefix: str):
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X_test)
    pc1, pc2 = pca.explained_variance_ratio_
    
    # Identificar pontos próximos da fronteira
    mid = (scores >= 0.45) & (scores <= 0.55)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Scatter plot principal colorido pelo score
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=scores, s=40, cmap="coolwarm", 
                   alpha=0.7, linewidths=0.5, edgecolors='white')
    
    # Destacar pontos próximos da fronteira
    ax.scatter(X2[mid, 0], X2[mid, 1], facecolors='none', edgecolors='black', 
              s=80, linewidths=1.5, label="Próximo da fronteira\n(0.45 ≤ P ≤ 0.55)")
    
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Probabilidade da Classe Positiva", fontsize=11)
    
    ax.set_xlabel(f"Componente Principal 1 ({pc1:.1%} variância)", fontsize=12)
    ax.set_ylabel(f"Componente Principal 2 ({pc2:.1%} variância)", fontsize=12)
    ax.set_title("Visualização em PCA Colorida pela Probabilidade", fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "pca_color_score.png")

def plot_tsne_diverse(X_test, y_test, svm_model, outdir: Path, title_prefix: str,
                     perplexity: float = 30.0, max_iter: int = 1000,
                     max_samples: int = 5000):
    n = len(X_test)
    if n > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_vis = X_test[idx]
        y_vis = y_test[idx]
        scores_vis = svm_model.predict_proba(X_vis)[:, 1] if hasattr(svm_model, "predict_proba") else svm_model.decision_function(X_vis)
    else:
        X_vis = X_test
        y_vis = y_test
        scores_vis = svm_model.predict_proba(X_vis)[:, 1] if hasattr(svm_model, "predict_proba") else svm_model.decision_function(X_vis)

    # Aplicar t-SNE
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, learning_rate=200,
                       init="pca", random_state=SEED, verbose=0)
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=max_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=max_iter)
    X_tsne = tsne.fit_transform(X_vis)

    # Treinar SVM no espaço t-SNE para as superfícies
    base = svm_model.named_steps["svm"] if hasattr(svm_model, "named_steps") else svm_model
    C_val = getattr(base, "C", 1.0)
    gamma_val = getattr(base, "gamma", "scale")
    svm2 = SVC(kernel="rbf", C=C_val, gamma=gamma_val, probability=True, random_state=SEED)
    svm2.fit(X_tsne, y_vis)

    # Grade para as superfícies
    GRID_H = 0.02
    x_min, x_max = X_tsne[:, 0].min()-0.5, X_tsne[:, 0].max()+0.5
    y_min, y_max = X_tsne[:, 1].min()-0.5, X_tsne[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, GRID_H), np.arange(y_min, y_max, GRID_H))
    grid = np.c_[xx.ravel(), yy.ravel()]
    P1 = svm2.predict_proba(grid)[:, 1].reshape(xx.shape)

    # 1. Plot: Apenas pontos da classe 0 com heatmap de probabilidade
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(xx, yy, P1, levels=50, cmap="RdYlBu", alpha=0.8)
    ax.contour(xx, yy, P1, levels=[0.5], colors="k", linewidths=2.5)
    # Apenas classe 0
    mask_0 = (y_vis == 0)
    ax.scatter(X_tsne[mask_0, 0], X_tsne[mask_0, 1], c='blue', alpha=0.7, s=40, 
               label='Classe Negativa', edgecolors='white', linewidth=0.5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(Classe = 1 | x)", fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE: Classe Negativa com Superfície de Probabilidade", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "tsne_classe_0_probabilidade.png")

    # 2. Plot: Apenas pontos da classe 1 com heatmap de probabilidade
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(xx, yy, P1, levels=50, cmap="RdYlBu", alpha=0.8)
    ax.contour(xx, yy, P1, levels=[0.5], colors="k", linewidths=2.5)
    # Apenas classe 1
    mask_1 = (y_vis == 1)
    ax.scatter(X_tsne[mask_1, 0], X_tsne[mask_1, 1], c='red', alpha=0.7, s=40, 
               label='Classe Positiva', edgecolors='white', linewidth=0.5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(Classe = 1 | x)", fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE: Classe Positiva com Superfície de Probabilidade", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "tsne_classe_1_probabilidade.png")

    # 3. Plot: Matchs e Mismatchs (acertos e erros)
    y_pred_vis = (scores_vis >= 0.5).astype(int)
    correct = (y_pred_vis == y_vis)
    incorrect = (y_pred_vis != y_vis)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(xx, yy, P1, levels=50, cmap="RdYlBu", alpha=0.6)
    ax.contour(xx, yy, P1, levels=[0.5], colors="k", linewidths=2.5)
    
    # Acertos
    ax.scatter(X_tsne[correct, 0], X_tsne[correct, 1], c='green', alpha=0.7, s=40, 
               label='Previsões Corretas', edgecolors='white', linewidth=0.5)
    # Erros
    ax.scatter(X_tsne[incorrect, 0], X_tsne[incorrect, 1], c='black', alpha=0.9, s=60, 
               label='Previsões Incorretas', edgecolors='yellow', linewidth=1.5, marker='X')
    
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(Classe = 1 | x)", fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE: Análise de Acertos e Erros de Classificação", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "tsne_acertos_erros.png")

    # 4. Plot: Todos os pontos com visualização sutil
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(xx, yy, P1, levels=50, cmap="RdYlBu", alpha=0.8)
    ax.contour(xx, yy, P1, levels=[0.5], colors="k", linewidths=2.5)
    
    # Todos os pontos com visualização sutil
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=scores_vis, cmap="coolwarm", 
                        alpha=0.4, s=20, linewidths=0)
    
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Probabilidade da Classe Positiva", fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE: Visão Geral com Probabilidades", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, outdir, "tsne_visao_geral.png")

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", type=str, default=EMBUTIDO_OUTDIR)
    parser.add_argument("--csv", type=str, default=EMBUTIDO_CSV)
    parser.add_argument("--vec-col", type=str, default=EMBUTIDO_VEC_COL)
    parser.add_argument("--autor-col", type=str, default=EMBUTIDO_AUTOR_COL)
    parser.add_argument("--pairs", type=int, default=EMBUTIDO_PAIRS)
    parser.add_argument("--test-size", type=float, default=EMBUTIDO_TEST_SIZE)
    parser.add_argument("--relation", type=str, default=EMBUTIDO_RELATION,
                        choices=["cos", "eucl", "manh", "cheb", "dot", "proj",
                                 "scalars", "diff", "absdiff", "hadamard", "v1v2"])
    parser.add_argument("--svm-pkl", type=str, default=EMBUTIDO_SVM_PKL)
    parser.add_argument("--do-tsne", type=int, default=EMBUTIDO_DO_TSNE)
    parser.add_argument("--tsne-max", type=int, default=EMBUTIDO_TSNE_MAX)
    parser.add_argument("--tsne-perplexity", type=float, default=EMBUTIDO_TSNE_PERPLEXITY)
    parser.add_argument("--tsne-iters", type=int, default=EMBUTIDO_TSNE_ITERS)
    args, _ = parser.parse_known_args([])

    set_global_seed(SEED)
    outdir = ensure_dir(args.outdir)
    title_prefix = f"SVM — {Path(args.csv).name} — relação={args.relation}"

    print(f"[INFO] Lendo CSV: {args.csv}")
    df_raw = read_csv_flexible(args.csv)
    if args.autor_col not in df_raw.columns or args.vec_col not in df_raw.columns:
        raise ValueError(f"CSV deve conter as colunas '{args.autor_col}' e '{args.vec_col}'.")
    print(f"[INFO] Total de textos: {len(df_raw)}")

    print("[INFO] Gerando pares balanceados (train/test por autor)...")
    df_idx = gerar_pares_balanceados_indices_split(
        df=df_raw, n_pairs_total=args.pairs, col_autor=args.autor_col,
        test_size=args.test_size, seed=SEED
    )
    n_train = int((df_idx["split"] == "train").sum())
    n_test  = int((df_idx["split"] == "test").sum())
    print(f"[OK] Pares gerados: {len(df_idx)} | train={n_train} | test={n_test}")

    df_pairs = pares_com_infos(df_raw, df_idx, coluna=args.vec_col)
    (train_v1, train_v2, y_train), (test_v1, test_v2, y_test) = preparar_train_test(df_pairs, col=args.vec_col)

    X_train = build_features(train_v1, train_v2, relation=args.relation)
    X_test  = build_features(test_v1,  test_v2,  relation=args.relation)

    print(f"[INFO] Dimensão das features: {X_train.shape[1]}")
    print(f"[INFO] Tamanho train={X_train.shape[0]} | test={X_test.shape[0]}")
    print(f"[INFO] Proporção classe positiva (train): {y_train.mean():.3f}")

    print(f"[INFO] Carregando SVM: {args.svm_pkl}")
    svm_model = joblib.load(args.svm_pkl)
    print("[OK] Modelo SVM carregado.")

    scores, y_proba, y_pred, has_proba = predict_scores_and_labels(svm_model, X_test)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
        metrics["ap"]      = float(average_precision_score(y_test, scores))
    except Exception:
        metrics["roc_auc"] = None
        metrics["ap"] = None

    write_json(outdir / "metrics.json", metrics)

    if has_proba and y_proba is not None:
        plot_calibration(y_test, y_proba, outdir, title_prefix)
    else:
        print("[WARN] Modelo sem predict_proba — curva de calibração pulada.")

    plot_pca_surfaces(X_test, y_test, svm_model, outdir, title_prefix)
    plot_hist_kde_scores(scores, y_test, outdir, title_prefix, scores_are_prob=has_proba)
    plot_pca_colored_by_score(X_test, y_test, scores, outdir, title_prefix)

    if int(args.do_tsne) == 1:
        plot_tsne_diverse(
            X_test, y_test, svm_model, outdir, title_prefix,
            perplexity=args.tsne_perplexity, max_iter=args.tsne_iters,
            max_samples=args.tsne_max
        )
    else:
        print("[INFO] t-SNE desativado.")

    print(f"[DONE] Concluído. Arquivos salvos em: {outdir.resolve()}")

if __name__ == "__main__":
    main()