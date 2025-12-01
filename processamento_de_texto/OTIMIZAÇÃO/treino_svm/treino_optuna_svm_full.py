
# -*- coding: utf-8 -*-


"""
Script completo e simples (CSV-only) para verificação de autoria com SVM + Optuna.

O script:
1) Lê um CSV "df_raw" com, no mínimo: a coluna de autor (COL_AUTOR) e
   uma ou duas colunas de vetores por texto (ex.: 'vec_mean' e/ou 'vec_cls').
2) (Opcional) Gera internamente os pares balanceados (df_idx) com split por AUTOR (sem interseção):
      - quantidade total de pares controlada por N_PAIRS_TOTAL
      - fração de autores de teste controlada por TEST_SIZE
3) Monta df_mean e/ou df_cls a partir de df_raw + df_idx (preserva split e label).
4) Roda Optuna com CV APENAS no train (sem vazamento), encontra melhores hiperparâmetros.
5) Reajusta no train inteiro com os melhores parâmetros e avalia uma única vez no test.
6) Salva relatórios simples em TXT (e modelo .joblib se desejar).

Requisitos:
    pip install numpy pandas scikit-learn optuna joblib
"""

# =====================
# VARIÁVEIS GLOBAIS
# =====================
SEED            = 41                # semente global
N_TRIALS        = 40                # número de trials do Optuna
SCORING_CV      = "roc_auc"         # 'roc_auc' ou 'f1'
N_SPLITS_CV     = 5                 # folds na validação cruzada
RUN_MEAN        = True              # roda pipeline para a coluna 'vec_mean'
RUN_CLS         = True              # roda pipeline para a coluna 'vec_cls'
RAW_CSV         = "../df_com_embeddings.csv"      # caminho do CSV com vetores por texto e coluna de autor
# Se você JÁ tiver um df_idx pronto (CSV com i1,i2,label,split),
# defina USE_EXISTING_IDX=True e PAIRS_CSV para seu arquivo.
USE_EXISTING_IDX= False
PAIRS_CSV       = "df_idx.csv"      # caminho do CSV com pares (usado se USE_EXISTING_IDX=True)

# Parâmetros para GERAR df_idx internamente (quando USE_EXISTING_IDX=False)
COL_AUTOR       = "autor"           # nome da coluna de autor em df_raw
N_PAIRS_TOTAL   = 60000            # número total de pares (será arredondado para par)
TEST_SIZE       = 0.2               # fração de autores que vão para o split de 'test'

# Nomes das colunas de vetores em df_raw
VEC_MEAN_COL    = "vec_mean"
VEC_CLS_COL     = "vec_cls"

OUT_DIR         = "resultados_svm"  # pasta de saída (csv/txt/modelos)
SAVE_MODELS     = False             # se True, salva modelos com joblib
RELATIONS       = (                 # conjunto de relações candidatas
    "scalars","cos","eucl","manh","cheb","dot","proj","diff","absdiff","hadamard","v1v2","siamese4"
)

import os, json, ast, random
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# =====================
# SEED GLOBAL
# =====================
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# =====================
# I/O SIMPLES (CSV)
# =====================
def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    """Tenta ler CSV com encodings comuns (sem firula)."""
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

    # índice posicional
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    autores = df[col_autor].values
    if len(df) < 2:
        return pd.DataFrame(columns=["i1","i2","label","split"])

    # Split por autor (disjunto)
    autores_unicos = np.array(pd.unique(autores))
    rng.shuffle(autores_unicos)
    n_test_aut = max(1, int(round(len(autores_unicos) * test_size)))
    autores_test = set(autores_unicos[:n_test_aut])
    autores_train = set(autores_unicos[n_test_aut:]) if n_test_aut < len(autores_unicos) else set()

    mask_train = np.array([a in autores_train for a in autores])
    mask_test  = np.array([a in autores_test  for a in autores])

    # Divisão de pares por split
    n_pairs_total = int(n_pairs_total // 2 * 2)  # garante par
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

        seen = set()
        rows = []

        # positivos (mesmo autor)
        if n_pos_target > 0 and autores_pos:
            while sum(1 for r in rows if r[2]==1) < n_pos_target:
                a = rng.choice(autores_pos, p=p_pos)
                L = author_to_idx[a]
                if len(L) < 2: continue
                i1, i2 = rng.choice(L, size=2, replace=False)
                i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
                key = (i_min, i_max, 1)
                if key in seen: continue
                seen.add(key)
                rows.append((i_min, i_max, 1))

        # negativos (autores diferentes)
        while len(rows) < n_pairs_target:
            a1, a2 = rng.choice(autores_todos, size=2, replace=False, p=p_all)
            i1 = rng.choice(author_to_idx[a1])
            i2 = rng.choice(author_to_idx[a2])
            if i1 == i2: continue
            i_min, i_max = (i1, i2) if i1 < i2 else (i2, i1)
            key = (i_min, i_max, 0)
            if key in seen: continue
            seen.add(key)
            rows.append((i_min, i_max, 0))

        return rows

    rows_train = amostrar_subset(mask_train, n_pairs_train)
    rows_test  = amostrar_subset(mask_test,  n_pairs_test)

    def montar_df(rows, split_name):
        if not rows:
            return pd.DataFrame(columns=["i1","i2","label","split"])
        out = pd.DataFrame(rows, columns=["i1","i2","label"])
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out["split"] = split_name
        return out

    df_train = montar_df(rows_train, "train")
    df_test  = montar_df(rows_test,  "test")
    out = pd.concat([df_train, df_test], ignore_index=True)

    if not out.empty:
        out = out.astype({"i1":"int32","i2":"int32","label":"uint8"})
        out["split"] = out["split"].astype("category")

    return out

# =====================
# (B) MONTAGEM DOS DATAFRAMES DE PARES (mean/cls)
# =====================
def pares_com_infos(df_raw: pd.DataFrame, df_pares: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Monta um DataFrame com a coluna vetorial desejada para cada par,
    preservando a coluna 'split' e a 'label'. Usa acesso posicional.
    """
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
# (C) CONVERSÃO DE VETORES (strings/listas -> np.ndarray)
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
        s = s.replace("\n", " ").strip()
        arr = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
        if arr.size == 0:
            arr = np.array(eval(s), dtype=np.float32)  # usa eval aqui por simplicidade controlada
        return arr
    raise TypeError(f"Tipo não suportado na coluna de vetor: {type(x)}")

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

# =====================
# (D) RELAÇÕES (features) ENTRE VETORES — HIPERPARÂMETRO
# =====================
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

    if relation == "cos":        return cos
    if relation == "eucl":       return eucl
    if relation == "manh":       return manh
    if relation == "cheb":       return cheb
    if relation == "dot":        return dot
    if relation == "proj":       return np.concatenate([proj_v1_em_v2, proj_v2_em_v1], axis=1)
    if relation == "scalars":    return np.concatenate([cos, eucl, manh, cheb, dot], axis=1)

    if relation == "diff":       return diff
    if relation == "absdiff":    return np.abs(diff)
    if relation == "hadamard":   return v1 * v2
    if relation == "v1v2":       return np.concatenate([v1, v2], axis=1)
    if relation == "siamese4":   return np.concatenate([v1, v2, np.abs(diff), v1 * v2], axis=1)

    raise ValueError(f"Relação desconhecida: {relation}")

# =====================
# (E) TUNING COM OPTUNA (CV APENAS NO TRAIN)
# =====================
def treinar_svm_optuna(
    v1_tr: np.ndarray, v2_tr: np.ndarray, y_tr: np.ndarray,
    relations: List[str],
    n_trials: int = 40,
    seed: int = 42,
    scoring: str = "roc_auc",
    n_splits_cv: int = 5,
    n_jobs_cv: int = -1,
    show_progress: bool = True,
):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        relation = trial.suggest_categorical("relation", relations)
        kernel   = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C        = trial.suggest_float("C", 1e-2, 1e3, log=True)
        class_w  = trial.suggest_categorical("class_weight", [None, "balanced"])

        X_tr = build_features(v1_tr, v2_tr, relation)

        if kernel == "linear":
            clf = SVC(kernel="linear", C=C, class_weight=class_w, probability=False, random_state=seed)
        else:
            gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
            clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_w, probability=False, random_state=seed)

        pipe = Pipeline([("scaler", StandardScaler()), ("svm", clf)])
        cv   = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)
        metric = "roc_auc" if scoring == "roc_auc" else "f1"
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=metric, n_jobs=n_jobs_cv)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    best = study.best_params

    # Reajuste final no train inteiro
    X_tr = build_features(v1_tr, v2_tr, best["relation"])
    if best["kernel"] == "linear":
        clf = SVC(kernel="linear", C=best["C"], class_weight=best["class_weight"], probability=False, random_state=seed)
    else:
        clf = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], class_weight=best["class_weight"], probability=False, random_state=seed)

    best_model = Pipeline([("scaler", StandardScaler()), ("svm", clf)])
    best_model.fit(X_tr, y_tr)

    return best_model, best, study

# =====================
# (F) AVALIAÇÃO NO TEST
# =====================
def avaliar_no_test(model: Pipeline, v1_te: np.ndarray, v2_te: np.ndarray, y_te: np.ndarray, relation: str) -> Dict[str, float]:
    X_te = build_features(v1_te, v2_te, relation)
    scores = model.decision_function(X_te)
    y_pred = (scores >= 0).astype(int)
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1":       float(f1_score(y_te, y_pred)),
        "roc_auc":  float(roc_auc_score(y_te, scores)),
        "ap":       float(average_precision_score(y_te, scores)),
    }

# =====================
# (G) ORQUESTRAÇÃO DE UM DF (mean OU cls)
# =====================
def rodar_pipeline_simples(df_pairs: pd.DataFrame, col: str, nome_tag: str):
    (v1_tr, v2_tr, y_tr), (v1_te, v2_te, y_te) = preparar_train_test(df_pairs, col)
    model, best_params, study = treinar_svm_optuna(
        v1_tr, v2_tr, y_tr,
        relations=list(RELATIONS),
        n_trials=N_TRIALS,
        scoring=SCORING_CV,
        seed=SEED,
        n_splits_cv=N_SPLITS_CV,
        show_progress=True,
    )
    test_metrics = avaliar_no_test(model, v1_te, v2_te, y_te, best_params["relation"])

    print(f"\n==== {nome_tag} ====")
    print("MELHORES HIPERPARÂMETROS:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("MÉTRICAS NO TEST:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    write_text(os.path.join(OUT_DIR, f"{nome_tag}_best_params.txt"), json.dumps(best_params, ensure_ascii=False, indent=2))
    write_text(os.path.join(OUT_DIR, f"{nome_tag}_test_metrics.txt"), json.dumps(test_metrics, ensure_ascii=False, indent=2))
    if SAVE_MODELS:
        joblib.dump(model, os.path.join(OUT_DIR, f"svm_{nome_tag}.joblib"))

# =====================
# MAIN
# =====================
def main():
    set_global_seed(SEED)

    # 1) Ler df_raw
    df_raw = read_csv_flexible(RAW_CSV)
    if COL_AUTOR not in df_raw.columns:
        raise ValueError(f"df_raw precisa conter a coluna de autor '{COL_AUTOR}'.")

    # 2) Obter df_idx
    if USE_EXISTING_IDX:
        df_idx = read_csv_flexible(PAIRS_CSV)
        for c in ["i1","i2","label","split"]:
            if c not in df_idx.columns:
                raise ValueError(f"df_idx precisa ter coluna '{c}'.")
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
        # salva df_idx por conveniência (CSV simples)
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        df_idx_path = os.path.join(OUT_DIR, "df_idx_gerado.csv")
        df_idx.to_csv(df_idx_path, index=False)
        print(f"[OK] df_idx salvo em: {df_idx_path}")

    # 3) Montar df_mean e/ou df_cls e rodar pipelines
    if RUN_MEAN:
        if VEC_MEAN_COL not in df_raw.columns:
            raise ValueError(f"Coluna '{VEC_MEAN_COL}' não encontrada em df_raw.")
        df_mean = pares_com_infos(df_raw, df_idx, VEC_MEAN_COL)
        rodar_pipeline_simples(df_mean, VEC_MEAN_COL, "mean")

    if RUN_CLS:
        if VEC_CLS_COL not in df_raw.columns:
            raise ValueError(f"Coluna '{VEC_CLS_COL}' não encontrada em df_raw.")
        df_cls = pares_com_infos(df_raw, df_idx, VEC_CLS_COL)
        rodar_pipeline_simples(df_cls, VEC_CLS_COL, "cls")


if __name__ == "__main__":
    main()
