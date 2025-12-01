import numpy as np
import pandas as pd
from collections import defaultdict

def gerar_pares_balanceados_indices_split(
    df: pd.DataFrame,
    n_pairs_total: int,
    col_autor: str = "autor",
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Gera pares balanceados (50/50) de índices + label, com split por autor (treino/teste disjuntos).

    Args:
        df: DataFrame com ao menos a coluna col_autor.
        n_pairs_total: número total de pares desejado (será arredondado para par).
        col_autor: nome da coluna com o autor.
        test_size: fração de autores para teste.
        seed: semente para reprodutibilidade.

    Returns:
        DataFrame com colunas: i1 (int32), i2 (int32), label (uint8), split ('train'/'test').
    """
    rng = np.random.default_rng(seed)

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    autores = df[col_autor].values
    if len(df) < 2:
        return pd.DataFrame(columns=["i1","i2","label","split"])

    # --- split por autor
    autores_unicos = np.array(pd.unique(autores))
    rng.shuffle(autores_unicos)
    n_test_aut = max(1, int(round(len(autores_unicos) * test_size)))
    autores_test = set(autores_unicos[:n_test_aut])
    autores_train = set(autores_unicos[n_test_aut:]) if n_test_aut < len(autores_unicos) else set()

    mask_train = np.array([a in autores_train for a in autores])
    mask_test  = np.array([a in autores_test  for a in autores])

    # --- proporção
    n_pairs_total = int(n_pairs_total // 2 * 2)
    n_pairs_train = int((1.0 - test_size) * n_pairs_total)
    n_pairs_test  = n_pairs_total - n_pairs_train

    def amostrar_pares_subset(mask_subset: np.ndarray, n_pairs_target: int):
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

        seen = set()
        rows = []

        # positivos
        if n_pos_target > 0 and autores_pos:
            while len(rows) < n_pos_target:
                a = rng.choice(autores_pos, p=p_pos)
                L = author_to_idx[a]
                if len(L) < 2: 
                    continue
                i1, i2 = rng.choice(L, size=2, replace=False)
                i_min, i_max = sorted((i1, i2))
                key = (i_min, i_max, 1)
                if key in seen: 
                    continue
                seen.add(key)
                rows.append((i_min, i_max, 1))

        # negativos
        while len(rows) < n_pairs_target:
            a1, a2 = rng.choice(autores_todos, size=2, replace=False, p=p_all)
            i1 = rng.choice(author_to_idx[a1])
            i2 = rng.choice(author_to_idx[a2])
            if i1 == i2: 
                continue
            i_min, i_max = sorted((i1, i2))
            key = (i_min, i_max, 0)
            if key in seen: 
                continue
            seen.add(key)
            rows.append((i_min, i_max, 0))

        return rows

    rows_train = amostrar_pares_subset(mask_train, n_pairs_train)
    rows_test  = amostrar_pares_subset(mask_test,  n_pairs_test)

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


df_raw = pd.read_csv('../df_com_embeddings.csv')

df_idx = gerar_pares_balanceados_indices_split(df_raw, 10000, seed=41)

def pares_com_infos(df, df_pares, coluna: str) -> pd.DataFrame:
    """
    Monta um DataFrame com a informação desejada de cada par,
    preservando também o split (train/test).

    Args:
        df (pd.DataFrame): DataFrame original (com colunas como 'texto', 'autor', etc.).
        df_pares (pd.DataFrame): DataFrame com pares ('i1', 'i2', 'label', 'split').
        coluna (str): Nome da coluna a extrair do df.

    Returns:
        pd.DataFrame: DataFrame com col1, col2, label e split.
    """
    col1 = df.loc[df_pares['i1'], coluna].reset_index(drop=True)
    col2 = df.loc[df_pares['i2'], coluna].reset_index(drop=True)

    out = pd.DataFrame({
        f"{coluna}_1": col1,
        f"{coluna}_2": col2,
        "label": df_pares["label"].reset_index(drop=True),
        "split": df_pares["split"].reset_index(drop=True)
    })
    return out

df_mean = pares_com_infos(df_raw, df_idx, 'vec_mean')
df_cls = pares_com_infos(df_raw, df_idx, 'vec_cls')

# Requisitos:
# pip install optuna scikit-learn numpy pandas

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import optuna

# -------------------------------
# Utilitários de dados
# -------------------------------
import numpy as np
import pandas as pd
import ast, re
from typing import Tuple

# --- parser robusto para célula -> np.ndarray(float32)
_array_pat = re.compile(r'^array\((.*)\)$', re.IGNORECASE|re.DOTALL)

def _to_float_vector(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)

    if isinstance(x, str):
        s = x.strip()
        # caso "array([..], dtype=float32)" -> extrai o conteúdo entre [..]
        if s.lower().startswith("array("):
            li, ri = s.find("["), s.rfind("]")
            if li != -1 and ri != -1 and ri > li:
                s = s[li:ri+1]
        s = s.replace("\n", " ").strip()

        # 1) tenta separar por vírgula; 2) por espaço; 3) literal_eval
        arr = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
        if arr.size == 0:
            try:
                arr = np.array(ast.literal_eval(s), dtype=np.float32)
            except Exception:
                raise ValueError(f"Não consegui converter string para vetor: {s[:80]}...")
        return arr

    raise TypeError(f"Tipo não suportado na coluna de vetor: {type(x)}")

# --- NOVA versão
def _stack_vectors(df_pairs: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lê f'{col}_1', f'{col}_2' e 'label' e retorna (v1, v2, y) numéricos.
    Coage strings/listas para np.ndarray(float32) e descarta linhas inválidas.
    """
    v1_list, v2_list, y_list = [], [], []
    col1 = df_pairs[f"{col}_1"].values
    col2 = df_pairs[f"{col}_2"].values
    ycol = df_pairs["label"].values

    for a, b, y in zip(col1, col2, ycol):
        try:
            va = _to_float_vector(a)
            vb = _to_float_vector(b)
            if va.shape != vb.shape:
                # se dimensões não batem, pula a amostra
                continue
            if not (np.isfinite(va).all() and np.isfinite(vb).all()):
                continue
            v1_list.append(va)
            v2_list.append(vb)
            y_list.append(int(y))
        except Exception:
            # célula malformada -> ignora
            continue

    if not v1_list:
        raise ValueError("Nenhuma amostra válida após conversão dos vetores.")

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

# -------------------------------
# Relações (features) entre vetores — HIPERPARÂMETRO
# -------------------------------
def build_features(v1: np.ndarray, v2: np.ndarray, relation: str) -> np.ndarray:
    """
    Gera features a partir de pares (v1, v2) de acordo com 'relation'.
    Opções:
      - 'cos'           : similaridade cosseno (1D)
      - 'eucl'          : distância L2 (1D)
      - 'manh'          : distância L1 (1D)
      - 'cheb'          : L-inf (1D)
      - 'dot'           : produto interno (1D)
      - 'proj'          : [proj_v1_em_v2, proj_v2_em_v1] (2D)
      - 'scalars'       : [cos, eucl, manh, cheb, dot] (5D)
      - 'diff'          : (v1 - v2) (D)
      - 'absdiff'       : |v1 - v2| (D)
      - 'hadamard'      : v1 * v2 (D)
      - 'v1v2'          : concatenação [v1, v2] (2D)
      - 'siamese4'      : concat [v1, v2, |v1-v2|, v1*v2] (4D)
    """
    eps = 1e-12
    diff = v1 - v2
    # escalares úteis
    v1n = np.linalg.norm(v1, axis=1, keepdims=True) + eps
    v2n = np.linalg.norm(v2, axis=1, keepdims=True) + eps
    cos = ((v1 * v2).sum(axis=1, keepdims=True)) / (v1n * v2n)
    eucl = np.linalg.norm(diff, axis=1, keepdims=True)
    manh = np.abs(diff).sum(axis=1, keepdims=True)
    cheb = np.abs(diff).max(axis=1, keepdims=True)
    dot  = (v1 * v2).sum(axis=1, keepdims=True)
    # projeções escalares (comprimento da projeção)
    proj_v1_em_v2 = dot / v2n
    proj_v2_em_v1 = dot / v1n

    if relation == "cos":
        return cos
    if relation == "eucl":
        return eucl
    if relation == "manh":
        return manh
    if relation == "cheb":
        return cheb
    if relation == "dot":
        return dot
    if relation == "proj":
        return np.concatenate([proj_v1_em_v2, proj_v2_em_v1], axis=1)
    if relation == "scalars":
        return np.concatenate([cos, eucl, manh, cheb, dot], axis=1)
    if relation == "diff":
        return diff
    if relation == "absdiff":
        return np.abs(diff)
    if relation == "hadamard":
        return v1 * v2
    if relation == "v1v2":
        return np.concatenate([v1, v2], axis=1)
    if relation == "siamese4":
        return np.concatenate([v1, v2, np.abs(diff), v1 * v2], axis=1)

    raise ValueError(f"Relação desconhecida: {relation}")

# -------------------------------
# Tuning com Optuna (apenas no train)
# -------------------------------
def treinar_svm_optuna(
    v1_tr: np.ndarray, v2_tr: np.ndarray, y_tr: np.ndarray,
    relations: List[str] = ("scalars","cos","eucl","manh","cheb","dot","proj","diff","absdiff","hadamard","v1v2","siamese4"),
    n_trials: int = 40,
    seed: int = 42,
    scoring: str = "roc_auc",  # 'roc_auc' ou 'f1'
    n_splits_cv: int = 5,
    n_jobs_cv: int = -1,
    verbose: bool = True,
):
    """
    Otimiza:
      - relation (tipo de relação vetorial usada como feature)
      - kernel ∈ {'linear','rbf'}
      - C (log-uniform)
      - gamma (log-uniform, se kernel='rbf')
      - class_weight ∈ {None, 'balanced'}
    Retorna (best_model, best_params, study).
    """
    def objective(trial: optuna.Trial) -> float:
        relation = trial.suggest_categorical("relation", relations)
        kernel   = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C        = trial.suggest_float("C", 1e-2, 1e3, log=True)
        class_w  = trial.suggest_categorical("class_weight", [None, "balanced"])

        X_tr = build_features(v1_tr, v2_tr, relation)

        if kernel == "linear":
            svm = SVC(kernel="linear", C=C, class_weight=class_w, probability=False, random_state=seed)
        else:
            gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
            svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_w, probability=False, random_state=seed)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", svm),
        ])
        cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)
        score_name = "roc_auc" if scoring == "roc_auc" else "f1"
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=score_name, n_jobs=n_jobs_cv)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Treina final com os melhores hiperparâmetros
    best = study.best_params
    X_tr = build_features(v1_tr, v2_tr, best["relation"])

    if best["kernel"] == "linear":
        best_svm = SVC(kernel="linear", C=best["C"], class_weight=best["class_weight"], probability=False, random_state=seed)
    else:
        best_svm = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], class_weight=best["class_weight"], probability=False, random_state=seed)

    best_model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", best_svm),
    ])
    best_model.fit(X_tr, y_tr)
    return best_model, best, study

# -------------------------------
# Avaliação no TEST
# -------------------------------
def avaliar_no_test(model: Pipeline, v1_te: np.ndarray, v2_te: np.ndarray, y_te: np.ndarray, relation: str) -> Dict[str, float]:
    X_te = build_features(v1_te, v2_te, relation)
    # scores contínuos (para AUC/AP)
    scores = model.decision_function(X_te)
    pred   = (scores >= 0).astype(int)

    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1":       float(f1_score(y_te, pred)),
        "roc_auc":  float(roc_auc_score(y_te, scores)),
        "ap":       float(average_precision_score(y_te, scores)),
    }

# -------------------------------
# Orquestrador simples (um df de cada vez: vec_mean OU vec_cls)
# -------------------------------
def rodar_pipeline_optuna(
    df_pairs: pd.DataFrame,
    col: str,                 # 'vec_mean' OU 'vec_cls'
    n_trials: int = 40,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> Dict[str, any]:
    (v1_tr, v2_tr, y_tr), (v1_te, v2_te, y_te) = preparar_train_test(df_pairs, col)

    model, best_params, study = treinar_svm_optuna(
        v1_tr, v2_tr, y_tr,
        n_trials=n_trials,
        scoring=scoring,
        seed=seed,
        verbose=True,
    )
    test_metrics = avaliar_no_test(model, v1_te, v2_te, y_te, best_params["relation"])

    retorno = {
        "best_params": best_params,
        "test_metrics": test_metrics,
        "model": model,
        "best_relation": best_params["relation"],
        "study": study,
    }
    # print rápido
    print("\n=== MELHORES HIPERPARÂMETROS ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("\n=== MÉTRICAS NO TEST ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    return retorno

# Para o conjunto com mean:
res_mean = rodar_pipeline_optuna(df_mean, col="vec_mean", n_trials=50, scoring="roc_auc", seed=41)

# Para o conjunto com cls:
res_cls = rodar_pipeline_optuna(df_cls, col="vec_cls", n_trials=50, scoring="roc_auc", seed=41)
