import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def framecutting(text, len_tokens):
    """
    Quebra um texto em blocos de len_tokens palavras.
    Mantém o máximo de blocos completos.
    
    Args:
        text (str): Texto original.
        len_tokens (int): Quantidade de tokens (palavras) por bloco.
    
    Returns:
        list[str]: Lista de trechos com len_tokens palavras (último
                   trecho incompleto é descartado).
    """
    tokens_list = text.split()
    n_t = len(tokens_list)

    # número de trechos completos
    n_chunks = n_t // len_tokens  

    frames = []
    for i in range(n_chunks):
        start = i * len_tokens
        end = start + len_tokens
        chunk = tokens_list[start:end]
        frames.append(" ".join(chunk))

    return frames


def cut_all(df, len_tokens, save=True ,output_csv=None):
    """
    Aplica framecutting em todos os textos de df['texto'] e gera
    um novo DataFrame com trechos cortados, repetindo as demais infos.

    Args:
        df (pd.DataFrame): DataFrame original que tem coluna 'texto'.
        len_tokens (int): nº de tokens por trecho.
        output_csv (str): caminho para salvar o CSV final.

    Returns:
        pd.DataFrame: novo DataFrame expandido com trechos cortados.
    """
    rows = []
    for idx, row in df.iterrows():
        text = row["texto"]
        chunks = framecutting(text, len_tokens)
        for i, chunk in enumerate(chunks):
            new_row = row.to_dict()
            new_row["texto"] = chunk
            new_row["chunk_id"] = i  # opcional: id do trecho dentro do texto original
            rows.append(new_row)

    df_new = pd.DataFrame(rows)
    df_new.to_csv(output_csv, index=False)
    return df_new

def parear_duplas_balanceado(df, output_csv=None):
    """
    Gera pares de trechos de texto balanceados (50% mesmo autor, 50% autores diferentes).
    
    Args:
        df (pd.DataFrame): DataFrame contendo pelo menos as colunas 'texto' e 'autor'.
        output_csv (str): Caminho de arquivo CSV para salvar os pares (opcional).
    
    Returns:
        pd.DataFrame: DataFrame com colunas ['texto1', 'texto2', 'autor1', 'autor2', 'label'],
                      onde label = 1 se os autores são os mesmos, ou 0 se são diferentes.
    """
    same_pairs = []
    diff_pairs = []
    # Dicionário para mapear autor -> lista de índices de seus textos
    author_to_indices = {}
    for author, group in df.groupby("autor"):
        indices = list(group.index)
        author_to_indices[author] = indices
        # Gera todos os pares (combinações) de textos do mesmo autor
        if len(indices) >= 2:
            for a in range(len(indices)):
                for b in range(a+1, len(indices)):
                    i = indices[a]
                    j = indices[b]
                    same_pairs.append((i, j))
    # Gera pares de textos de autores diferentes
    authors = sorted(author_to_indices.keys())
    for x in range(len(authors)):
        for y in range(x+1, len(authors)):
            auth_i = authors[x]
            auth_j = authors[y]
            # Para cada par de autores distintos, combina todos os textos de um com todos do outro
            for i in author_to_indices[auth_i]:
                for j in author_to_indices[auth_j]:
                    diff_pairs.append((i, j))
    # Quantidade de pares possíveis em cada categoria
    same_count = len(same_pairs)
    diff_count = len(diff_pairs)
    if same_count == 0 or diff_count == 0:
        # Se não há como formar pares de um dos tipos, retorna DataFrame vazio
        result_df = pd.DataFrame(columns=["texto1", "texto2", "autor1", "autor2", "label"])
        if output_csv:
            result_df.to_csv(output_csv, index=False)
        return result_df
    # Balanceamento: limitamos a quantidade de cada tipo ao menor dos dois valores
    if same_count > diff_count:
        # Muitos pares de mesmo autor - reduz para igualar aos de autores diferentes
        random.shuffle(same_pairs)                     # embaralha para seleção aleatória
        same_pairs = same_pairs[:diff_count]
    elif diff_count > same_count:
        # Muitos pares de autores diferentes - reduz para igualar aos de mesmo autor
        random.shuffle(diff_pairs)
        diff_pairs = diff_pairs[:same_count]
    # Monta lista final de pares com dados completos
    final_pairs = []
    for i, j in same_pairs:
        final_pairs.append({
            "texto1": df.loc[i, "texto"], 
            "texto2": df.loc[j, "texto"],
            "autor1": df.loc[i, "autor"], 
            "autor2": df.loc[j, "autor"],
            "label": 1  # mesmo autor
        })
    for i, j in diff_pairs:
        final_pairs.append({
            "texto1": df.loc[i, "texto"], 
            "texto2": df.loc[j, "texto"],
            "autor1": df.loc[i, "autor"], 
            "autor2": df.loc[j, "autor"],
            "label": 0  # autores diferentes
        })
    # Embaralha a lista final para misturar pares 0 e 1 de forma aleatória
    random.shuffle(final_pairs)
    result_df = pd.DataFrame(final_pairs)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
    return result_df

def parear_duplas_bow_cosine(df, output_csv=None):
    """
    Gera pares de trechos de texto balanceados, priorizando aqueles com conteúdo semelhante.
    
    Usa Bag-of-Words (TF-IDF) para calcular similaridade de cosseno entre textos e seleciona
    pares de maior similaridade, mantendo 50% de cada classe (mesmo autor vs autores diferentes).
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'texto' e 'autor'.
        output_csv (str): Caminho de arquivo CSV para salvar o resultado (opcional).
    
    Returns:
        pd.DataFrame: DataFrame com colunas ['texto1', 'texto2', 'autor1', 'autor2', 'label'].
    """
    # Vetorização dos textos em TF-IDF (Bag-of-Words com normalização L2)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["texto"].astype(str).tolist())
    # Mapeia índice do DataFrame -> posição correspondente na matriz X
    idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
    
    same_candidates = []
    diff_candidates = []
    # Dicionário autor -> índices dos textos
    author_to_indices = {author: list(group.index) for author, group in df.groupby("autor")}
    
    # Calcula similaridades para pares de mesmo autor
    for author, indices in author_to_indices.items():
        if len(indices) < 2:
            continue
        # Submatriz de vetores TF-IDF dos textos desse autor
        positions = [idx_to_pos[i] for i in indices]
        X_group = X[positions]
        sim_matrix = linear_kernel(X_group, X_group)  # similaridades coseno entre cada par desse autor
        # Armazena todos os pares (i, j) com sua similaridade
        L = len(indices)
        for a in range(L):
            for b in range(a+1, L):
                i = indices[a]
                j = indices[b]
                sim_val = sim_matrix[a, b]
                same_candidates.append((i, j, sim_val))
    # Calcula similaridades para pares de autores diferentes
    authors = sorted(author_to_indices.keys())
    for x in range(len(authors)):
        for y in range(x+1, len(authors)):
            auth_i = authors[x]
            auth_j = authors[y]
            indices_i = author_to_indices[auth_i]
            indices_j = author_to_indices[auth_j]
            if not indices_i or not indices_j:
                continue
            # Submatrizes TF-IDF para cada autor
            pos_i = [idx_to_pos[i] for i in indices_i]
            pos_j = [idx_to_pos[j] for j in indices_j]
            X_i = X[pos_i]
            X_j = X[pos_j]
            sim_matrix = linear_kernel(X_i, X_j)  # similaridades coseno entre todos textos de auth_i vs auth_j
            for a in range(len(indices_i)):
                for b in range(len(indices_j)):
                    i = indices_i[a]
                    j = indices_j[b]
                    sim_val = sim_matrix[a, b]
                    diff_candidates.append((i, j, sim_val))
    # Quantidade de candidatos de cada tipo
    same_count = len(same_candidates)
    diff_count = len(diff_candidates)
    if same_count == 0 or diff_count == 0:
        # Retorna vazio se não houver pelo menos um par de cada tipo
        result_df = pd.DataFrame(columns=["texto1", "texto2", "autor1", "autor2", "label"])
        if output_csv:
            result_df.to_csv(output_csv, index=False)
        return result_df
    # Ordena candidatos por similaridade (maior primeiro)
    same_candidates.sort(key=lambda x: x[2], reverse=True)
    diff_candidates.sort(key=lambda x: x[2], reverse=True)
    # Seleciona pares balanceados priorizando maior similaridade
    if same_count > diff_count:
        # Existem mais pares de mesmo autor que de autores diferentes
        same_selected = same_candidates[:diff_count]    # pega os pares de mesmo autor mais similares
        diff_selected = diff_candidates                 # pega todos de autores diferentes
    elif diff_count > same_count:
        # Existem mais pares de autores diferentes
        diff_selected = diff_candidates[:same_count]    # pega os pares de autores diferentes mais similares
        same_selected = same_candidates                 # pega todos de mesmo autor
    else:
        # Quantidade igual
        same_selected = same_candidates
        diff_selected = diff_candidates
    # Garante que ambos têm exatamente o mesmo número de pares
    pairs_per_class = min(len(same_selected), len(diff_selected))
    same_selected = same_selected[:pairs_per_class]
    diff_selected = diff_selected[:pairs_per_class]
    # Monta lista final de pares (com textos e autores)
    final_pairs = []
    for i, j, sim in same_selected:
        final_pairs.append({
            "texto1": df.loc[i, "texto"],
            "texto2": df.loc[j, "texto"],
            "autor1": df.loc[i, "autor"],
            "autor2": df.loc[j, "autor"],
            "label": 1  # mesmo autor
        })
    for i, j, sim in diff_selected:
        final_pairs.append({
            "texto1": df.loc[i, "texto"],
            "texto2": df.loc[j, "texto"],
            "autor1": df.loc[i, "autor"],
            "autor2": df.loc[j, "autor"],
            "label": 0  # autores diferentes
        })
    # Embaralha para misturar classes de forma aleatória (opcional)
    random.shuffle(final_pairs)
    result_df = pd.DataFrame(final_pairs)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
    return result_df


df = pd.read_csv('full_txt_author.csv', index_col=0)

df_cut = cut_all(df,510,output_csv='full_txt_author_510t.csv')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1) escolher modelo e dispositivo (GPU se existir)
MODEL  = "neuralmind/bert-base-portuguese-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) carregar tokenizer e backbone (sem "cabeça" de tarefa)
tok = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)
enc = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

# 3) função de mean pooling (média mascarada dos tokens válidos)
def mean_pool(last_hidden_state, attention_mask):
    mask   = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,L,1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B,H)
    count  = mask.sum(dim=1).clamp(min=1e-9)                          # (B,1)
    return summed / count                                             # (B,H)

# 4) função de vetorização em BATCH (um vetor por texto)
@torch.inference_mode()
def embed(
    texts,
    pooling="mean",
    normalize=True,
    max_length=512,
    batch_size=16,
    fp16=True,
):
    """
    texts: list[str] ou str único
    pooling: "mean" ou "cls"
    retorna: torch.Tensor (N, 768) no CPU
    """
    if isinstance(texts, str):
        texts = [texts]

    outputs = []
    use_fp16 = (DEVICE == "cuda") and fp16

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        batch = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        if use_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = enc(**batch)
        else:
            out = enc(**batch)

        if pooling == "mean":
            vec = mean_pool(out.last_hidden_state.float(), batch["attention_mask"])
        elif pooling == "cls":
            # usa pooler_output se existir; senão o token [CLS] cru
            has_pooler = hasattr(out, "pooler_output") and out.pooler_output is not None
            vec = (out.pooler_output if has_pooler else out.last_hidden_state[:, 0, :]).float()
        else:
            raise ValueError("pooling deve ser 'mean' ou 'cls'")

        if normalize:
            vec = F.normalize(vec, p=2, dim=1)

        outputs.append(vec.cpu())

        # higiene de memória entre lotes
        del out, batch, vec
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)

# 5) demonstração rápida (mesmo uso de antes, agora batched)
texts = [
    "I like to drink coofee",
    "Havia uma rocha no caminho."
]
v_mean = embed(texts, pooling="mean", batch_size=16)
v_cls  = embed(texts, pooling="cls",  batch_size=16)

print("shape (mean):", v_mean.shape)  # ex.: torch.Size([2, 768])
print("shape (cls): ", v_cls.shape)

sim_mean = F.cosine_similarity(v_mean[0], v_mean[1], dim=0).item()
sim_cls  = F.cosine_similarity(v_cls[0],  v_cls[1],  dim=0).item()
print(f"cosine(mean) = {sim_mean:.3f}")
print(f"cosine(cls)  = {sim_cls:.3f}")


lista_textos = df_cut['texto'].tolist()
vec_cls = embed(lista_textos, pooling="mean", batch_size=16)
vec_mean = embed(lista_textos, pooling="cls", batch_size=16)

df_cut["vec_mean"] = vec_mean.tolist()
df_cut["vec_cls"]  = vec_cls.tolist()

df_cut.to_csv("df_com_embeddings.csv", index=False)
print("CSV salvo em df_com_embeddings.csv")
