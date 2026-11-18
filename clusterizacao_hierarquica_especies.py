# %% [markdown]
# # Desafio: Clusterização Hierárquica de espécies fictícias
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

N_CLUSTERS = 10   # o desafio pede pelo menos 10 clusters

# %% 1. Carregar e explorar
df = pd.read_csv("species_dataset.csv")
print(df.shape, df.isna().sum().sum(), "ausentes,", df["species_id"].duplicated().sum(), "ids duplicados")
print(df.describe().T.round(2))
for c in ["diet_type", "skin_type", "social_behavior"]:
    print(df[c].value_counts(normalize=True).round(3))
num = ["body_mass_kg", "tail_length_cm", "avg_lifespan_years", "num_legs", "eye_count"]
bin_ = ["has_wings", "nocturnal", "has_venom"]
cat = ["diet_type", "skin_type", "social_behavior"]
print(df[num + bin_].corr().abs().where(lambda m: m < 1).max().round(2))   # correlações entre atributos

fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for a, c in zip(ax.ravel(), num + bin_):
    df[c].hist(ax=a, bins=20); a.set_title(c, fontsize=9)
plt.tight_layout(); plt.savefig("exploracao.png", dpi=110); plt.close()

# %% 2. Preparação: numéricas padronizadas, booleanas como 0/1 e categóricas em one-hot
X = pd.concat([
    pd.DataFrame(StandardScaler().fit_transform(df[num]), columns=num),
    df[bin_],
    pd.get_dummies(df[cat]).astype(int)], axis=1)
print("Matriz final:", X.shape)

# %% 3. Algoritmos de clusterização hierárquica: comparação dos métodos de ligação
dist = pdist(X)
Z, resumo = {}, []
for m in ["ward", "average", "complete", "single"]:
    Z[m] = linkage(X, m)
    r = fcluster(Z[m], N_CLUSTERS, "maxclust")
    resumo.append({"ligação": m, "cofenética": cophenet(Z[m], dist)[0],
                   f"silhueta (k={N_CLUSTERS})": silhouette_score(X, r),
                   "maior cluster (%)": np.bincount(r).max() / len(r) * 100})
print(pd.DataFrame(resumo).round(3).to_string(index=False))
# Ward é usado como modelo principal: gera clusters mais equilibrados e compactos (single/average formam um cluster gigante).

# %% 4. Dendrogramas
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for a, m in zip(ax.ravel(), Z):
    dendrogram(Z[m], truncate_mode="lastp", p=30, ax=a, no_labels=True)
    a.set_title(f"Dendrograma — ligação {m} (30 últimos agrupamentos)")
plt.tight_layout(); plt.savefig("dendrogramas.png", dpi=110); plt.close()

# Corte do Ward para 10 clusters: a linha fica entre as duas últimas fusões que levam de 10 a 9 clusters
corte = (Z["ward"][-N_CLUSTERS, 2] + Z["ward"][-N_CLUSTERS + 1, 2]) / 2
plt.figure(figsize=(13, 5))
dendrogram(Z["ward"], truncate_mode="lastp", p=40, no_labels=True, color_threshold=corte)
plt.axhline(corte, color="k", ls="--", label=f"corte → {N_CLUSTERS} clusters"); plt.legend()
plt.title("Dendrograma Ward com corte"); plt.tight_layout(); plt.savefig("dendrograma_ward_corte.png", dpi=110); plt.close()

# %% 5. Corte em 10 clusters e silhueta por número de clusters
df["cluster"] = fcluster(Z["ward"], N_CLUSTERS, "maxclust")
print(df["cluster"].value_counts().sort_index())
print({k: round(silhouette_score(X, fcluster(Z["ward"], k, "maxclust")), 3) for k in range(2, 16)})

# %% 6. Perfil dos grupos
perfil = df.groupby("cluster")[num].mean().join(df.groupby("cluster")[bin_].mean())
for c in cat:
    perfil = perfil.join(pd.crosstab(df["cluster"], df[c], normalize="index"))
perfil.insert(0, "n", df["cluster"].value_counts().sort_index())
print(perfil.round(2).T.to_string())

# diferença de cada grupo para a média geral, em desvios-padrão (z), para achar o que mais distingue cada grupo
cols = num + bin_ + list(pd.get_dummies(df[cat]).columns)
Xd = pd.concat([df[num + bin_], pd.get_dummies(df[cat]).astype(int)], axis=1)
z = (Xd.groupby(df["cluster"]).mean() - Xd.mean()) / Xd.std()
for c in z.index:
    top = z.loc[c].reindex(z.loc[c].abs().sort_values(ascending=False).index).head(3)
    print(f"Cluster {c} (n={int((df['cluster']==c).sum())}):", ", ".join(f"{k} ({v:+.2f})" for k, v in top.items()))

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(z.T, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(z))); ax.set_xticklabels(z.index); ax.set_yticks(range(len(z.columns))); ax.set_yticklabels(z.columns, fontsize=8)
ax.set_xlabel("cluster"); ax.set_title("Perfil dos clusters (desvios-padrão em relação à média geral)"); plt.colorbar(im)
plt.tight_layout(); plt.savefig("perfil_clusters.png", dpi=110); plt.close()

# %% 7. Estabilidade: os clusters mudam com outro método de ligação?
for m in ["complete", "average"]:
    print(f"ARI Ward x {m} (k={N_CLUSTERS}):", round(adjusted_rand_score(df["cluster"], fcluster(Z[m], N_CLUSTERS, "maxclust")), 3))
df.to_csv("species_com_clusters.csv", index=False)

# %% [markdown]
# ## 8. Análise dos resultados
# **Dados.** 600 espécies, 11 atributos, sem valores ausentes. As correlações entre atributos numéricos/booleanos são todas fracas (≤ 0,09) e as
# categorias são bem equilibradas (cada uma entre ~24% e 37%). `avg_lifespan_years` varia pouco (média 120, desvio ≈ 1), mas foi padronizado como as demais.
#
# **Método.** Atributos numéricos padronizados, booleanos como 0/1 e categóricos em one-hot (18 colunas). Comparei as ligações Ward, average, complete e single.
# Single e average formam um cluster gigante (98% e 72% das espécies), então não servem para um corte em 10 grupos úteis; Ward deu os grupos mais equilibrados
# (maior cluster com ~16%) e foi usado como modelo principal. O corte do dendrograma em 10 clusters tem de 39 a 94 espécies cada.
#
# **Grupos encontrados (Ward, k=10)** — o que mais distingue cada um (desvios-padrão em relação à média geral):
# | Cluster | n | Perfil |
# |---|---|---|
# | 1 | 43 | Maior massa corporal (média ≈ 78 kg), mais olhos (~3), maior longevidade; 63% vivem em grupo |
# | 2 | 89 | Todas com 4 olhos |
# | 3 | 54 | Muitas patas (média ≈ 6,9), cauda curta, dieta onívora (52%) |
# | 4 | 94 | Muitas patas (≈ 6), cauda longa, menor massa |
# | 5 | 43 | Poucas patas (≈ 1,6), vida em pares (77%), onívoras (70%), maior longevidade |
# | 6 | 62 | Cauda mais curta, poucas patas, herbívoras (58%) |
# | 7 | 54 | Nenhum olho (0), herbívoras (57%) |
# | 8 | 73 | Menor longevidade (≈ 118,9), cauda curta |
# | 9 | 39 | Menor massa (≈ 31 kg), poucas patas (≈ 1,3), sem penas, herbívoras (64%) |
# | 10 | 49 | Cauda mais longa (≈ 56,5 cm), mais massa, carnívoras (53%) |
#
# **Relações "evolutivas" simuladas (leitura do dendrograma).** A primeira divisão separa as espécies com mais olhos (clusters 1 e 2, 132 espécies)
# das demais. Depois, os clusters 3 e 4 (muitas patas) se juntam, 5 e 6 (poucas patas) se juntam e 7, 8, 9 e 10 formam o ramo restante. Ou seja, número de olhos e
# de patas são os eixos que organizam a árvore; massa, cauda e longevidade refinam os grupos. Na biologia real essas características seriam sinais de
# parentesco, mas aqui a árvore mede só semelhança entre atributos, então é um dendrograma fenético, não uma filogenia.
#
# **Limitações importantes.** A estrutura de grupos é fraca: a silhueta fica em ≈ 0,04 com 10 clusters (a melhor, 0,108, é com k=2) e o resultado é instável:
# o ARI entre Ward e complete é 0,27 e entre Ward e average, 0,10. Isso é coerente com atributos praticamente independentes, como nos dados simulados:
# os clusters são, em boa parte, fatias do espaço de atributos (por exemplo, "0 olhos" e "4 olhos" são categorias discretas), e não grupos naturais bem separados.
# Os perfis acima devem ser lidos como descrição das diferenças, não como espécies que "pertencem" a famílias distintas.
#
# **Melhorias possíveis.** Usar distância de Gower, que trata variáveis mistas melhor que one-hot com padronização; testar pesos por atributo
# (ex.: reduzir o peso de `avg_lifespan_years`); avaliar estabilidade com bootstrap; e comparar com K-Means ou Gaussian Mixture.
