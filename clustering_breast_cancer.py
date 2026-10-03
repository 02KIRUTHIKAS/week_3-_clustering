"""
Week 3 - Unsupervised Learning and Clustering Analysis
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn
Run: python clustering_breast_cancer.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid", palette="Set2")
FIG = "figures/"
PAL3 = {0: "#66c2a5", 1: "#fc8d62", 2: "#8da0cb"}

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
mean_cols = [c for c in df.columns if c.startswith("mean ")]
X = df[mean_cols].copy()
print("Feature matrix shape:", X.shape)
print("Missing values:", X.isna().sum().sum())

# ------------------------------------------------------------------
# 2. Preprocessing: standardize
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean after scaling (should be ~0):", X_scaled.mean().round(3))
print("Std after scaling (should be ~1):", X_scaled.std().round(3))

# ------------------------------------------------------------------
# 3. Choose k: elbow method + silhouette score
# ------------------------------------------------------------------
inertias, sils = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}  inertia={km.inertia_:.1f}  silhouette={sils[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(list(K_range), inertias, marker="o", color="#4c72b0")
axes[0].set_title("Figure 1: Elbow method")
axes[0].set_xlabel("Number of clusters (k)"); axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[1].plot(list(K_range), sils, marker="o", color="#dd8452")
axes[1].set_title("Figure 2: Silhouette score vs k")
axes[1].set_xlabel("Number of clusters (k)"); axes[1].set_ylabel("Average silhouette score")
plt.tight_layout(); plt.savefig(FIG + "fig1_fig2_k_selection.png", dpi=150); plt.close()

best_k = list(K_range)[int(np.argmax(sils))]
print("Best k by silhouette:", best_k)
k = 2  # chosen: matches the 2 natural diagnosis groups; also near-top silhouette

# ------------------------------------------------------------------
# 4. K-Means with chosen k
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)
df["kmeans_cluster"] = km_labels
sil_km = silhouette_score(X_scaled, km_labels)
print("K-Means silhouette (k=2):", round(sil_km, 3))
print(pd.crosstab(df["kmeans_cluster"], df["diagnosis"]))
print("Adjusted Rand Index vs true diagnosis:", round(adjusted_rand_score(df["diagnosis"], km_labels), 3))

# ------------------------------------------------------------------
# 5. PCA for 2D visualization
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_.round(3),
      "total:", round(pca.explained_variance_ratio_.sum(), 3))
df["pc1"], df["pc2"] = X_pca[:, 0], X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="kmeans_cluster", palette=PAL3, alpha=0.75, ax=axes[0])
axes[0].set_title("Figure 3a: K-Means clusters (PCA projection)")
sns.scatterplot(data=df, x="pc1", y="pc2", hue="diagnosis", palette={"Benign": "#66c2a5", "Malignant": "#fc8d62"}, alpha=0.75, ax=axes[1])
axes[1].set_title("Figure 3b: True diagnosis (PCA projection)")
for ax in axes: ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout(); plt.savefig(FIG + "fig3_pca_clusters.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 6. Silhouette plot (per-sample)
# ------------------------------------------------------------------
sample_sil = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
for i in range(k):
    vals = np.sort(sample_sil[km_labels == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=PAL3[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * len(vals), str(i))
    y_lower = y_upper + 10
ax.axvline(sil_km, color="red", linestyle="--", label=f"Average = {sil_km:.2f}")
ax.set_title("Figure 4: Silhouette plot for K-Means (k=2)")
ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Samples per cluster"); ax.legend()
plt.tight_layout(); plt.savefig(FIG + "fig4_silhouette.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 7. Hierarchical clustering (Agglomerative, Ward linkage)
# ------------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
df["hier_cluster"] = agg_labels
sil_agg = silhouette_score(X_scaled, agg_labels)
print("Hierarchical silhouette (k=2):", round(sil_agg, 3))
print(pd.crosstab(df["hier_cluster"], df["diagnosis"]))
print("Agreement between KMeans and Hierarchical (ARI):", round(adjusted_rand_score(km_labels, agg_labels), 3))

# Dendrogram on a sample (full 569 is too dense)
rng = np.random.RandomState(42)
idx = rng.choice(len(X_scaled), 40, replace=False)
Z = linkage(X_scaled[idx], method="ward")
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, ax=ax, color_threshold=None)
ax.set_title("Figure 5: Dendrogram (Ward linkage, 40-sample subset)")
ax.set_xlabel("Sample index"); ax.set_ylabel("Distance")
plt.tight_layout(); plt.savefig(FIG + "fig5_dendrogram.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 8. Cluster profiling
# ------------------------------------------------------------------
profile = df.groupby("kmeans_cluster")[mean_cols].mean().T.round(3)
profile.columns = [f"Cluster {c}" for c in profile.columns]
print(profile)
profile.to_csv("cluster_profile.csv")

fig, ax = plt.subplots(figsize=(9, 5))
scaled_means = pd.DataFrame(X_scaled, columns=mean_cols)
scaled_means["cluster"] = km_labels
hm = scaled_means.groupby("cluster").mean().T
sns.heatmap(hm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "Mean (standardized)"})
ax.set_title("Figure 6: Cluster feature profile (standardized means)")
ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
plt.tight_layout(); plt.savefig(FIG + "fig6_cluster_profile.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
sizes = df["kmeans_cluster"].value_counts().sort_index()
ax.bar(sizes.index.astype(str), sizes.values, color=[PAL3[i] for i in sizes.index])
for i, v in enumerate(sizes.values):
    ax.text(i, v + 5, str(v), ha="center")
ax.set_title("Figure 7: Cluster sizes")
ax.set_xlabel("Cluster"); ax.set_ylabel("Number of samples")
plt.tight_layout(); plt.savefig(FIG + "fig7_cluster_sizes.png", dpi=150); plt.close()

print("\nDone.")
