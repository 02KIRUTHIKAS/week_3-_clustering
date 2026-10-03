# Unsupervised Learning and Clustering Analysis - Breast Cancer Wisconsin (Diagnostic)

Week 3 task of the Yuva Intern *Data Science with Python* virtual internship.

## Goal
Apply clustering (K-Means and Hierarchical) to a public dataset to find meaningful
groups without using the diagnosis label, then interpret the clusters.

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (569 samples, reused from
Week 2). Loaded via `sklearn.datasets.load_breast_cancer`.

## What the script does
1. Loads the 10 "mean" features and standardizes them (StandardScaler)
2. Chooses k using the elbow method and silhouette score (k=2 selected)
3. Fits K-Means (k=2) and Agglomerative/Hierarchical (Ward linkage, k=2)
4. Visualizes clusters via PCA, a silhouette plot, and a dendrogram
5. Profiles each cluster's average feature values and compares clusters to
   the true diagnosis (held out, used only for evaluation)
6. Saves seven annotated figures to `figures/` and the profile to `cluster_profile.csv`

## Run
```
pip install -r requirements.txt
python clustering_breast_cancer.py
```

## Key findings
- k=2 is best by both elbow and silhouette score (0.395)
- K-Means clusters agree strongly with the real diagnosis (ARI = 0.646);
  Cluster 0 is 87.8% benign, Cluster 1 is 96.4% malignant
- Cluster 1 (malignant-leaning) has ~1.5x radius, ~2.1x area, 2-3.5x the
  concavity/concave points of Cluster 0
- K-Means (silhouette 0.395) separated the data better than Hierarchical (0.333)
  on this dataset; agreement between the two methods: ARI = 0.619

Full write-up: `Week3_Clustering_Report.docx`
