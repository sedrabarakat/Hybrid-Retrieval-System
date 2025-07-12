
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import time

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

vectorize_path = os.path.abspath(os.path.join(project_root, "vectorize"))
if vectorize_path not in sys.path:
    sys.path.insert(0, vectorize_path)


import vectorize.tokenizer_definition

sys.modules["TF_IDF"] = vectorize.tokenizer_definition

from storage.vector_storage import load_tfidf_matrix, load_vectorizer, save_clusters


# -------------------------------------------------------------------------------------
# Elbow Curve
# -------------------------------------------------------------------------------------

def plot_elbow_curve(dataset_name: str, max_k=30):
    """
    يرسم مخطط الكوع لتحديد عدد المجموعات الأمثل (k) باستخدام MiniBatchKMeans.

    Parameters:
    - dataset_name (str): اسم مجموعة البيانات.
    - max_k (int): أقصى عدد من المجموعات لاختباره.
    """
    print(f"Generating Elbow Curve for {dataset_name} using MiniBatchKMeans...")
    start_time_load = time.time()
    loaded_tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    end_time_load = time.time()
    print(f"TF-IDF matrix loaded in {end_time_load - start_time_load:.2f} seconds. Matrix shape: {loaded_tfidf_matrix.shape}")

    inertias = []
    k_values = list(range(2, max_k + 1))

    for k in k_values:
        print(f"Calculating WCSS for k = {k} using MiniBatchKMeans...")
        start_time_kmeans = time.time()
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=256)
        kmeans.fit(loaded_tfidf_matrix)
        inertias.append(kmeans.inertia_)
        end_time_kmeans = time.time()
        print(f"MiniBatchKMeans for k={k} completed in {end_time_kmeans - start_time_kmeans:.2f} seconds.")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster SSE)')
    plt.title(f'Elbow Curve for {dataset_name} (MiniBatchKMeans)')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()
    print("Elbow Curve plot displayed.")

# -------------------------------------------------------------------------------------
# Build Clusters and visualize them
# -------------------------------------------------------------------------------------

def build_and_view_clusters(dataset_name: str, n_clusters: int, top_terms=True):

    print(f"Clustering {dataset_name} into {n_clusters} clusters using MiniBatchKMeans...")
    start_time_total = time.time()

    print("Loading TF-IDF matrix...")
    start_time_load = time.time()
    loaded_tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    end_time_load = time.time()
    print(f"TF-IDF matrix loaded in {end_time_load - start_time_load:.2f} seconds. Matrix shape: {loaded_tfidf_matrix.shape}")

    print(f"Starting MiniBatchKMeans training for {n_clusters} clusters...")
    start_time_kmeans_fit = time.time()
    k_means = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=256)
    clusters = k_means.fit_predict(loaded_tfidf_matrix)
    end_time_kmeans_fit = time.time()
    print(f"MiniBatchKMeans training and prediction completed in {end_time_kmeans_fit - start_time_kmeans_fit:.2f} seconds.")

    print("Calculating Silhouette Score (this might take a very long time for large datasets)...")
    start_time_silhouette = time.time()
    try:
        sample_size = 50000
        if loaded_tfidf_matrix.shape[0] > sample_size:
            print(f"Sampling {sample_size} documents for Silhouette Score calculation...")
            sample_indices = np.random.choice(loaded_tfidf_matrix.shape[0], sample_size, replace=False)
            sampled_matrix = loaded_tfidf_matrix[sample_indices]
            sampled_clusters = clusters[sample_indices]
            silhouette = silhouette_score(sampled_matrix, sampled_clusters)
            print(f"Silhouette Score (sampled) for {dataset_name}: {silhouette:.4f}")
        else:
            silhouette = silhouette_score(loaded_tfidf_matrix, clusters)
            print(f"Silhouette Score for {dataset_name}: {silhouette:.4f}")
    except Exception as e:
        print(f"Could not calculate Silhouette Score: {e}. Consider reducing data size or skipping.")
        silhouette = None
    end_time_silhouette = time.time()
    print(f"Silhouette Score calculation completed in {end_time_silhouette - start_time_silhouette:.2f} seconds.")

    print("Counting documents per cluster...")
    counts = np.bincount(clusters)
    for i, count in enumerate(counts):
        print(f'Cluster {i} has {count} vectors')
    print("Document counts per cluster displayed.")

    
    save_clusters(clusters, f"{dataset_name}_all") 
    print(f"Clusters array saved to disk for {dataset_name}.")

    print("Performing TruncatedSVD for 2D visualization...")
    start_time_svd = time.time()
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced_data = svd.fit_transform(loaded_tfidf_matrix)
    end_time_svd = time.time()
    print(f"TruncatedSVD completed in {end_time_svd - start_time_svd:.2f} seconds.")

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ticks=range(n_clusters))
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title(f'2D Visualization of Clusters for {dataset_name} Dataset (MiniBatchKMeans)')
    plt.show()
    print("2D Cluster visualization plot displayed.")

    if top_terms:
        print("Extracting top terms for each cluster...")
        start_time_top_terms = time.time()
        try:
            vectorizer = load_vectorizer(f"{dataset_name}_all")
            terms = vectorizer.get_feature_names_out()
            for cluster_num in range(n_clusters):
                indices = np.where(clusters == cluster_num)[0]
                if len(indices) == 0:
                    continue
                cluster_vectors = loaded_tfidf_matrix[indices]
                mean_tfidf = cluster_vectors.mean(axis=0)
                top_indices = np.argsort(mean_tfidf)[-10:]
                top_terms_list = [terms[i] for i in top_indices]
                print(f"Cluster {cluster_num} top terms:")
                print(top_terms_list)
                print("-" * 50)
            end_time_top_terms = time.time()
            print(f"Top terms extraction completed in {end_time_top_terms - start_time_top_terms:.2f} seconds.")
        except Exception as e:
            print(f"[!] Could not extract top terms: {str(e)}. Please ensure 'storage.load_vectorizer' is correctly implemented and accessible, and that the vectorizer file exists and all its dependencies (like 'TF_IDF' module) are in Python's path.")
   
    end_time_total = time.time()
    print(f"Total execution time for build_and_view_clusters: {end_time_total - start_time_total:.2f} seconds.")

