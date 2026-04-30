import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import cosine_similarity

def calculate_variance_ratio(X_2d, labels):
    global_mean = X_2d.mean(axis=0)
    total_variance = ((X_2d - global_mean)**2).sum()
    df_2d = pd.DataFrame(X_2d, columns=['D1', 'D2'])
    df_2d['label'] = labels.values
    between_variance = 0
    for label, group in df_2d.groupby('label'):
        n_k = len(group)
        centroid = group[['D1', 'D2']].mean().values
        between_variance += n_k * np.sum((centroid - global_mean)**2)
    return between_variance / total_variance if total_variance > 0 else 0

def calculate_cosine_metrics(X_high_dim, labels):
    n_samples = min(1000, len(X_high_dim))
    idx = np.random.choice(len(X_high_dim), n_samples, replace=False)
    X_sample = X_high_dim[idx]
    labels_sample = labels.iloc[idx].values
    sim_matrix = cosine_similarity(X_sample)
    label_grid_row, label_grid_col = np.meshgrid(labels_sample, labels_sample)
    within_mask = (label_grid_row == label_grid_col) & ~np.eye(n_samples, dtype=bool)
    between_mask = (label_grid_row != label_grid_col)
    within_sim = sim_matrix[within_mask].mean()
    between_sim = sim_matrix[between_mask].mean()
    return within_sim, between_sim, (within_sim / between_sim)

def plot_projections(X_2d, df, model_name, method, output_dir):
    df_plot = df.copy()
    df_plot['Dim1'] = X_2d[:, 0]
    df_plot['Dim2'] = X_2d[:, 1]
    
    features_to_plot = {
        'phoneme_label': 'tab10',
        'L1_status': 'Set1',
        'gender': 'Set2'
    }
    
    for feature, palette in features_to_plot.items():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_plot, x='Dim1', y='Dim2', 
            hue=feature, palette=palette, s=15, alpha=0.7
        )
        plt.title(f'{model_name} {method} Projection - Colored by {feature}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_{method}_{feature}.png'), dpi=300)
        plt.close()

def process_model(model_name, npz_path, layer_name, df, output_dir, metrics_file):
    data = np.load(npz_path)[layer_name]
    pca_2d = PCA(n_components=2).fit_transform(data)
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(data)
    
    plot_projections(pca_2d, df, model_name, "PCA", output_dir)
    plot_projections(umap_2d, df, model_name, "UMAP", output_dir)

    pca_var_ratio = calculate_variance_ratio(pca_2d, df['phoneme_label'])
    umap_var_ratio = calculate_variance_ratio(umap_2d, df['phoneme_label'])
    w_sim, b_sim, ratio = calculate_cosine_metrics(data, df['phoneme_label'])
    metrics_file.write(f"--- {model_name} ({layer_name}) ---\n")
    metrics_file.write(f"Variance Ratio (PCA 2D): {pca_var_ratio:.4f}\n")
    metrics_file.write(f"Variance Ratio (UMAP 2D): {umap_var_ratio:.4f}\n")
    metrics_file.write(f"Avg Within-Phoneme Cosine Sim: {w_sim:.4f}\n")
    metrics_file.write(f"Avg Between-Phoneme Cosine Sim: {b_sim:.4f}\n")
    metrics_file.write(f"Similarity Ratio (Within/Between): {ratio:.4f}\n\n")

def main():
    csv_path = 'data/processed/corpus_parsed.csv'
    whisper_path = 'data/processed/features_whisper.npz'
    xlsr_path = 'data/processed/features_xlsr.npz'
    output_dir = 'results'
    df = pd.read_csv(csv_path)
    oral_vowels = ['a', 'ɑ', 'e', 'ɛ', 'i', 'o', 'u', 'y', 'ø', 'œ']
    mask = df['phoneme_label'].isin(oral_vowels)
    df_filtered = df[mask].reset_index(drop=True)
    
    def load_and_filter(path):
        data = np.load(path)
        return {k: data[k][mask] for k in data.files}
        
    np.savez('temp_w.npz', **load_and_filter(whisper_path))
    np.savez('temp_x.npz', **load_and_filter(xlsr_path))
    
    with open(os.path.join(output_dir, 'neural_metrics_report.txt'), 'w') as f:
        process_model('Whisper', 'temp_w.npz', 'layer_20', df_filtered, output_dir, f)
        process_model('XLS-R', 'temp_x.npz', 'layer_12', df_filtered, output_dir, f)
    os.remove('temp_w.npz')
    os.remove('temp_x.npz')

if __name__ == "__main__":
    main()