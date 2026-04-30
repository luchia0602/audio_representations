import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def compute_summary_stats(df, output_dir):
    oral_vowels = ['a', 'ɑ', 'e', 'ɛ', 'i', 'o', 'u', 'y', 'ø', 'œ']
    vowel_df = df[df['phoneme_label'].isin(oral_vowels)].copy()
    grouped = vowel_df.groupby(['phoneme_label', 'L1_status', 'gender'])
    
    stats = grouped.agg(
        F1_mean=('F1_norm', 'mean'),
        F1_median=('F1_norm', 'median'),
        F1_sd=('F1_norm', 'std'),
        F2_mean=('F2_norm', 'mean'),
        F2_median=('F2_norm', 'median'),
        F2_sd=('F2_norm', 'std')
    ).reset_index()
    
    for f in ['F1', 'F2']:
        stats[f'{f}_IQR'] = grouped[f'{f}_norm'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).values
        stats[f'{f}_CV'] = stats[f'{f}_sd'] / stats[f'{f}_mean'].abs() 
    stats.to_csv(os.path.join(output_dir, 'acoustic_summary_stats.csv'), index=False)    
    return vowel_df

def plot_vowel_chart(df, output_dir):
    plt.figure(figsize=(10, 8))
    df['Speaker_Group'] = df['L1_status'] + "_" + df['gender']
    centroids = df.groupby(['phoneme_label', 'Speaker_Group'])[['F1_norm', 'F2_norm']].mean().reset_index()
    
    sns.scatterplot(
        data=centroids, 
        x='F2_norm', 
        y='F1_norm', 
        hue='Speaker_Group', 
        style='phoneme_label',
        s=150, 
        palette='tab10'
    )
    
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.title('Lobanov-Normalised Vowel Space (Centroids)')
    plt.xlabel('F2 (Normalised)')
    plt.ylabel('F1 (Normalised)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vowel_chart_centroids.png'), dpi=300)
    plt.close()

def plot_boxplots(df, output_dir):
    for formant in ['F1_norm', 'F2_norm']:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df, 
            x='phoneme_label', 
            y=formant, 
            hue=df[['L1_status', 'gender']].apply(tuple, axis=1),
            palette='Set2'
        )
        plt.title(f'{formant} Distribution per Phoneme by Group')
        plt.legend(title='(L1, Gender)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{formant}.png'), dpi=300)
        plt.close()

def main():
    input_file = 'data/processed/features_acoustic_norm.csv'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    oral_vowel_df = compute_summary_stats(df, output_dir)
    plot_vowel_chart(oral_vowel_df, output_dir)
    plot_boxplots(oral_vowel_df, output_dir)

if __name__ == "__main__":
    main()