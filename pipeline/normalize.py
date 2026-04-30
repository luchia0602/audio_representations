import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def apply_lobanov(df):
    # Taken from the parse_corpus.csv
    vowels = [
        'a', 'ɑ', 'e', 'ɛ', 'i', 'o', 'u', 'y', 'ø', 'œ', 'ə', # French Oral Vowels & Schwa
        'ɑ̃', 'ɛ̃', 'œ̃',                                        # French Nasal Vowels
        'ɨ', 'ɪ', 'ʉ', 'æ'                                     # L2 Interference Vowels
    ]

    df['F1_norm'] = np.nan
    df['F2_norm'] = np.nan
    
    for speaker, group in df.groupby('speaker_id'):
        speaker_vowels = group[group['phoneme_label'].isin(vowels)]
        f1_mean = speaker_vowels['F1'].mean()
        f1_sd = speaker_vowels['F1'].std()
        f2_mean = speaker_vowels['F2'].mean()
        f2_sd = speaker_vowels['F2'].std()
        df.loc[group.index, 'F1_norm'] = (group['F1'] - f1_mean) / f1_sd
        df.loc[group.index, 'F2_norm'] = (group['F2'] - f2_mean) / f2_sd
    return df

def reduce_neural_dims(npz_path, output_path, d=50):
    data = np.load(npz_path)
    reduced_data = {}
    pca = PCA(n_components=d)
    for layer_name in data.files:
        layer_embeddings = data[layer_name]
        reduced = pca.fit_transform(layer_embeddings)
        reduced_data[layer_name] = reduced
    np.savez(output_path, **reduced_data)

def main():
    acoustic_df = pd.read_csv('data/processed/features_acoustic.csv')
    norm_df = apply_lobanov(acoustic_df)
    norm_df.to_csv('data/processed/features_acoustic_norm.csv', index=False)
    print("Acoustic features Lobanov normalised.")
    reduce_neural_dims('data/processed/features_whisper.npz', 'data/processed/features_whisper_pca.npz', d=50)
    print("Whisper embeddings reduced via PCA.")
    reduce_neural_dims('data/processed/features_xlsr.npz', 'data/processed/features_xlsr_pca.npz', d=50)
    print("XLS-R embeddings reduced via PCA.")

if __name__ == "__main__":
    main()