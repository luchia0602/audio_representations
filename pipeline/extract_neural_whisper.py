import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor
from tqdm import tqdm

def main():
    drive_base_path = "/content/drive/MyDrive/speech_project" # processed this on Google Colab with mounting Google Drive
    input_csv = f"{drive_base_path}/data/processed/corpus_parsed.csv"
    output_file = f"{drive_base_path}/data/processed/features_whisper.npz"
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model_name = "openai/whisper-medium"
    target_layers = [8, 20] 
    model = WhisperModel.from_pretrained(model_name).to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    df = pd.read_csv(input_csv)
    embeddings = {f"layer_{l}": [] for l in target_layers}

    for wav_path, group in tqdm(df.groupby('wav_path')):
        clean_wav_path = wav_path.replace("\\", "/")
        actual_path = os.path.join(drive_base_path, clean_wav_path)
        
        if not os.path.exists(actual_path):
            continue
            
        try:
            waveform, sample_rate = torchaudio.load(actual_path)
            if sample_rate != 16000: # Whisper requirement
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
        except Exception:
            continue
        
        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)        
        with torch.no_grad():
            outputs = model.encoder(input_features, output_hidden_states=True)
            

        for idx, row in group.iterrows():
            start_frame = int(row['onset'] * 50)
            end_frame = int(row['offset'] * 50)
            
            for layer_idx in target_layers:
                hidden_states = outputs.hidden_states[layer_idx] 
                phoneme_slice = hidden_states[:, start_frame:end_frame, :]
                if phoneme_slice.size(1) > 0:
                    pooled = phoneme_slice.mean(dim=1).squeeze().cpu().numpy()
                else:
                    pooled = np.zeros(hidden_states.size(-1))
                embeddings[f"layer_{layer_idx}"].append(pooled)

    np.savez(output_file, **{k: np.array(v) for k, v in embeddings.items()})

if __name__ == "__main__":
    main()