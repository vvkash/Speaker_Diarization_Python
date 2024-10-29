import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import json
from sklearn.metrics import confusion_matrix
import os

# Configuration
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token='ADD_AUTH_TOKEN'
)

audio_files = [
    "/Users/aakash/SDP/ami_sample/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav",
    "/Users/aakash/SDP/ami_sample/amicorpus/IS1000a/audio/IS1000a.Mix-Headset.wav",
    "/Users/aakash/SDP/ami_sample/amicorpus/TS3003a/audio/TS3003a.Mix-Headset.wav"
]

ground_truth = {
    "/Users/aakash/SDP/ami_sample/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav": [
        (0, 10, "speaker_A"), (10, 20, "speaker_B"), (20, 30, "speaker_A")
    ],
    "/Users/aakash/SDP/ami_sample/amicorpus/IS1000a/audio/IS1000a.Mix-Headset.wav": [
        (0, 15, "speaker_A"), (15, 25, "speaker_B")
    ],
    "/Users/aakash/SDP/ami_sample/amicorpus/TS3003a/audio/TS3003a.Mix-Headset.wav": [
        (0, 12, "speaker_A"), (12, 22, "speaker_C"), (22, 32, "speaker_B")
    ]
}

def process_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    duration_seconds = len(audio) / 1000.0
    print(f"Audio duration: {duration_seconds:.2f} seconds")
    
    if duration_seconds > 60:
        print("Creating 60-second test file...")
        test_audio = audio[:60000]
        test_path = audio_path.replace(".wav", "_test_60sec.wav")
        test_audio.export(test_path, format="wav")
        audio_path = test_path
    
    print("Starting diarization...")
    diarization = pipeline(audio_path)
    
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append((turn.start, turn.end, speaker))
    
    return results

def calculate_error(predicted, actual):
    total_time = sum(end - start for start, end, _ in actual)
    error_time = 0
    
    for p_start, p_end, p_speaker in predicted:
        for a_start, a_end, a_speaker in actual:
            overlap_start = max(p_start, a_start)
            overlap_end = min(p_end, a_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                if p_speaker != a_speaker:
                    error_time += overlap_duration
    
    return error_time / total_time * 100

# Process each audio file
for audio_file in audio_files:
    print(f"\nProcessing {audio_file}...")
    
    try:
        diarization_results = process_audio(audio_file)
        error_rate = calculate_error(diarization_results, ground_truth[audio_file])
        
        print(f"Diarization Results:")
        for start, end, speaker in diarization_results:
            print(f"Speaker {speaker}: {start:.1f}s - {end:.1f}s")
        print(f"Error Rate: {error_rate:.2f}%")
        
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")