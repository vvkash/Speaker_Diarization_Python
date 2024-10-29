Speaker Diarization in Python

To Run: download audio files (can be anything)
Make huggingface access token: https://huggingface.co/docs/hub/en/security-tokens
Get access for these libraries in huggingface:
https://huggingface.co/pyannote/speaker-diarization-3.1 (newer)
https://huggingface.co/pyannote/speaker-diarization (older)
https://huggingface.co/pyannote/segmentation-3.0 (for segmentation)
https://huggingface.co/pyannote/segmentation (older segmentation model)

file tree should be like:

project-directory/
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── sample3.wav
├── annotations/
│   ├── sample1.rttm
│   ├── sample2.rttm
│   └── sample3.rttm
│── Source Code/
│   ├── main.py
│────────
