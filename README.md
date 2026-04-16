# Video Captioning Project

MSR-VTT video captioning using CLIP + DINOv2 + VGGish multimodal fusion with LSTM decoder.

## Project Structure
PROJECT/
├── src/
│   ├── data/           # data loading and preprocessing
│   ├── models/         # encoder and decoder models
│   └── training/       # training loops
├── data/               # not tracked by git (too large)
├── outputs/            # not tracked by git
└── venv/               # not tracked by git

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt