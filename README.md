# Video Captioning Project

MSR-VTT video captioning using CLIP + DINOv2 + VGGish multimodal fusion with LSTM decoder.

## Project Structure
PROJECT/ <br>
├── src/<br>
│   ├── data/           # data loading and preprocessing<br>
│   ├── models/         # encoder and decoder models<br>
│   └── training/       # training loops<br>
├── data/               # not tracked by git <(too large)<br>
├── outputs/            # not tracked by git<br>
└── venv/               # not tracked by git<br>

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt