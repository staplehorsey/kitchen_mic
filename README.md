# Kitchen Mic 🎙️

An always-on audio recording system that captures, transcribes, and organizes conversations with high-fidelity audio preservation.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `src/`: Source code modules
  - `audio/`: Audio processing modules
  - `transcription/`: Whisper integration
  - `vad/`: Voice activity detection
  - `llm/`: LLM integration
  - `utils/`: Shared utilities
- `debug/`: Debug tools and audio monitoring
- `storage/`: Data storage
  - `raw/`: High fidelity audio storage
  - `processed/`: Transcriptions and summaries
- `config/`: Configuration files

## Audio Source
- Stream URL: staple.local:12345
- Sample Rate: 44kHz raw audio
