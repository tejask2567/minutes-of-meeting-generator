# Audio Transcription and Summarization App

## Project Overview

This Streamlit application provides an end-to-end solution for audio transcription and summarization. It allows users to:
- Upload individual speaker audio files
- Upload a conversation audio file
- Generate transcripts from the conversation
- Create a summary using Google's Gemini AI

## Features

- Multi-speaker audio file upload
- Conversation audio transcription
- AI-powered summarization
- User-friendly Streamlit interface
- Loading indicators for processing steps

## Prerequisites

- Python 3.8+
- Pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-transcript-summarizer.git
cd audio-transcript-summarizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Gemini API Key
Replace the placeholder API key in `app.py` with your actual Google Gemini API key:
```python
genai.configure(api_key="YOUR_ACTUAL_API_KEY")
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### Workflow
1. Upload individual speaker audio files in the sidebar
2. Upload the main conversation audio file
3. Click "Generate Summary" to get AI-powered insights

## Project Structure
- `app.py`: Main Streamlit application
- `main.py`: Core audio processing logic
- `requirements.txt`: Project dependencies
- `speaker_audio/`: Directory for storing individual speaker audio files

## Dependencies
- Streamlit
- Google Generative AI
- Transformers
- PyTorch

## Troubleshooting
- Ensure all audio files are in WAV format
- Check that speaker audio files are correctly named
- Verify Gemini API connectivity

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/audio-transcript-summarizer](https://github.com/yourusername/audio-transcript-summarizer)
