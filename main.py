from pyannote.audio import Model, Pipeline, Inference
import torch
import whisper
import numpy as np
from scipy.spatial.distance import cdist
import os
from subprocess import run
import logging
import time 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_audio(input_file, output_file, start, end):
    """Split audio file into segments."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    length = round(float(end - start), 6)
    start = round(float(start), 6)

    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", input_file, "-t", str(length),
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", "-hide_banner", output_file
    ]
    result = run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

def ensure_2d(embedding):
    """Ensure embedding is 2D."""
    return embedding.reshape(1, -1) if embedding.ndim == 1 else embedding

def load_speaker_embeddings(speaker_dir, inference):
    """Load speaker embeddings from all audio files in a directory."""
    speaker_embeddings = {}
    for file in os.listdir(speaker_dir):
        if file.endswith(".wav"):
            speaker_name = os.path.splitext(file)[0]  # Extract name from file
            speaker_path = os.path.join(speaker_dir, file)
            try:
                embedding = ensure_2d(inference(speaker_path))
                speaker_embeddings[speaker_name] = embedding
                logger.info(f"Loaded embedding for speaker: {speaker_name}")
            except Exception as e:
                logger.error(f"Error loading embedding for {file}: {str(e)}")
    return speaker_embeddings

def process_audio(conversation_audio, speaker_dir):
    """Process conversation audio and assign segments to multiple speakers."""
    if not os.path.exists(conversation_audio):
        raise FileNotFoundError(f"Conversation audio not found: {conversation_audio}")

    logger.info("Initializing models...")
    embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
    diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: ",device)
    embedding_model.to(device)
    diarization.to(device)

    asr_model = whisper.load_model("medium")
    inference = Inference(embedding_model, window="whole").to(device)

    # Load all speaker embeddings
    speaker_embeddings = load_speaker_embeddings(speaker_dir, inference)
    if not speaker_embeddings:
        raise RuntimeError("No valid speaker embeddings found.")

    # Diarize audio
    logger.info("Running diarization...")
    diarization_result = diarization(conversation_audio)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    all_transcripts = []
    segment_count = 1
    start_time = time.time()
    for turn, _, _ in diarization_result.itertracks(yield_label=True):
        logger.info(f"Processing segment {segment_count}...")
        segment_file = os.path.join(output_dir, f"segment_{segment_count}.wav")
        try:
            # Split audio
            split_audio(conversation_audio, segment_file, turn.start, turn.end)
            segment_embedding = ensure_2d(inference(segment_file))

            # Find the closest speaker
            distances = {
                speaker: cdist(segment_embedding, embedding, metric="cosine")[0, 0]
                for speaker, embedding in speaker_embeddings.items()
            }
            speaker, min_distance = min(distances.items(), key=lambda x: x[1])

            # Transcribe segment
            result = asr_model.transcribe(segment_file)
            transcript = result["text"].strip()
            
            # Append to transcript
            transcript_entry = f"{speaker}: {transcript}"
            all_transcripts.append(transcript_entry)
            logger.info(f"Segment {segment_count}: {transcript_entry}")
        except Exception as e:
            logger.error(f"Error processing segment {segment_count}: {str(e)}")
        finally:
            if os.path.exists(segment_file):
                os.remove(segment_file)

        segment_count += 1

    # Save final transcript
    transcript_file = os.path.join(output_dir, "full_transcript.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_transcripts))
    elapsed_time = time.time() - start_time 
    logger.info(f"Elapsed Time{elapsed_time}")
    logger.info(f"Processing complete. Transcript saved to {transcript_file}")
    return all_transcripts