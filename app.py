import os
import streamlit as st
import google.generativeai as genai
import shutil

# Assuming these are imported from your existing project
from main import process_audio

def setup_gemini_model():
    """Configure Gemini model for summarization"""
    genai.configure(api_key="AIzaSyBAbaAg4QxMuYyKi8OrecFdyeVa7n-EMHs")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are an expert summarizer of minutes of meeting transcripts, you not only give the summary, but you also provide the context of what speakers are talking about and give some background to that. The final output should have the normal summarization first and then the additional background context\n",
    )
    
    return model

def generate_summary(model, transcripts):
    """Generate summary using Gemini model"""
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(transcripts)
    return response.text

def main():
    st.title("Audio Transcription and Summarization App")

    # Sidebar for speaker audio uploads
    st.sidebar.header("Speaker Audio Files")
    
    # Create speaker directory if it doesn't exist
    speaker_dir = "speaker_audio"
    os.makedirs(speaker_dir, exist_ok=True)

    # Upload individual speaker audio files
    uploaded_speaker_files = st.sidebar.file_uploader(
        "Upload Individual Speaker Audio Files", 
        type=['wav'], 
        accept_multiple_files=True
    )

    # Store uploaded speaker files
    if uploaded_speaker_files:
        for uploaded_file in uploaded_speaker_files:
            file_path = os.path.join(speaker_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        st.sidebar.success(f"Saved {len(uploaded_speaker_files)} speaker audio files")

    # Main area for conversation audio upload
    st.header("Conversation Audio")
    conversation_audio = st.file_uploader(
        "Upload Conversation Audio (WAV only)", 
        type=['wav']
    )

    # Process and display transcripts
    if conversation_audio:
        # Save conversation audio
        conversation_audio_path = "conversation_audio.wav"
        with open(conversation_audio_path, "wb") as f:
            f.write(conversation_audio.getvalue())
        
        st.success("Conversation audio uploaded successfully!")

        # Process audio
        try:
            with st.spinner("Diariazation..."):
                all_transcripts = process_audio(
                    conversation_audio=conversation_audio_path,
                    speaker_dir=speaker_dir
                )

                model = setup_gemini_model()
            
                summary = generate_summary(model, all_transcripts)
                st.subheader("Transcripts")
                st.json(all_transcripts)
                st.subheader("Transcript Summary")
                st.write(summary)

        except Exception as e:
            st.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()