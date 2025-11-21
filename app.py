# app_memory_full.py
import os
import io
import requests
import streamlit as st
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import wave

# Load environment variables
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
AZURE_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")

if not (OPENROUTER_KEY and AZURE_KEY and AZURE_REGION):
    st.warning("Set OPENROUTER_API_KEY, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION in env or .env.")
    st.stop()

st.title("🗣️ Voice → LLM → 🗣️ Voice Assistant")

st.markdown("Record a question, it will be transcribed by Azure, answered by OpenRouter, and spoken back.")

# 1️⃣ Record audio
audio_file = st.audio_input("Press and speak (short utterances work best).")

if audio_file:
    st.audio(audio_file)  # playback
    if st.button("Send to LLM"):
        st.info("Processing...")

        # ----------------------------
        # 2️⃣ Azure STT (in-memory)
        # ----------------------------
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)

        class BytesStreamCallback(speechsdk.audio.PullAudioInputStreamCallback):
            """Feed WAV bytes into Azure STT"""
            def __init__(self, wav_bytes):
                super().__init__()
                self.wav_file = wave.open(io.BytesIO(wav_bytes), 'rb')
                self.sampwidth = self.wav_file.getsampwidth()
                self.nchannels = self.wav_file.getnchannels()

            def read(self, buffer: memoryview) -> int:
                frames_to_read = len(buffer) // (self.sampwidth * self.nchannels)
                data = self.wav_file.readframes(frames_to_read)
                buffer[:len(data)] = data
                return len(data)

            def close(self):
                self.wav_file.close()

        callback = BytesStreamCallback(audio_file.getvalue())
        stream = speechsdk.audio.PullAudioInputStream(callback)
        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        try:
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                user_text = result.text
                st.success("Transcription: " + user_text)
            else:
                st.error(f"Transcription failed. Reason: {result.reason}")
                st.stop()
        except Exception as e:
            st.error("Azure STT error: " + str(e))
            st.stop()

        # ----------------------------
        # 3️⃣ OpenRouter LLM call
        # ----------------------------
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": 512,
        }

        try:
            r = requests.post(openrouter_url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            llm_text = r.json()["choices"][0]["message"]["content"].strip()
            st.write("LLM answer:")
            st.info(llm_text)
        except Exception as e:
            st.error(f"OpenRouter call failed: {e}")
            st.stop()

        # ----------------------------
        # 4️⃣ Azure TTS (in-memory)
        # ----------------------------
        try:
            synthesizer_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
            synthesizer_config.speech_synthesis_voice_name = "en-US-AvaNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=synthesizer_config, audio_config=None)
            result = synthesizer.speak_text_async(llm_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get audio bytes and wrap in BytesIO
                audio_bytes = result.audio_data
                audio_bytes_io = io.BytesIO(audio_bytes)
                st.audio(audio_bytes_io)
            else:
                st.error(f"TTS failed. Reason: {result.reason}")
        except Exception as e:
            st.error("Azure TTS error: " + str(e))
