import os
import time
from dotenv import load_dotenv

# Assuming audio_capture.py contains CleanAudioCapture and a mock/real STT class
# For this example, we'll mock STT if SarvamSTTIntegration isn't fully defined.
try:
    # from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration # Assuming STT class
    raise ImportError("Mocking CleanAudioCapture, SarvamSTTIntegration for this example")  # Mock import for demonstration
except ImportError:
    print("Warning: src.audio_capture not fully available. Mocking audio capture components.")
    # Mock CleanAudioCapture if not found, for basic testing
    class CleanAudioCapture:
        def __init__(self, sample_rate=16000): self.sample_rate = sample_rate; self.is_recording = False
        def start_recording(self): print("Mock: Start recording..."); self.is_recording = True; time.sleep(1) # Simulate recording
        def stop_recording(self): print("Mock: Stop recording..."); self.is_recording = False
        def get_cleaned_audio(self): print("Mock: Get cleaned audio"); return b"mock_audio_data" # Return mock bytes
        def save_audio(self, data, filename): print(f"Mock: Save audio to {filename}")

    class SarvamSTTIntegration: # Mock STT
        def transcribe_audio(self, audio_data, sample_rate, source_language="hi-IN"):
            print(f"Mock STT: Transcribing with lang {source_language}")
            if source_language == "hi-IN":
                return {"transcription": "मुझे दो दिन से बुखार और खांसी है", "confidence": 0.9}
            return {"transcription": "I have fever and cough for two days", "confidence": 0.9}


from src.nlu_processor import SarvamMNLUProcessor, NLUResult, HealthIntent
from src.response_generator import HealHubResponseGenerator

def run_healhub_voice_app():
    """
    Main function to run the HealHub Voice Application.
    """
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        print("❌ SARVAM_API_KEY not found in environment variables. Please set it in your .env file.")
        print("   You can get your API key from: https://dashboard.sarvam.ai")
        return

    print("🚀 Initializing HealHub Voice Application...")
    
    # Initialize components
    # For actual use, ensure CleanAudioCapture and SarvamSTTIntegration are correctly implemented
    audio_capturer = CleanAudioCapture(sample_rate=16000)
    stt_service = SarvamSTTIntegration() # Replace with actual STT client if available
    nlu_processor = SarvamMNLUProcessor(api_key=api_key)
    response_gen = HealHubResponseGenerator(api_key=api_key)

    try:
        print("\n🎤 Speak your health query now (simulated audio capture)...")
        audio_capturer.start_recording()
        # In a real app, VAD would stop recording or user interaction
        time.sleep(2) # Simulate speaking time
        audio_capturer.stop_recording()

        cleaned_audio_data = audio_capturer.get_cleaned_audio()

        if not cleaned_audio_data:
            print("⚠️ No audio data captured.")
            return

        # --- STT Step ---
        # Determine language for STT - this might come from user preference or auto-detection
        # For now, let's allow choosing or default.
        stt_language = "hi-IN" # or "en-IN", etc.
        print(f"\n🔊 Transcribing audio (Language: {stt_language})...")
        stt_result = stt_service.transcribe_audio(
            cleaned_audio_data,
            sample_rate=audio_capturer.sample_rate,
            source_language=stt_language
        )
        transcribed_text = stt_result.get("transcription")
        if not transcribed_text:
            print("❌ STT failed to transcribe audio.")
            return
        print(f"📝 User Query (Transcribed): \"{transcribed_text}\" (Confidence: {stt_result.get('confidence', 'N/A')})")

        # --- NLU Step ---
        print("\n🧠 Processing NLU...")
        nlu_output: NLUResult = nlu_processor.process_transcription(
            transcribed_text,
            source_language=stt_language # Pass STT language to NLU
        )
        print(f"🔍 NLU Intent: {nlu_output.intent.value}")
        if nlu_output.entities:
            print(f"   Entities: {', '.join([e.text + ' (' + e.entity_type + ')' for e in nlu_output.entities])}")
        else:
            print("   Entities: None found")
        print(f"   Is Emergency (NLU): {nlu_output.is_emergency}")
        print(f"   Detected Language (NLU): {nlu_output.language_detected}")


        # --- Response Generation Step ---
        print("\n💬 Generating response...")
        final_response = response_gen.generate_response(transcribed_text, nlu_output)
        
        print("\n\n💡 HealHub Assistant Says:")
        print("--------------------------------------------------")
        print(final_response)
        print("--------------------------------------------------")

        # --- TTS Step (Placeholder) ---
        print("\n🗣️ (TTS Placeholder: Speaking response...)")
        # tts_service.speak(final_response, language=nlu_output.language_detected)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the main application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Create a .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("SARVAM_API_KEY=your_sarvam_api_key_here\n")
        print("📝 Created .env file. Please add your SARVAM_API_KEY.")
    
    run_healhub_voice_app()