import sounddevice as sd
import numpy as np
import queue
import threading
import time
import wave
from scipy import signal
from scipy.io import wavfile

class AudioCleaner:
    """Audio processing utilities for cleaning speech audio"""
    
    @staticmethod
    def remove_silence(audio_data, sample_rate, silence_threshold=0.01, min_silence_duration=0.5):
        """Remove silence segments from audio"""
        # Calculate frame size for the minimum silence duration
        frame_size = int(min_silence_duration * sample_rate)
        
        # Calculate RMS energy for each frame
        frames = []
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame**2))
                frames.append((i, i + len(frame), rms > silence_threshold))
        
        # Keep only non-silent frames
        cleaned_audio = []
        for start, end, is_voice in frames:
            if is_voice:
                cleaned_audio.append(audio_data[start:end])
        
        return np.concatenate(cleaned_audio) if cleaned_audio else np.array([])
    
    @staticmethod
    def apply_noise_reduction(audio_data, sample_rate):
        """Apply basic noise reduction using spectral subtraction"""
        # Apply a high-pass filter to remove low-frequency noise
        nyquist = sample_rate / 2
        high_cutoff = 80  # Remove frequencies below 80 Hz
        
        if high_cutoff < nyquist:
            sos = signal.butter(4, high_cutoff / nyquist, btype='high', output='sos')
            filtered_audio = signal.sosfilt(sos, audio_data)
        else:
            filtered_audio = audio_data
        
        return filtered_audio
    
    @staticmethod
    def normalize_audio(audio_data, target_level=0.7):
        """Normalize audio to target level"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized = audio_data * (target_level / max_val)
            return normalized
        return audio_data
    
    @staticmethod
    def apply_voice_enhancement(audio_data, sample_rate):
        """Enhance voice frequencies (300-3400 Hz)"""
        nyquist = sample_rate / 2
        low_cutoff = 300 / nyquist
        high_cutoff = 3400 / nyquist
        
        # Band-pass filter for voice frequencies
        sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
        enhanced_audio = signal.sosfilt(sos, audio_data)
        
        return enhanced_audio

class CleanAudioCapture:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.int16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.cleaner = AudioCleaner()
        
        # Voice activity detection parameters
        self.voice_threshold = 0.02
        self.silence_duration = 2.0  # seconds
        self.last_voice_time = time.time()
        self.voice_detected = False
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input with voice activity detection"""
        if status:
            print(f"Audio input error: {status}")
        
        # Calculate volume (RMS) for voice activity detection
        volume = np.sqrt(np.mean(indata**2))
        
        # Voice activity detection
        if volume > self.voice_threshold:
            self.voice_detected = True
            self.last_voice_time = time.inputBufferAdcTime
            
        # Add audio to queue if voice is detected
        if self.voice_detected:
            audio_data = (indata * 32767).astype(self.dtype)
            self.audio_queue.put(audio_data.copy())
            
        # Check for silence timeout
        current_time = time.inputBufferAdcTime
        if (current_time - self.last_voice_time) > self.silence_duration:
            if self.voice_detected:
                print("Silence detected, processing audio...")
                self.stop_recording()
    
    def start_recording(self):
        """Start real-time audio capture with voice activity detection"""
        self.is_recording = True
        self.voice_detected = False
        self.last_voice_time = time.time()
        
        print("🎤 Listening for voice... Speak now!")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
        
    def stop_recording(self):
        """Stop audio capture"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        print("🛑 Recording stopped.")
        
    def get_raw_audio_buffer(self):
        """Get raw audio data from queue"""
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if audio_chunks:
            return np.concatenate(audio_chunks, axis=0)
        return np.array([])
    
    def get_cleaned_audio(self, apply_enhancement=True):
        """Get cleaned and processed audio ready for STT"""
        raw_audio = self.get_raw_audio_buffer()
        
        if len(raw_audio) == 0:
            return np.array([])
            
        print("🧹 Cleaning audio...")
        
        # Convert to float for processing
        audio_float = raw_audio.astype(np.float32) / 32767.0
        
        # Step 1: Remove silence segments
        cleaned_audio = self.cleaner.remove_silence(
            audio_float, 
            self.sample_rate,
            silence_threshold=0.01,
            min_silence_duration=0.3
        )
        
        if len(cleaned_audio) == 0:
            print("⚠️ No voice detected in audio")
            return np.array([])
        
        # Step 2: Apply noise reduction
        cleaned_audio = self.cleaner.apply_noise_reduction(cleaned_audio, self.sample_rate)
        
        # Step 3: Enhance voice frequencies (optional)
        if apply_enhancement:
            cleaned_audio = self.cleaner.apply_voice_enhancement(cleaned_audio, self.sample_rate)
        
        # Step 4: Normalize audio
        cleaned_audio = self.cleaner.normalize_audio(cleaned_audio, target_level=0.8)
        
        # Convert back to int16
        cleaned_audio_int16 = (cleaned_audio * 32767).astype(np.int16)
        
        print(f"✅ Audio cleaned: {len(raw_audio)} → {len(cleaned_audio_int16)} samples")
        return cleaned_audio_int16
    
    def save_audio(self, audio_data, filename):
        """Save audio data to WAV file"""
        if len(audio_data) > 0:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            print(f"💾 Audio saved to {filename}")
        else:
            print("⚠️ No audio data to save")

class SarvamSTTIntegration:
    """Integration class for Sarvam's Saarika v2 STT model"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        # Note: Replace with actual Sarvam API endpoint and implementation
        
    def transcribe_audio(self, audio_data, sample_rate=16000, source_language="hi-IN"):
        """
        Send cleaned audio to Sarvam's Saarika v2 for transcription
        
        Args:
            audio_data: Cleaned audio data (int16)
            sample_rate: Audio sample rate
            source_language: Source language code (e.g., "hi-IN", "ta-IN", etc.)
        """
        # Convert audio to bytes for API call
        audio_bytes = audio_data.tobytes()
        
        # Placeholder for Sarvam API call
        # Replace this with actual Sarvam API implementation
        print(f"🔄 Sending {len(audio_bytes)} bytes to Sarvam Saarika v2...")
        print(f"📝 Language: {source_language}")
        
        # Mock response - replace with actual API call
        transcribed_text = "Mock transcription result"
        confidence_score = 0.95
        
        return {
            "transcription": transcribed_text,
            "confidence": confidence_score,
            "language_detected": source_language
        }

# Main usage example
def main():
    """Example usage of the clean audio capture system"""
    
    # Initialize components
    audio_capture = CleanAudioCapture(sample_rate=16000)
    stt_service = SarvamSTTIntegration()
    
    try:
        # Start recording
        audio_capture.start_recording()
        
        # Wait for recording to complete (voice activity detection will stop it)
        while audio_capture.is_recording:
            time.sleep(0.1)
        
        # Get cleaned audio
        cleaned_audio = audio_capture.get_cleaned_audio(apply_enhancement=True)
        
        if len(cleaned_audio) > 0:
            # Save for debugging
            audio_capture.save_audio(cleaned_audio, "cleaned_audio.wav")
            
            # Send to Sarvam STT
            result = stt_service.transcribe_audio(
                cleaned_audio, 
                sample_rate=audio_capture.sample_rate,
                source_language="hi-IN"  # Change as needed
            )
            
            print(f"🎯 Transcription: {result['transcription']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            
        else:
            print("❌ No valid audio captured")
            
    except KeyboardInterrupt:
        audio_capture.stop_recording()
        print("\n👋 Recording interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()