import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
import numpy as np
import re


class HealHubUtilities:
    """Core utilities for HealHub healthcare application"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_api_url = "https://api.sarvam.ai"
        self._initialize_language_support()

    def _initialize_language_support(self):
        """Initialize language configurations"""
        self.LANGUAGE_MAP = {
            "en-IN": {
                "display": "English",
                "disclaimer": "This information is for general knowledge and informational purposes only, and does not constitute professional advice."
            },
            "hi-IN": {
                "display": "हिन्दी",
                "disclaimer": "यह जानकारी केवल सामान्य ज्ञान और सूचनात्मक उद्देश्यों के लिए है, और इसे पेशेवर सलाह नहीं माना जाना चाहिए।"
            },
            "bn-IN": {
                "display": "বাংলা",
                "disclaimer": "এই তথ্য শুধুমাত্র সাধারণ জ্ঞান এবং তথ্যের উদ্দেশ্যে, এবং পেশাদারী পরামর্শ হিসাবে গণ্য করা উচিত নয়।"
            },
            "mr-IN": {
                "display": "मराठी",
                "disclaimer": "ही माहिती केवळ सामान्य ज्ञान आणि माहितीच्या उद्देशाने आहे आणि याला व्यावसायिक सल्ला मानले जाऊ नये."
            },
            "kn-IN": {
                "display": "ಕನ್ನಡ",
                "disclaimer": "ಈ ಮಾಹಿತಿಯು ಸಾಮಾನ್ಯ ಜ್ಞಾನ ಮತ್ತು ಮಾಹಿತಿ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ, ಮತ್ತು ಇದನ್ನು ವೃತ್ತಿಪರ ಸಲಹೆಯೆಂದು ಪರಿಗಣಿಸಬಾರದು."
            },
            "ta-IN": {
                "display": "தமிழ்",
                "disclaimer": "இந்தத் தகவல் பொது அறிவு மற்றும் தகவல் நோக்கங்களுக்காக மட்டுமே, மேலும் இது தொழில்முறை ஆலோசனை எனக் கருதப்படக்கூடாது."
            },
            "te-IN": {
                "display": "తెలుగు",
                "disclaimer": "ఈ సమాచారం సాధారణ జ్ఞానం మరియు సమాచార ప్రయోజనాల కోసం మాత్రమే, మరియు దీనిని వృత్తిపరమైన సలహాగా పరిగణించరాదు."
            },
            "ml-IN": {
                "display": "മലയാളം",
                "disclaimer": "ഈ വിവരങ്ങൾ പൊതുവായ അറിവിനും വിവരങ്ങൾക്കും മാത്രമുള്ളതാണ്, ഇത് ഒരു പ്രൊഫഷണൽ ഉപദേശമായി കണക്കാക്കരുത്."
            }
            # Add other supported Indian languages
        }

    def clean_whitespace(self, text:str):
        # Replace multiple whitespace characters (spaces, tabs, etc.) with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()  # Remove leading/trailing spaces

    def translate_text(self, text: str, target_lang: str) -> str:
        """
        Translate text to target language using Sarvam-M
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'hi-IN')
            context: Translation context (healthcare, general, etc.)
        """
        if target_lang.startswith("en"):
            return text  # No translation needed for English

        headers = {"api-subscription-key": self.api_key}
        payload = {
            "input": text,
            "target_language_code": target_lang,
            "source_language_code": 'auto',
            "mode": "formal",
            "model": "mayura:v1",
        }

        try:
            response = requests.post(
                f"{self.base_api_url}/translate",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            # print(response.json())
            return self.clean_whitespace(response.json()["translated_text"])
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Fallback to original

    def translate_text_to_english(self, text: str) -> str:
        """
        Translate text to target language using Sarvam-M
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'hi-IN')
            context: Translation context (healthcare, general, etc.)
        """
        
        headers = {"api-subscription-key": self.api_key}
        payload = {
            "input": text,
            "target_language_code": 'en-IN',
            "source_language_code": 'auto',
            "mode": "formal",
            "model": "mayura:v1",
        }

        try:
            response = requests.post(
                f"{self.base_api_url}/translate",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return self.clean_whitespace(response.json()["translated_text"])
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Fallback to original


    def batch_translate(self, texts: List[str], target_lang: str) -> List[str]:
        """Optimized batch translation for multiple texts"""
        if target_lang.startswith("en"):
            return texts

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "texts": texts,
            "target_lang": target_lang,
            "context": "healthcare"
        }

        try:
            response = requests.post(
                f"{self.base_api_url}/translate/batch",
                headers=headers,
                json=payload,
                timeout=45
            )
            response.raise_for_status()
            return response.json()["translations"]
        except Exception as e:
            print(f"Batch translation error: {e}")
            return texts

    def detect_language(self, text: str) -> str:
        """Robust language detection with code-mixing support"""
        # First check for strong indicators
        if re.search(r"[\u0900-\u097F]", text):  # Hindi/Sanskrit Unicode range
            return "hi-IN"
        if re.search(r"[\u0B80-\u0BFF]", text):  # Tamil
            return "ta-IN"
        
        # Fallback to API detection
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.base_api_url}/detect-language",
                headers=headers,
                json={"text": text},
                timeout=15
            )
            response.raise_for_status()
            return response.json().get("language", "en-IN")
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en-IN"

    def get_display_language(self, lang_code: str) -> str:
        """Get user-friendly language name"""
        return self.LANGUAGE_MAP.get(lang_code, {}).get("display", "English")

    def get_disclaimer(self, lang_code: str) -> str:
        """Get localized disclaimer text"""
        return self.LANGUAGE_MAP.get(lang_code, {}).get("disclaimer", "")

    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target decibel level"""
        if audio_data.size == 0:
            return audio_data
            
        rms = np.sqrt(np.mean(audio_data**2))
        if rms == 0:
            return audio_data
            
        target_linear = 10 ** (target_db / 20)
        return audio_data * (target_linear / rms)

@dataclass
class HealthSafetyResult:
    is_emergency: bool
    requires_redirect: bool
    message: str
    detected_issues: List[str]

def create_safety_layer() -> HealthSafetyResult:
    """Factory for health safety result objects"""
    return HealthSafetyResult(
        is_emergency=False,
        requires_redirect=False,
        message="",
        detected_issues=[]
    )
