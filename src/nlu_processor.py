import json
import time
import re
import os
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class HealthIntent(Enum):
    """Healthcare-specific intents"""
    SYMPTOM_QUERY = "symptom_query"
    DISEASE_INFO = "disease_info"
    MEDICATION_INFO = "medication_info"
    WELLNESS_TIP = "wellness_tip"
    EMERGENCY = "emergency"
    DIAGNOSIS_REQUEST = "diagnosis_request"
    PREVENTION_INFO = "prevention_info"
    GENERAL_HEALTH = "general_health"
    UNKNOWN = "unknown"

@dataclass
class MedicalEntity:
    """Represents extracted medical entities"""
    text: str
    entity_type: str  # symptom, disease, medication, body_part, etc.
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class NLUResult:
    """NLU processing result"""
    original_text: str
    intent: HealthIntent
    confidence: float
    entities: List[MedicalEntity]
    is_emergency: bool
    requires_disclaimer: bool
    language_detected: str

class SarvamAPIClient:
    """Client for Sarvam AI API services"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable or api_key parameter is required")
        
        self.base_url = "https://api.sarvam.ai"
        
    def chat_completion(self, messages: List[Dict], model: str = "sarvam-m", **kwargs) -> Dict:
        """
        Generate chat completion using Sarvam-M model
        
        Args:
            messages: List of message objects with role and content
            model: Model name (default: sarvam-m)
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "n": kwargs.get("n", 1)
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Sarvam API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return {}

class SarvamMNLUProcessor:
    """NLU processor using Sarvam-M for healthcare queries"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.sarvam_client = SarvamAPIClient(api_key)
        
        # Emergency keywords in multiple Indian languages
        self.emergency_keywords = {
            'en': ['emergency', 'urgent', 'chest pain', 'heart attack', 'stroke', 'bleeding', 'unconscious'],
            'hi': ['आपातकाल', 'तुरंत', 'सीने में दर्द', 'दिल का दौरा', 'बेहोश'],
            'ta': ['அவசரம்', 'உடனடி', 'மார்பு வலி'],
            'te': ['అత్యవసరం', 'తక్షణం', 'ఛాతీ నొప్పి'],
            'bn': ['জরুরি', 'তাৎক্ষণিক', 'বুকে ব্যথা'],
        }
        
        # Diagnosis request patterns
        self.diagnosis_patterns = [
            r'\b(what.*wrong|diagnose|what.*disease|what.*illness)\b',
            r'\b(do i have|am i suffering)\b',
            r'\b(क्या.*बीमारी|निदान)\b',  # Hindi
            r'\b(என்ன.*நோய்|கண்டறிதல்)\b',  # Tamil
        ]
    
    def process_transcription(self, transcribed_text: str, source_language: str = "hi-IN") -> NLUResult:
        """
        Process transcribed text through Sarvam-M for NLU
        
        Args:
            transcribed_text: Text from Saarika v2 STT
            source_language: Source language code
            
        Returns:
            NLUResult with intent, entities, and safety flags
        """
        print(f"🧠 Processing NLU for: '{transcribed_text}'")
        
        # Step 1: Safety checks first
        is_emergency = self._detect_emergency(transcribed_text, source_language)
        requires_disclaimer = self._requires_medical_disclaimer(transcribed_text)
        
        # Step 2: Intent classification using Sarvam-M
        intent, intent_confidence = self._classify_intent(transcribed_text, source_language)
        
        # Step 3: Entity extraction
        entities = self._extract_medical_entities(transcribed_text, source_language)
        
        # Step 4: Language detection refinement
        detected_language = self._detect_language(transcribed_text)
        
        result = NLUResult(
            original_text=transcribed_text,
            intent=intent,
            confidence=intent_confidence,
            entities=entities,
            is_emergency=is_emergency,
            requires_disclaimer=requires_disclaimer,
            language_detected=detected_language
        )
        
        print(f"✅ NLU Result - Intent: {intent.value}, Confidence: {intent_confidence:.2%}")
        print(f"🚨 Emergency: {is_emergency}, Disclaimer: {requires_disclaimer}")
        
        return result
    
    def _detect_emergency(self, text: str, language: str) -> bool:
        """Detect emergency situations"""
        text_lower = text.lower()
        
        # Check language-specific emergency keywords
        lang_code = language.split('-')[0] if '-' in language else language
        emergency_words = self.emergency_keywords.get(lang_code, self.emergency_keywords['en'])
        
        for keyword in emergency_words:
            if keyword.lower() in text_lower:
                return True
                
        return False
    
    def _requires_medical_disclaimer(self, text: str) -> bool:
        """Check if query requires medical disclaimer"""
        # Any health-related query should have disclaimer
        # This is a conservative approach for healthcare
        return True
    
    def _classify_intent(self, text: str, language: str) -> Tuple[HealthIntent, float]:
        """Classify intent using real Sarvam-M API"""
        
        messages = [
            {
                "role": "system",
                "content": """You are a healthcare intent classifier. Classify user queries into these categories:
                
1. symptom_query - Questions about symptoms
2. disease_info - Information about diseases/conditions  
3. medication_info - Medicine-related queries
4. wellness_tip - Health and wellness advice
5. emergency - Urgent medical situations
6. diagnosis_request - Seeking medical diagnosis
7. prevention_info - Disease prevention information
8. general_health - General health questions

Respond ONLY with JSON format: {"intent": "category_name", "confidence": 0.95}"""
            },
            {
                "role": "user", 
                "content": f"Classify this healthcare query: '{text}'\nLanguage: {language}"
            }
        ]
        
        try:
            print(f"🔄 Calling Sarvam-M for intent classification...")
            response = self.sarvam_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                # Clean the response to extract JSON
                content = content.strip()
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].strip()
                
                result = json.loads(content)
                
                intent_str = result.get("intent", "unknown")
                confidence = result.get("confidence", 0.5)
                
                # Map to enum
                intent_mapping = {
                    "symptom_query": HealthIntent.SYMPTOM_QUERY,
                    "disease_info": HealthIntent.DISEASE_INFO,
                    "medication_info": HealthIntent.MEDICATION_INFO,
                    "wellness_tip": HealthIntent.WELLNESS_TIP,
                    "emergency": HealthIntent.EMERGENCY,
                    "diagnosis_request": HealthIntent.DIAGNOSIS_REQUEST,
                    "prevention_info": HealthIntent.PREVENTION_INFO,
                    "general_health": HealthIntent.GENERAL_HEALTH,
                }
                
                intent = intent_mapping.get(intent_str, HealthIntent.UNKNOWN)
                
                # Check for diagnosis request patterns as backup
                if self._is_diagnosis_request(text):
                    intent = HealthIntent.DIAGNOSIS_REQUEST
                    
                return intent, confidence
                
        except Exception as e:
            print(f"⚠️ Error in intent classification: {e}")
            
        return HealthIntent.UNKNOWN, 0.5
    
    def _extract_medical_entities(self, text: str, language: str) -> List[MedicalEntity]:
        """Extract medical entities using real Sarvam-M API"""
        
        messages = [
            {
                "role": "system",
                "content": """You are a medical entity extractor. Extract these entity types from healthcare queries:

- symptoms: fever, headache, cough, pain, etc.
- diseases: diabetes, hypertension, covid, etc.
- medications: paracetamol, metformin, aspirin, etc.
- body_parts: head, chest, stomach, heart, etc.
- medical_terms: blood pressure, sugar level, etc.

Respond ONLY with JSON format:
{"entities": [{"text": "fever", "type": "symptom", "start": 5, "end": 10, "confidence": 0.95}]}"""
            },
            {
                "role": "user",
                "content": f"Extract medical entities from: '{text}'\nLanguage: {language}"
            }
        ]
        
        entities = []
        
        try:
            print(f"🔄 Calling Sarvam-M for entity extraction...")
            response = self.sarvam_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                # Clean the response to extract JSON
                content = content.strip()
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].strip()
                
                result = json.loads(content)
                
                entity_list = result.get("entities", [])
                
                for entity_data in entity_list:
                    entity = MedicalEntity(
                        text=entity_data.get("text", ""),
                        entity_type=entity_data.get("type", "unknown"),
                        confidence=entity_data.get("confidence", 0.5),
                        start_pos=entity_data.get("start", 0),
                        end_pos=entity_data.get("end", 0)
                    )
                    entities.append(entity)
                    
        except Exception as e:
            print(f"⚠️ Error in entity extraction: {e}")
            
        return entities
    
    def _is_diagnosis_request(self, text: str) -> bool:
        """Check if text contains diagnosis request patterns"""
        text_lower = text.lower()
        
        for pattern in self.diagnosis_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
                
        return False
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text"""
        # Simplified language detection
        # In practice, use Sarvam-M's language detection capabilities
        
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', text))
        tamil_chars = bool(re.search(r'[\u0B80-\u0BFF]', text))
        telugu_chars = bool(re.search(r'[\u0C00-\u0C7F]', text))
        
        if hindi_chars:
            return "hi-IN"
        elif tamil_chars:
            return "ta-IN"
        elif telugu_chars:
            return "te-IN"
        else:
            return "en-IN"
            
# Integration with audio capture
def integrate_stt_nlu_pipeline():
    """Example integration of STT + NLU pipeline"""
    try:
        # For now, we'll use a mock transcription since audio_capture module isn't available
        # from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration
        
        # Initialize NLU processor
        nlu_processor = SarvamMNLUProcessor()
        
        print("🎤 Starting voice input simulation...")
        
        # Mock transcribed text for testing
        mock_transcriptions = [
            ("मुझे बुखार है", "hi-IN"),
            ("What are the symptoms of diabetes?", "en-IN"),
            ("Emergency! Chest pain!", "en-IN"),
        ]
        
        for transcribed_text, source_language in mock_transcriptions:
            print(f"\n📝 Processing: {transcribed_text}")
            
            # Step 3: NLU with Sarvam-M
            nlu_result = nlu_processor.process_transcription(
                transcribed_text,
                source_language=source_language
            )
            
            # Step 4: Handle results based on NLU
            if nlu_result.is_emergency:
                print("🚨 EMERGENCY DETECTED!")
                print("Response: Please call emergency services or visit nearest hospital immediately.")
                continue
            
            if nlu_result.intent == HealthIntent.DIAGNOSIS_REQUEST:
                print("⚠️ DIAGNOSIS REQUEST DETECTED!")
                print("Response: I cannot provide medical diagnosis. Please consult a qualified doctor.")
                continue
            
            print(f"✅ Intent: {nlu_result.intent.value}")
            print(f"📊 Entities found: {len(nlu_result.entities)}")
            for entity in nlu_result.entities:
                print(f"  - {entity.text} ({entity.entity_type})")
                
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Check if API key is set
    if not os.getenv("SARVAM_API_KEY"):
        print("❌ Please set SARVAM_API_KEY environment variable")
        print("Get your API key from: https://dashboard.sarvam.ai")
        print("\nExample:")
        print("export SARVAM_API_KEY='your_api_key_here'")
    else:
        integrate_stt_nlu_pipeline()