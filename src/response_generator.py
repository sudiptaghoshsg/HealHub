import json
from typing import Optional

from src.nlu_processor import NLUResult, HealthIntent, SarvamAPIClient
from src.prompts import HEALTHCARE_SYSTEM_PROMPT

class HealHubResponseGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.sarvam_client = SarvamAPIClient(api_key=api_key)

    def _get_hardcoded_safety_response(self, nlu_result: NLUResult) -> Optional[str]:
        """
        Provides immediate hardcoded responses for critical safety scenarios
        based on NLU output.
        """
        # Determine language for response (simplified, assumes NLU provides it or defaults)
        lang = nlu_result.language_detected.split('-')[0] if nlu_result.language_detected else "en"

        if nlu_result.is_emergency:
            if lang == "hi":
                return "आपके द्वारा बताए गए लक्षण गंभीर लग रहे हैं और इसके लिए तत्काल चिकित्सा ध्यान देने की आवश्यकता हो सकती है। कृपया तुरंत डॉक्टर से सलाह लें या नजदीकी अस्पताल जाएँ। मैं आपातकालीन चिकित्सा सहायता प्रदान करने के लिए सुसज्जित नहीं हूँ।"
            else: # Default to English
                return "The symptoms you're describing sound serious and may require immediate medical attention. Please consult a doctor or go to the nearest hospital right away. I am not equipped to provide emergency medical assistance."

        if nlu_result.intent == HealthIntent.DIAGNOSIS_REQUEST:
            if lang == "hi":
                return "मैं समझता/सकती हूँ कि आप उत्तर ढूंढ रहे हैं, लेकिन मैं मेडिकल निदान प्रदान नहीं कर सकता/सकती। किसी भी स्वास्थ्य चिंता या निदान के लिए, कृपया एक योग्य स्वास्थ्य पेशेवर से सलाह लें।"
            else: # Default to English
                return "I understand you're looking for answers, but I cannot provide a medical diagnosis. For any health concerns or to get a diagnosis, it's very important to consult a qualified healthcare professional."

        if nlu_result.intent == HealthIntent.MEDICATION_INFO and "advice" in nlu_result.original_text.lower(): # Simple check
             # More robust check for treatment/medication advice needed in NLU
            if lang == "hi":
                return "मैं उपचार सलाह या विशिष्ट दवाएं सुझाने में असमर्थ हूँ। उपचार, दवाओं या अपनी स्वास्थ्य स्थिति के प्रबंधन के बारे में किसी भी प्रश्न के लिए, कृपया अपने डॉक्टर या एक योग्य स्वास्थ्य सेवा प्रदाता से सलाह लें।"
            else: # Default to English
                return "I am unable to offer treatment advice or suggest specific medications. Please consult with your doctor or a qualified healthcare provider for any questions about treatments, medications, or managing your health condition."
        return None

    def generate_response(self, user_query: str, nlu_result: NLUResult) -> str:
        """
        Generates a response based on the user query and NLU result.
        Applies a two-layer safety check.
        """
        # Layer 1: Application-level hardcoded safety responses
        safety_response = self._get_hardcoded_safety_response(nlu_result)
        if safety_response:
            print("ℹ️ Applying hardcoded safety response.")
            return safety_response

        # Layer 2: LLM-level response generation with comprehensive system prompt
        print(f"💬 Generating response for query: '{user_query}' using LLM.")
        
        # Construct messages for the LLM
        # The NLU result can be passed to the LLM for more context if needed,
        # but the system prompt already guides it extensively.
        # For now, we'll just pass the user query.
        # You could enhance this by adding a summary of NLU findings to the user message.
        messages = [
            {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: \"{user_query}\"\nDetected language: {nlu_result.language_detected}\nNLU Intent: {nlu_result.intent.value}\nNLU Entities: {[e.text for e in nlu_result.entities]}"}
        ]

        try:
            llm_response_data = self.sarvam_client.chat_completion(
                messages=messages,
                temperature=0.5, # Adjust for desired creativity/factuality
                max_tokens=500  # Adjust as needed
            )

            if llm_response_data and "choices" in llm_response_data and llm_response_data["choices"]:
                generated_text = llm_response_data["choices"][0]["message"]["content"]
                # The system prompt instructs the LLM to include disclaimers.
                return generated_text.strip()
            else:
                print("⚠️ LLM response was empty or malformed.")
                # Fallback response if LLM fails
                if nlu_result.language_detected.startswith("hi"):
                    return "माफ़ कीजिए, मैं अभी आपकी मदद नहीं कर सकता। कृपया बाद में प्रयास करें।"
                return "Sorry, I am unable to assist you at the moment. Please try again later."

        except Exception as e:
            print(f"❌ Error during LLM call: {e}")
            if nlu_result.language_detected.startswith("hi"):
                return "क्षमा करें, प्रतिक्रिया उत्पन्न करते समय एक त्रुटि हुई।"
            return "Sorry, an error occurred while generating the response."