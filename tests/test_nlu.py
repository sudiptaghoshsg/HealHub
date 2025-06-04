from src.nlu_processor import SarvamMNLUProcessor

def test_nlu():
    try:
        nlu_processor = SarvamMNLUProcessor()
        
        # Test queries
        test_queries = [
            ("मुझे बुखार और सिरदर्द है", "hi-IN"), # Fever and Headache
            ("What is diabetes?", "en-IN"),
            ("chest pain emergency help", "en-IN"),
            ("Do I have COVID?", "en-IN"),
            ("मुझे सांस लेने में दिक्कत हो रही है और घरघराहट हो रही है", "hi-IN"), # Trouble breathing and wheezing
            ("I have severe chest pain and difficulty breathing", "en-IN"),
            ("আমি ঘন ঘন প্রস্রাব করছি এবং খুব তৃষ্ণার্ত অনুভব করছি", "bn-IN"),  # Frequent urination, thirst
            ("எனக்கு வாந்தி மற்றும் வயிற்று வலி உள்ளது", "ta-IN"),  # Vomiting and stomach pain
            ("నాకు జ్వరం మరియు శరీరంలో నొప్పి ఉంది", "te-IN"),  # Fever and body aches
            ("ನನಗೆ ಎದೆನೋವು ಮತ್ತು ಉಸಿರಾಟದಲ್ಲಿ ತೊಂದರೆ ಇದೆ", "kn-IN"),  # Chest pain and shortness of breath
            ("എനിക്ക് പലതവണ മൂത്രം വരുന്നു, അധികം ദാഹിക്കുന്നു", "ml-IN"),  # Frequent urination and thirst
        ]
        
        for query, lang in test_queries:
            print(f"\n--- Testing: '{query}' ---")
            result = nlu_processor.process_transcription(query, lang)
            
            print(f"Intent: {result.intent.value}")
            print(f"Emergency: {result.is_emergency}")
            print(f"Entities: {[e.text for e in result.entities]}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure to set your SARVAM_API_KEY environment variable")
        print("Get your API key from: https://dashboard.sarvam.ai")

if __name__ == "__main__":
    test_nlu()