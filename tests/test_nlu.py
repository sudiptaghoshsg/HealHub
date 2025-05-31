from src.nlu_processor import SarvamMNLUProcessor

def test_nlu():
    try:
        nlu_processor = SarvamMNLUProcessor()
        
        # Test queries
        test_queries = [
            ("मुझे बुखार और सिरदर्द है", "hi-IN"),
            ("What is diabetes?", "en-IN"),
            ("chest pain emergency help", "en-IN"),
            ("Do I have COVID?", "en-IN")
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