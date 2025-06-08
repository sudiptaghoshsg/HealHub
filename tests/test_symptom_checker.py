import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Add src to sys.path to allow direct import of src modules
# This is a common way to handle imports in tests when tests are outside the src package
# and not run as part of an installed package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.symptom_checker import SymptomChecker
    # Assuming HealthIntent, MedicalEntity, NLUResult are available via src.nlu_processor
    # or through symptom_checker.py's own placeholders if nlu_processor isn't fully there.
    from src.nlu_processor import NLUResult, MedicalEntity, HealthIntent
except ImportError as e:
    print(f"Failed to import from src: {e}. Ensure PYTHONPATH is set up correctly or tests are run as a module.")
    # Fallback for environments where src is not directly on path
    # This requires symptom_checker.py to have its own robust placeholders if this path is taken.
    # For this subtask, assume src.symptom_checker can be imported.
    # If there's an issue, the subtask runner might need to adjust sys.path or use relative imports if possible.
    # However, direct imports are cleaner if the execution environment (like a tox setup) handles paths.
    # For now, let's assume the import from src works. If not, the execution log will show it.
    # As a last resort, we might need to define minimal placeholders here, but that's not ideal.
    # Re-raising to make it clear if imports fail.
    raise


# Dummy KB content for tests
DUMMY_KB_CONTENT = {
    "symptoms": [
        {
            "symptom_name": "fever", # Stored as lowercase key in SymptomChecker
            "keywords": ["temperature", "hot"],
            "follow_up_questions": ["How high is the fever?", "Any chills?"],
            "basic_triage_points": ["Fever > 3 days needs check.", "High fever is concerning."]
        },
        {
            "symptom_name": "cough", # Stored as lowercase key
            "keywords": ["hacking"],
            "follow_up_questions": ["Is it a dry cough or wet cough?", "How long have you had it?"],
            "basic_triage_points": ["Persistent cough needs check.", "Coughing blood is urgent."]
        },
        {
            "symptom_name": "headache", # Stored as lowercase key
            "keywords": ["head pain"],
            "follow_up_questions": ["Where is the headache located?", "Is it throbbing?"],
            "basic_triage_points": ["Sudden severe headache is urgent."]
        }
    ]
}

# Utility to create NLUResult for tests
def create_mock_nlu_result(text="test query", intent_val=HealthIntent.SYMPTOM_QUERY, entities_data=None):
    intent_enum_val = intent_val if isinstance(intent_val, HealthIntent) else HealthIntent.UNKNOWN

    mock_entities = []
    if entities_data:
        for ent_text, ent_type in entities_data:
            mock_entities.append(MedicalEntity(text=ent_text, entity_type=ent_type, confidence=0.9, start_pos=0, end_pos=len(ent_text)))

    return NLUResult(
        original_text=text,
        intent=intent_enum_val,
        entities=mock_entities,
        confidence=0.9,
        is_emergency=False,
        requires_disclaimer=True,
        language_detected="en-IN"
    )

class TestSymptomChecker(unittest.TestCase):

    def setUp(self):
        # Ensure the tests directory exists for placing test files
        if not os.path.exists("tests"):
            os.makedirs("tests", exist_ok=True)
        self.dummy_kb_path = "tests/test_symptom_kb.json" # Place dummy file in tests directory
        self.malformed_kb_path = "tests/malformed_kb.json"

        self._create_dummy_kb_file(self.dummy_kb_path, DUMMY_KB_CONTENT)
        self.mock_api_key = "test_api_key_123"

    def tearDown(self):
        if os.path.exists(self.dummy_kb_path):
            os.remove(self.dummy_kb_path)
        if os.path.exists(self.malformed_kb_path):
            os.remove(self.malformed_kb_path)

    def _create_dummy_kb_file(self, filepath, content):
        with open(filepath, 'w') as f:
            json.dump(content, f)

    def test_initialization_and_kb_loading(self):
        checker = SymptomChecker(create_mock_nlu_result(), api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        self.assertIsNotNone(checker.symptom_kb)
        self.assertTrue("fever" in checker.symptom_kb) # KB keys are lowercased

        # Test with non-existent KB
        checker_no_kb = SymptomChecker(create_mock_nlu_result(), api_key=self.mock_api_key, symptom_kb_path="tests/non_existent_kb.json")
        self.assertEqual(checker_no_kb.symptom_kb, {})

        # Test with malformed JSON
        with open(self.malformed_kb_path, 'w') as f:
            f.write("{'symptoms': [}") # Invalid JSON
        checker_malformed_kb = SymptomChecker(create_mock_nlu_result(), api_key=self.mock_api_key, symptom_kb_path=self.malformed_kb_path)
        self.assertEqual(checker_malformed_kb.symptom_kb, {})

    def test_identify_relevant_symptoms(self):
        # Test with direct match (fever) and one non-KB symptom (sore throat)
        checker = SymptomChecker(create_mock_nlu_result(entities_data=[("fever", "symptom"), ("sore throat", "symptom")]),
                                 api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        relevant = checker.identify_relevant_symptoms()
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0]["symptom_name"].lower(), "fever") # Compare with KB's symptom_name, which is then lowercased for key

        # Test with keyword match ("hot" for "fever")
        checker_keyword = SymptomChecker(create_mock_nlu_result(entities_data=[("feeling hot", "symptom")]),
                                         api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        relevant_keyword = checker_keyword.identify_relevant_symptoms()
        self.assertEqual(len(relevant_keyword), 1)
        self.assertEqual(relevant_keyword[0]["symptom_name"].lower(), "fever")

        # Test with no matching symptoms
        checker_none = SymptomChecker(create_mock_nlu_result(entities_data=[("feeling great", "general_feeling")]),
                                      api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        self.assertEqual(len(checker_none.identify_relevant_symptoms()), 0)

    def test_prepare_follow_up_questions(self):
        checker = SymptomChecker(create_mock_nlu_result(entities_data=[("fever", "symptom"), ("cough", "symptom")]),
                                 api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        checker.prepare_follow_up_questions()
        # DUMMY_KB_CONTENT: fever (2) + cough (2) = 4 questions
        self.assertEqual(len(checker.pending_follow_up_questions), 4)

        # Test no re-adding if details collected
        # SymptomChecker stores collected_symptom_details keys as lowercase
        checker.collected_symptom_details["fever"] = {"How high is the fever?": "102"}
        checker.prepare_follow_up_questions()
        # Should now only have cough's 2 questions as fever details exist
        self.assertEqual(len(checker.pending_follow_up_questions), 2)
        q_texts = [q['question'] for q in checker.pending_follow_up_questions]
        self.assertNotIn("How high is the fever?", q_texts)
        self.assertIn("Is it a dry cough or wet cough?", q_texts) # cough question should be there

    def test_get_next_question_and_record_answer(self):
        checker = SymptomChecker(create_mock_nlu_result(entities_data=[("headache", "symptom")]),
                                 api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        checker.prepare_follow_up_questions() # headache has 2 questions

        q1_data = checker.get_next_question()
        self.assertIsNotNone(q1_data)
        self.assertEqual(q1_data['symptom_name'].lower(), "headache")
        checker.record_answer(q1_data['symptom_name'], q1_data['question'], "forehead")

        q2_data = checker.get_next_question()
        self.assertIsNotNone(q2_data)
        checker.record_answer(q2_data['symptom_name'], q2_data['question'], "yes, throbbing")

        self.assertIsNone(checker.get_next_question()) # No more questions
        self.assertTrue("headache" in checker.collected_symptom_details) # Key is lowercased
        self.assertEqual(len(checker.collected_symptom_details["headache"]), 2)

    @patch('src.symptom_checker.SarvamAPIClient') # Target where SarvamAPIClient is used
    def test_generate_preliminary_assessment_llm_success(self, MockSarvamAPIClient):
        mock_api_instance = MockSarvamAPIClient.return_value
        mock_llm_response_content = { # This is the content of the "content" field
            "assessment_summary": "User reports fever and cough.",
            "suggested_severity": "May require attention",
            "recommended_next_steps": "Rest and hydrate. Consult a doctor if worsens.",
            "potential_warnings": ["Fever present"],
            "disclaimer": "This is a test disclaimer that will be overwritten."
        }
        mock_api_instance.chat_completion.return_value = { # This is the full API response
            "choices": [{"message": {"content": json.dumps(mock_llm_response_content)}}]
        }

        nlu_res = create_mock_nlu_result(entities_data=[("fever", "symptom"), ("cough", "symptom")])
        checker = SymptomChecker(nlu_res, api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        checker.record_answer("fever", "How high is the fever?", "101F") # Stored as 'fever'
        checker.record_answer("cough", "Is it a dry cough or wet cough?", "Dry") # Stored as 'cough'

        assessment = checker.generate_preliminary_assessment()

        mock_api_instance.chat_completion.assert_called_once()
        self.assertEqual(assessment["assessment_summary"], mock_llm_response_content["assessment_summary"])
        self.assertEqual(assessment["disclaimer"], SymptomChecker.DEFAULT_ASSESSMENT_ERROR["disclaimer"])
        self.assertTrue(len(assessment["relevant_kb_triage_points"]) > 0)
        # Check for points from both 'fever' and 'cough' based on DUMMY_KB_CONTENT
        self.assertIn("Fever > 3 days needs check.", assessment["relevant_kb_triage_points"])
        self.assertIn("Persistent cough needs check.", assessment["relevant_kb_triage_points"])

    @patch('src.symptom_checker.SarvamAPIClient')
    def test_generate_preliminary_assessment_llm_failure_json_decode(self, MockSarvamAPIClient):
        mock_api_instance = MockSarvamAPIClient.return_value
        mock_api_instance.chat_completion.return_value = {
            "choices": [{"message": {"content": "this is not json"}}]
        }
        checker = SymptomChecker(create_mock_nlu_result(entities_data=[("fever", "symptom")]),
                                 api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        assessment = checker.generate_preliminary_assessment()
        self.assertEqual(assessment["assessment_summary"], SymptomChecker.DEFAULT_ASSESSMENT_ERROR["assessment_summary"])
        # Ensure default error structure is returned
        self.assertDictEqual(assessment, SymptomChecker.DEFAULT_ASSESSMENT_ERROR)


    @patch('src.symptom_checker.SarvamAPIClient')
    def test_generate_preliminary_assessment_llm_missing_keys(self, MockSarvamAPIClient):
        mock_api_instance = MockSarvamAPIClient.return_value
        mock_llm_response_content = {"summary_is_wrong_key": "User reports fever."} # Missing required keys
        mock_api_instance.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_llm_response_content)}}]
        }
        checker = SymptomChecker(create_mock_nlu_result(entities_data=[("fever", "symptom")]),
                                 api_key=self.mock_api_key, symptom_kb_path=self.dummy_kb_path)
        assessment = checker.generate_preliminary_assessment()
        self.assertEqual(assessment["assessment_summary"], SymptomChecker.DEFAULT_ASSESSMENT_ERROR["assessment_summary"])
        self.assertTrue(any("missing key" in warning for warning in assessment["potential_warnings"]))

    @patch('src.symptom_checker.SarvamAPIClient')
    def test_generate_preliminary_assessment_no_api_key(self, MockSarvamAPIClient):
        # This test assumes SymptomChecker instantiates SarvamAPIClient,
        # and SarvamAPIClient (or its placeholder) sets self.api_key based on input.
        # The SymptomChecker.generate_preliminary_assessment checks getattr(self.sarvam_client, 'api_key', None)

        # Configure the mock class's instance to have api_key = None
        mock_client_instance = MockSarvamAPIClient.return_value
        mock_client_instance.api_key = None

        # Initialize SymptomChecker with api_key=None
        # This should lead to checker.sarvam_client.api_key being None
        checker = SymptomChecker(create_mock_nlu_result(), api_key=None, symptom_kb_path=self.dummy_kb_path)

        # Verify that the SarvamAPIClient instance within checker has no API key
        self.assertIsNotNone(checker.sarvam_client) # Client is instantiated
        self.assertIsNone(checker.sarvam_client.api_key) # But its api_key attribute is None

        assessment = checker.generate_preliminary_assessment()

        self.assertEqual(assessment["assessment_summary"], SymptomChecker.DEFAULT_ASSESSMENT_ERROR["assessment_summary"])
        # Ensure chat_completion was not called because of the missing API key
        mock_client_instance.chat_completion.assert_not_called()

if __name__ == '__main__':
    unittest.main(verbosity=2)
