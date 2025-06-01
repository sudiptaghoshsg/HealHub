import streamlit as st
import os
import json
import time # For polling audio capture status
import numpy as np # For checking audio data (though not directly used in this version)
from dotenv import load_dotenv
from typing import Optional

# Adjust import paths
try:
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration # Import audio modules
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration

# --- Environment and API Key Setup ---
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# --- Session State Initialization ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_language_display' not in st.session_state: 
    st.session_state.current_language_display = 'English'
if 'current_language_code' not in st.session_state: 
    st.session_state.current_language_code = 'en-IN'
if 'text_query_input_area' not in st.session_state:
    st.session_state.text_query_input_area = ""

# Symptom Checker states
if 'symptom_checker_active' not in st.session_state:
    st.session_state.symptom_checker_active = False
if 'symptom_checker_instance' not in st.session_state:
    st.session_state.symptom_checker_instance = None
if 'pending_symptom_question_data' not in st.session_state:
    st.session_state.pending_symptom_question_data = None

# Voice Input states
if 'voice_input_stage' not in st.session_state:
    # Stages: None, "arming", "recording", "transcribing", "processing_stt"
    st.session_state.voice_input_stage = None 
if 'audio_capturer' not in st.session_state: 
    st.session_state.audio_capturer = None
if 'captured_audio_data' not in st.session_state:
    st.session_state.captured_audio_data = None

# --- Language Mapping ---
LANGUAGE_MAP = {
    "English": "en-IN", "हिन्दी (Hindi)": "hi-IN", "বাংলা (Bengali)": "bn-IN", "मराठी (Marathi)": "mr-IN"
}
DISPLAY_LANGUAGES = list(LANGUAGE_MAP.keys())

# --- Helper Functions ---
def add_message_to_conversation(role: str, content: str, lang_code: Optional[str] = None):
    message = {"role": role, "content": content}
    if lang_code and role == "user":
        message["lang"] = lang_code 
    st.session_state.conversation.append(message)

def process_and_display_response(user_query_text: str, lang_code: str):
    if not SARVAM_API_KEY:
        st.error("API Key not configured.")
        add_message_to_conversation("system", "Error: API Key not configured.")
        st.session_state.voice_input_stage = None # Reset voice stage on error
        return

    nlu_processor = SarvamMNLUProcessor(api_key=SARVAM_API_KEY)
    response_gen = HealHubResponseGenerator(api_key=SARVAM_API_KEY)
    try:
        # User message is now added *before* calling this function for both text and voice.
        # So, this function should not add the user message again.
        
        with st.spinner("🧠 Thinking..."):
            nlu_output: NLUResult = nlu_processor.process_transcription(user_query_text, source_language=lang_code)
            if nlu_output.intent == HealthIntent.SYMPTOM_QUERY and not nlu_output.is_emergency:
                st.session_state.symptom_checker_active = True
                st.session_state.symptom_checker_instance = SymptomChecker(nlu_result=nlu_output, api_key=SARVAM_API_KEY)
                st.session_state.symptom_checker_instance.prepare_follow_up_questions()
                st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
                if st.session_state.pending_symptom_question_data:
                    question_to_ask = st.session_state.pending_symptom_question_data['question']
                    symptom_context = st.session_state.pending_symptom_question_data['symptom_name']
                    add_message_to_conversation("assistant", f"Regarding {symptom_context}: {question_to_ask}")
                else:
                    generate_and_display_assessment()
            else:
                bot_response = response_gen.generate_response(user_query_text, nlu_output)
                add_message_to_conversation("assistant", bot_response)
                st.session_state.symptom_checker_active = False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Standardized error message
        add_message_to_conversation("system", f"Sorry, an error occurred while processing your request. Please try rephrasing or try again later. (Details: {str(e)})")
        st.session_state.symptom_checker_active = False # Reset states on error
        st.session_state.symptom_checker_instance = None
        st.session_state.pending_symptom_question_data = None
    finally:
        st.session_state.voice_input_stage = None # Always reset voice stage after processing or error

def handle_follow_up_answer(answer_text: str):
    if st.session_state.symptom_checker_instance and st.session_state.pending_symptom_question_data:
        # Add user's follow-up answer to conversation log
        add_message_to_conversation("user", answer_text, lang_code=st.session_state.current_language_code.split('-')[0])
        
        question_asked = st.session_state.pending_symptom_question_data['question']
        symptom_name = st.session_state.pending_symptom_question_data['symptom_name']
        with st.spinner("Recording answer..."):
            st.session_state.symptom_checker_instance.record_answer(symptom_name, question_asked, answer_text)
            st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
        if st.session_state.pending_symptom_question_data:
            question_to_ask = st.session_state.pending_symptom_question_data['question']
            symptom_context = st.session_state.pending_symptom_question_data['symptom_name']
            add_message_to_conversation("assistant", f"Regarding {symptom_context}: {question_to_ask}")
        else:
            generate_and_display_assessment()
    else: 
        st.warning("No pending question to answer or symptom checker not active.")
        st.session_state.symptom_checker_active = False
    st.session_state.voice_input_stage = None # Reset voice stage

# New callback function for text submission
def handle_text_submission():
    user_input = st.session_state.text_query_input_area # Read from session state key
    current_lang_code = st.session_state.current_language_code

    if not user_input: # Do nothing if input is empty
        return

    # Add the current user input to conversation log REGARDLESS of whether it's new or follow-up
    add_message_to_conversation("user", user_input, lang_code=current_lang_code.split('-')[0])

    if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data:
        # handle_follow_up_answer will process the answer.
        # It should NOT add the user message again as it's already added above.
        handle_follow_up_answer(user_input) 
    else: 
        if st.session_state.symptom_checker_active: # Reset if symptom checker was active but no pending q
             st.session_state.symptom_checker_active = False 
             st.session_state.symptom_checker_instance = None
             st.session_state.pending_symptom_question_data = None
        # process_and_display_response will process the new query.
        # It should NOT add the user message again.
        process_and_display_response(user_input, current_lang_code)
    
    st.session_state.text_query_input_area = "" # Clear the text area state for next render
    # No explicit st.rerun() here, on_click handles it for buttons.
    # If called from a non-button context that needs immediate UI update, rerun might be needed.

def generate_and_display_assessment():
    if st.session_state.symptom_checker_instance:
        with st.spinner("🔬 Generating preliminary assessment..."):
            assessment = st.session_state.symptom_checker_instance.generate_preliminary_assessment()
            try:
                assessment_str = "### Preliminary Health Assessment:\n\n"
                assessment_str += f"**Summary:** {assessment.get('assessment_summary', 'N/A')}\n\n"
                assessment_str += f"**Suggested Severity:** {assessment.get('suggested_severity', 'N/A')}\n\n"
                assessment_str += "**Recommended Next Steps:**\n"
                next_steps = assessment.get('recommended_next_steps', 'N/A')
                if isinstance(next_steps, list): 
                    for step in next_steps: assessment_str += f"- {step}\n"
                elif isinstance(next_steps, str): # This is the block to modify
                    # Replace the original problematic f-string line here
                    temp_steps = next_steps.replace('. ', '.\n- ') 
                    temp_steps = temp_steps.strip().lstrip('- ')    
                    assessment_str += f"{temp_steps}\n"
                else: assessment_str += "- N/A\n"
                warnings = assessment.get('potential_warnings')
                if warnings and isinstance(warnings, list) and len(warnings) > 0 :
                    assessment_str += "\n**Potential Warnings:**\n"
                    for warning in warnings: assessment_str += f"- {warning}\n"
                kb_points = assessment.get('relevant_kb_triage_points')
                if kb_points and isinstance(kb_points, list) and len(kb_points) > 0:
                    assessment_str += "\n**Relevant Triage Points from Knowledge Base:**\n"
                    for point in kb_points: assessment_str += f"- {point}\n"
                assessment_str += f"\n\n**Disclaimer:** {assessment.get('disclaimer', 'Always consult a doctor for medical advice.')}"
                add_message_to_conversation("assistant", assessment_str)
            except Exception as e:
                st.error(f"Error formatting assessment: {e}")
                try:
                    raw_assessment_json = json.dumps(assessment, indent=2)
                    add_message_to_conversation("assistant", f"Could not format assessment. Raw data:\n```json\n{raw_assessment_json}\n```")
                except Exception as json_e:
                    add_message_to_conversation("assistant", f"Could not format or serialize assessment: {json_e}")
        st.session_state.symptom_checker_active = False
        st.session_state.symptom_checker_instance = None
        st.session_state.pending_symptom_question_data = None
    st.session_state.voice_input_stage = None # Reset voice stage

# --- Streamlit UI ---
def main_ui():
    st.set_page_config(page_title="HealHub Assistant", layout="wide")
    st.title("💬 HealHub Assistant")
    st.caption("Your AI healthcare companion. Supporting English, Hindi, Bengali, and Marathi.")

    # Enhanced Recording Visual Cue
    if st.session_state.voice_input_stage == "recording":
        st.info("🔴 Recording audio... Speak now. Silence or the 'Stop' button will end recording.", icon="🎤")

    if not SARVAM_API_KEY: 
        st.error("🚨 SARVAM_API_KEY not found. Please set it in your .env file for the application to function.")
        st.stop()

    selected_lang_display = st.selectbox(
        "Select Language / भाषा चुनें / ভাষা নির্বাচন করুন / भाषा निवडा:",
        options=DISPLAY_LANGUAGES,
        index=DISPLAY_LANGUAGES.index(st.session_state.current_language_display),
        key='language_selector_widget' 
    )
    if selected_lang_display != st.session_state.current_language_display:
        st.session_state.current_language_display = selected_lang_display
        st.session_state.current_language_code = LANGUAGE_MAP[selected_lang_display]
        st.session_state.conversation = [] 
        st.session_state.symptom_checker_active = False; st.session_state.symptom_checker_instance = None; st.session_state.pending_symptom_question_data = None
        st.session_state.voice_input_stage = None
        st.rerun()

    current_lang_code_for_query = st.session_state.current_language_code

    st.markdown("### Conversation")
    chat_container = st.container(height=500) 
    with chat_container:
        for msg_data in st.session_state.conversation:
            role = msg_data.get("role", "system"); content = msg_data.get("content", "")
            avatar = "🧑‍💻" if role == "user" else "⚕️"
            if role == "user":
                with st.chat_message(role, avatar=avatar):
                    lang_display = msg_data.get('lang', st.session_state.current_language_code.split('-')[0])
                    st.markdown(f"{content} *({lang_display})*")
            else:
                 with st.chat_message(role, avatar=avatar if role=="assistant" else "ℹ️"): st.markdown(content) 

    st.markdown("---")
    is_recording = st.session_state.voice_input_stage == "recording"
    input_label = "Type your answer here..." if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data else "Type your health query here..."
    
    # Text area widget - its current value is stored in st.session_state.text_query_input_area due to its key
    st.text_area(input_label, height=100, key="text_query_input_area", disabled=is_recording)
    
    col1, col2 = st.columns([3,1]) 
    with col1:
        # Send button now uses the on_click callback
        st.button(
            "✉️ Send", 
            use_container_width=True, 
            key="send_button_widget", # Key can be kept if useful for other logic, or removed
            disabled=is_recording,
            on_click=handle_text_submission # Assign the callback
        )
    with col2:
        record_voice_button_text = "🔴 Stop & Process" if is_recording else "🎤 Record Voice"
        record_voice_button = st.button(record_voice_button_text, use_container_width=True, key="record_voice_button_widget")

    # --- Voice Input Logic (remains largely the same, but text submission is handled by callback) ---
    if record_voice_button: # This handles the click of the voice button
        if is_recording: 
            if st.session_state.audio_capturer:
                st.session_state.audio_capturer.stop_recording() 
            st.session_state.voice_input_stage = "transcribing" 
            st.rerun()
        else: 
            st.session_state.voice_input_stage = "recording"
            st.session_state.captured_audio_data = None 
            if st.session_state.audio_capturer is None: 
                 st.session_state.audio_capturer = CleanAudioCapture(sample_rate=16000)
            try:
                st.session_state.audio_capturer.start_recording()
                if not st.session_state.conversation or st.session_state.conversation[-1].get("content") != "🎤 Voice recording started... Speak now. Silence will stop it, or click 'Stop & Process'.":
                    add_message_to_conversation("system", "🎤 Voice recording started... Speak now. Silence will stop it, or click 'Stop & Process'.")
            except Exception as e:
                st.error(f"Failed to start recording: {e}. Ensure microphone is connected and permissions are granted.")
                add_message_to_conversation("system", f"Error: Could not start voice recording. Please check microphone permissions. (Details: {e})")
                st.session_state.voice_input_stage = None
            st.rerun()

    # State machine for voice processing (remains the same)
    if st.session_state.voice_input_stage == "recording":
        if st.session_state.audio_capturer and not st.session_state.audio_capturer.is_recording:
            st.session_state.voice_input_stage = "transcribing"
            st.rerun()
        else:
            time.sleep(0.1) 
            if st.session_state.audio_capturer and not st.session_state.audio_capturer.is_recording: # Check again
                 st.session_state.voice_input_stage = "transcribing"
            st.rerun() 

    if st.session_state.voice_input_stage == "transcribing":
        cleaned_audio = None
        if st.session_state.audio_capturer:
            cleaned_audio = st.session_state.audio_capturer.get_cleaned_audio()
        if cleaned_audio is not None and len(cleaned_audio) > 0:
            if not st.session_state.conversation or st.session_state.conversation[-1].get("content") != "🎙️ Audio captured. Transcribing...":
                add_message_to_conversation("system", "🎙️ Audio captured. Transcribing...")
            st.session_state.captured_audio_data = cleaned_audio 
            st.session_state.voice_input_stage = "processing_stt"
        else:
            add_message_to_conversation("system", "⚠️ No valid audio captured or VAD stopped too early. Please try again.")
            st.session_state.voice_input_stage = None
        st.rerun()

    if st.session_state.voice_input_stage == "processing_stt":
        if st.session_state.captured_audio_data is not None:
            stt_service = SarvamSTTIntegration(api_key=SARVAM_API_KEY)
            lang_for_stt = st.session_state.current_language_code 
            try:
                with st.spinner("Transcribing audio..."):
                    stt_result = stt_service.transcribe_audio(
                        st.session_state.captured_audio_data, sample_rate=16000, source_language=lang_for_stt
                    )
                transcribed_text = stt_result.get("transcription")
                if transcribed_text and transcribed_text.strip():
                    add_message_to_conversation("user", transcribed_text, lang_code=lang_for_stt.split('-')[0])
                    process_and_display_response(transcribed_text, lang_for_stt) 
                else:
                    add_message_to_conversation("system", "⚠️ STT failed to transcribe audio or returned empty. Please try again.")
            except Exception as e:
                st.error(f"STT Error: {e}")
                add_message_to_conversation("system", f"Sorry, an error occurred during voice transcription. Please try again. (Details: {e})")
            st.session_state.captured_audio_data = None 
            st.session_state.voice_input_stage = None 
            st.rerun()
        else: 
            st.session_state.voice_input_stage = None
            st.rerun()

    # The old `if send_button and user_query_text_from_area:` block is now removed,
    # as its logic is handled by the handle_text_submission callback.

if __name__ == "__main__":
    main_ui()
