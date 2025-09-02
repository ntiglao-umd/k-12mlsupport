import streamlit as st
import google.generativeai as genai
import PyPDF2
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import speech_recognition as sr
import numpy as np
import av

# --- Configure Gemini API ---
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
chat = model.start_chat()

# --- Helper: PDF text extraction ---
def extract_text_from_pdfs(files):
    full_text = ""
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text.strip()

# --- Helper: Transcribe voice ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_data = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.audio_data.append(audio)
        return frame

    def get_text(self):
        if self.audio_data:
            audio_bytes = np.concatenate(self.audio_data).tobytes()
            audio_file = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
            try:
                text = self.recognizer.recognize_google(audio_file)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio."
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"
        return ""

# --- UI Setup ---
st.title("📚 Gemini PDF Lesson Enhancer + Chat + Voice")
st.markdown("Upload **Knowledge PDFs** and **Lesson PDFs**, revise the lesson, and then refine via **chat or voice**.")

# --- Upload PDFs ---
knowledge_files = st.file_uploader("📘 Upload Knowledge PDFs (for context)", type="pdf", accept_multiple_files=True)
lesson_files = st.file_uploader("📗 Upload Lesson PDFs (to revise or enhance)", type="pdf", accept_multiple_files=True)
custom_instruction = st.text_area("✏️ Optional instructions", placeholder="E.g., Make lesson easier for high school")

# --- Global session state for output ---
if "lesson_output" not in st.session_state:
    st.session_state.lesson_output = ""

# --- Revise Button ---
if st.button("🔁 Revise Lessons"):
    if not knowledge_files or not lesson_files:
        st.warning("Upload both Knowledge and Lesson PDFs.")
    else:
        with st.spinner("📖 Reading and generating revised lesson..."):
            try:
                knowledge_text = extract_text_from_pdfs(knowledge_files)
                lesson_text = extract_text_from_pdfs(lesson_files)
                prompt = (
                    f"You are an expert educator. Revise the following lesson using the knowledge below.\n\n"
                    f"Knowledge:\n{knowledge_text}\n\n"
                    f"Lesson:\n{lesson_text}\n\n"
                )
                if custom_instruction.strip():
                    prompt += f"Instruction: {custom_instruction.strip()}"

                response = model.generate_content(prompt)
                output = response.text.strip()
                st.session_state.lesson_output = output

                st.success("📝 Revised Lesson")
                st.markdown(output)

                # Send initial message to chat history
                chat.send_message(prompt)

            except Exception as e:
                st.error(f"Error: {e}")

# --- Show Revised Output ---
if st.session_state.lesson_output:
    st.subheader("💬 Chat with Gemini to Refine Further")

    # Text chat input
    chat_input = st.text_input("Type a follow-up question or prompt", key="text_chat")
    if st.button("Send Text Prompt"):
        if chat_input.strip():
            with st.spinner("💡 Thinking..."):
                reply = chat.send_message(chat_input)
                st.markdown(f"**Gemini:** {reply.text}")

    # Voice chat input
    st.markdown("🎙️ Or use your voice:")
    audio_ctx = webrtc_streamer(key="speech", audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True, "video": False})
    if st.button("🎤 Transcribe and Send Voice Prompt"):
        if audio_ctx and audio_ctx.audio_processor:
            voice_text = audio_ctx.audio_processor.get_text()
            st.markdown(f"**You said:** {voice_text}")
            if voice_text.strip():
                with st.spinner("💬 Gemini is replying..."):
                    reply = chat.send_message(voice_text)
                    st.markdown(f"**Gemini:** {reply.text}")
