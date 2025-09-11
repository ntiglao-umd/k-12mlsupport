import streamlit as st
# import google.generativeai as genai
import PyPDF2
from openai import OpenAI

# --- Load API key securely ---
# API_KEY = st.secrets["GEMINI_API_KEY"]
# genai.configure(api_key=API_KEY)

# def list_gemini_models():
#     for m in genai.list_models():
#         if 'generateContent' in m.supported_generation_methods:
#             print(m.name)
        
# --- Initialize Gemini model ---
#model = genai.GenerativeModel("gemini-pro")
# model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
HUGGINGFACEHUB_ACCESS_TOKEN = st.secrets["HUGGINGFACEHUB_ACCESS_TOKEN"]

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HUGGINGFACEHUB_ACCESS_TOKEN,
)


# --- Helper function to extract text from PDFs ---
def extract_text_from_pdfs(files):
    full_text = ""
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text.strip()

# --- UI ---
st.title("K-12 ML Support")
st.markdown("Upload **Knowledge PDFs** for context and **Lesson PDFs** to be revised based on that context.")

# --- Upload section for Knowledge PDFs ---
knowledge_files = st.file_uploader("📘 Upload Knowledge PDFs (for context)", type="pdf", accept_multiple_files=True)

# --- Upload section for Lesson PDFs ---
lesson_files = st.file_uploader("📗 Upload Lesson PDFs (to revise or enhance)", type="pdf", accept_multiple_files=True)

# --- Optional custom prompt ---
custom_instruction = st.text_area("✏️ Optional: Add custom instructions for revising the lesson", 
                                  placeholder="E.g., 'Make the lesson easier for high school students' or 'Align with AI ethics principles'")

# --- Trigger Gemini generation ---
if st.button("🔁 Revise Lessons"):
    if not knowledge_files:
        st.warning("Please upload at least one Knowledge PDF.")
    elif not lesson_files:
        st.warning("Please upload at least one Lesson PDF.")
    else:
        with st.spinner("Reading files and revising lessons..."):
            try:
                # Extract text from PDFs
                knowledge_text = extract_text_from_pdfs(knowledge_files)
                lesson_text = extract_text_from_pdfs(lesson_files)

                # Compose the RAG-enhanced prompt
                prompt = (
                    f"Based on the following knowledge documents:\n\n{knowledge_text}\n\n"
                    f"Please revise or enhance the following lesson materials:\n\n{lesson_text}\n\n"
                )

                prompt += f"""Context: Strive for high-quality instructional design with a preference for active, student-centered, inquiry-based, differentiated lessons.

                Assume the teacher would appreciate understanding the content, pedagogy, and technology principles that guide your recommendations. Explain your reasoning step by step, or “think aloud” before answering.

                Always keep the human at the center of this work. The human is a teacher who should be your thought partner in the process. Suggest ways to improve the instruction, but ensure the teacher has the final say in what is included in the lesson plan.

                There are different ways to design lessons. One approach is to use a gradual release of responsibility model, while another is to employ the 5E model. Teachers should be able to select the lesson model that best suits their preferences.

                Regardless of the lesson format or model, lessons should include:
                * Brief summary/description
                * Grade level(s)
                * Subject(s)
                * Estimated duration
                * Objectives
                * Standards
                * Technology used by students (if any)
                * Materials
                * Engaging Hook
                * Procedure
                * Assessment
                * Differentiation and Extension Suggestions
                * Glossary of key vocabulary
                * Citation or credits for the source(s) of the lesson

                When knowledge bases (i.e., documents incorporated as retrieval augmented generation sources) are provided in a prompt, use them explicitly and first. If insufficient suggestions are generated or if the user requests more, use the web to uncover additional ways to improve the lessons, thereby addressing the learner variability criteria exhibited by the class.

                """

                if custom_instruction.strip():
                    prompt += f"Instruction: {custom_instruction.strip()}"

                # Get response from Gemini
                # response = model.generate_content(prompt)

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b:novita",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )
                answer = response.choices[0].message.content
                
                # Display the result
                st.success("📝 Revised Lesson Content")
                st.markdown(answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
