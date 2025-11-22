import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import edge_tts
import asyncio
import base64
import numpy as np
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
import speech_recognition as sr
import tempfile
import PyPDF2
from docx import Document

# 1. CONFIGURATION
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Green & Gold Guru",
    page_icon="favicon.png" if os.path.exists("favicon.png") else "🤘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for USF Branding & Dark Mode Fixes
st.markdown("""
    <style>
    .stButton>button {
        background-color: #006747;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005035;
        color: white;
    }
    h1, h2, h3 {
        color: #B8860B !important; 
        font-family: 'Arial', sans-serif;
    }
    .stAudioInput {
        border: 2px solid #006747;
        border-radius: 10px;
    }
    /* Canvas Toolbar Fix */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize OpenAI Client (OpenRouter)
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    st.error("⚠️ Please set OPENROUTER_API_KEY in .env file (or Secrets on Cloud).")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "USF Guru"}
)

# 2. SYSTEM PROMPT (Rocky Persona + USF Resources + Visuals)
SYSTEM_PROMPT = """
You are "Rocky the Bull", the energetic USF Mascot and Socratic Tutor.
You help students by guiding them, not just giving answers.

YOUR PERSONALITY:
- Energetic, supportive ("Go Bulls!", "Horns Up!").
- Patient but authoritative.

YOUR STRATEGY:
1. **Analyze:** Look at the whiteboard drawing and conversation.
2. **Guide:** Ask a leading question to help them solve it.
3. **Visuals:** If a concept needs a diagram, generate a tag: [Image Prompt: description of diagram].
4. **USF Resources:** If they seem stressed or stuck on a specific subject, recommend:
   - Math/Science -> SMART Lab (Library).
   - Writing -> Writing Studio.
   - Stress -> Wellness Center or Bull2Bull.
   - Late Night -> SAFE Team.

OUTPUT FORMAT:
1. **Visual Response:** - Use clear text. 
   - Use $$ \int x dx $$ for block math.
   - [Image Prompt: description] for diagrams.
2. **Audio Script:** Concise, natural, spoken directly to the student.
Separator: '---AUDIO---'
"""

# 3. HELPER FUNCTIONS

# --- FIXED AUDIO GENERATION FOR CLOUD ---
async def _run_edge_tts(text, output_file):
    # Voice: Christopher (Deep/Mascot-like)
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural", pitch="-5Hz", rate="+10%")
    await communicate.save(output_file)

def generate_audio(text):
    if not text or not text.strip():
        return None
    
    try:
        # Create a temporary file to avoid permission/concurrency issues in cloud
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            output_file = fp.name
            
        # Create a new event loop for this thread to avoid Streamlit conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_edge_tts(text, output_file))
        loop.close()
        
        return output_file
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None
# ----------------------------------------

def get_response(messages):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-2024-08-06", 
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def transcribe_audio(audio_file):
    try:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        os.unlink(tmp_path)
        return text
    except Exception:
        return "Error: Could not understand audio."

def read_file_content(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() for page in reader.pages])
        elif "docx" in uploaded_file.type:
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

def process_response(user_text, canvas_result):
    # 1. Capture Canvas
    img_str = None
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype("uint8")
        image = Image.fromarray(img_data)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

    # 2. Build Messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # History
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Context (File + User Input)
    full_prompt = user_text
    if "uploaded_file_content" in st.session_state:
        full_prompt = f"CONTEXT FROM UPLOADED FILE:\n{st.session_state.uploaded_file_content[:5000]}\n\nUSER QUESTION: {user_text}"

    # Payload
    content_payload = [{"type": "text", "text": full_prompt}]
    if img_str:
        content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})
    
    messages.append({"role": "user", "content": content_payload})
    
    # 3. AI Call
    with st.spinner("Rocky is thinking..."):
        response_text = get_response(messages)
    
    # 4. Parse Audio Separator
    if "---AUDIO---" in response_text:
        visual, audio = response_text.split("---AUDIO---")
    else:
        visual, audio = response_text, response_text

    # 5. Image Generation Parsing (Pollinations Flux)
    if "[Image Prompt:" in visual:
        import re
        match = re.search(r"\[Image Prompt: (.*?)\]", visual)
        if match:
            prompt = match.group(1)
            # The Flux URL with Style Injection
            img_url = f"https://image.pollinations.ai/prompt/{prompt}%20scientific%20vector%20diagram%20white%20background%20no%20text?width=1024&height=1024&model=flux"
            # Remove tag from text and store image
            visual = visual.replace(match.group(0), "")
            st.session_state.generated_image = img_url

    # 6. Save & Rerun
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": visual.strip(), "audio": audio.strip()})
    st.rerun()

# --- APP LAYOUT ---

col_header_1, col_header_2 = st.columns([1, 12])
with col_header_1:
    if os.path.exists("favicon.png"):
        st.image("favicon.png", width=90)
    elif os.path.exists("Rocky.png"):
        st.image("Rocky.png", width=90)
    else:
        st.write("🤘")
with col_header_2:
    st.title("Green & Gold Guru")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.info("🎙️ **Chat & Voice**")
    
    # File Upload
    with st.expander("📂 Upload Study Material"):
        uploaded_file = st.file_uploader("PDF, DOCX, TXT", type=["pdf", "docx", "txt"])
        if uploaded_file and ("last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name):
            content = read_file_content(uploaded_file)
            st.session_state.uploaded_file_content = content
            st.session_state.last_file = uploaded_file.name
            st.success("File loaded!")

    # Chat History
    container = st.container(height=400)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with container:
        for msg in st.session_state.messages:
            avatar = "favicon.png" if msg["role"] == "assistant" and os.path.exists("favicon.png") else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                # Show image if it was just generated and belongs to this message
                if msg == st.session_state.messages[-1] and "generated_image" in st.session_state:
                    st.image(st.session_state.generated_image)
                    del st.session_state.generated_image

    # Audio Input
    audio_value = st.audio_input("🎤 Talk to Rocky")
    if audio_value:
        # Process only if new
        audio_id = f"{audio_value.name}_{audio_value.size}"
        if "last_audio_id" not in st.session_state or st.session_state.last_audio_id != audio_id:
            st.session_state.last_audio_id = audio_id
            current_canvas = st.session_state.get("canvas_result", None)
            with st.spinner("Listening..."):
                text = transcribe_audio(audio_value)
                process_response(text, current_canvas)

    # Clear Button
    if st.button("🧹 Start New Session"):
        st.session_state.messages = []
        if "uploaded_file_content" in st.session_state: del st.session_state.uploaded_file_content
        st.rerun()

with col_right:
    st.warning("🎨 **Whiteboard** (Draw Math/Diagrams Here)")
    
    # Canvas Tools
    c1, c2, c3 = st.columns([1,1,1])
    with c1: color = st.color_picker("Pen", "#000000")
    with c2: width = st.slider("Width", 2, 10, 3)
    with c3: 
        if st.button("Submit Drawing"):
            process_response("Look at my drawing on the whiteboard.", st.session_state.get("canvas_result"))

    # Canvas
    canvas_result = st_canvas(
        stroke_width=width, stroke_color=color, background_color="#FFFFFF",
        height=550, width=800, drawing_mode="freedraw", key="canvas"
    )
    st.session_state.canvas_result = canvas_result

    # Text Input
    if prompt := st.chat_input("Type a message..."):
        process_response(prompt, canvas_result)

# Audio Playback (Running outside async loop)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    audio_script = last_msg.get("audio", "")
    
    if audio_script:
        msg_id = str(len(st.session_state.messages))
        # Check if already played
        if "last_played_id" not in st.session_state or st.session_state.last_played_id != msg_id:
            audio_file_path = generate_audio(audio_script)
            if audio_file_path:
                st.audio(audio_file_path, autoplay=True)
                st.session_state.last_played_id = msg_id
