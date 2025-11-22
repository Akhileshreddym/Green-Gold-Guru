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

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Green & Gold Guru",
    page_icon="🤘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for USF Branding
st.markdown("""
    <style>
    /* .stApp {
        background-color: #FFFFFF;
    } */
    .stButton>button {
        background-color: #006747;
        color: white;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #005035;
        color: white;
    }
    h1, h2, h3 {
        color: #CFC493;
        font-family: 'Arial', sans-serif;
    }
    .stAudioInput {
        border: 2px solid #006747;
        border-radius: 10px;
    }
    /* Fix for canvas toolbar in dark mode */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        background-color: white;
        border-radius: 5px;
    }
    </style>
    <script>
    // Inject CSS into canvas iframe for red toolbar icons
    function styleCanvasToolbar() {
        const iframe = document.querySelector('iframe[title="streamlit_drawable_canvas.st_canvas"]');
        if (iframe && iframe.contentDocument) {
            const style = iframe.contentDocument.createElement('style');
            style.textContent = `
                button, svg {
                    filter: brightness(0) invert(1) !important;
                }
                button:disabled, button:disabled svg {
                    filter: brightness(0) invert(0.7) !important;
                }
            `;
            if (!iframe.contentDocument.querySelector('#custom-toolbar-style')) {
                style.id = 'custom-toolbar-style';
                iframe.contentDocument.head.appendChild(style);
            }
        }
    }
    // Run after iframe loads
    setTimeout(styleCanvasToolbar, 1000);
    setInterval(styleCanvasToolbar, 2000);
    </script>
    """, unsafe_allow_html=True)

# Initialize OpenAI Client (OpenRouter)
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    st.error("Please set OPENROUTER_API_KEY in .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)

# System Prompt - Socratic Tutoring Mode
SYSTEM_PROMPT = """
You are "Rocky the Bull", the energetic and helpful mascot of the University of South Florida (USF).
You are a SOCRATIC TUTOR who helps students learn by asking guiding questions, not by giving complete answers immediately.

Your Teaching Philosophy:
1. **Guide, Don't Solve**: Ask questions that lead students to discover the solution themselves
2. **Break It Down**: Help students identify one step at a time
3. **Encourage Thinking**: "What do you think?", "Why might that be?", "What's the first step?"
4. **Provide Hints**: If stuck, give small hints rather than full solutions
5. **Build Confidence**: Celebrate progress and encourage effort

Your Approach:
- When a student asks a question or shows a problem:
  1. First, ask what specifically they need help with (if not clear)
  2. Ask what they've tried or what they think the first step might be
  3. Guide them through the problem with questions
  4. Only provide the full solution if they're completely stuck after multiple attempts

Examples of Socratic Questions:
- "What information do we have in this problem?"
- "What are we trying to find?"
- "What formula or concept might apply here?"
- "Can you break this down into smaller steps?"
- "What would happen if we tried [approach]?"

Output Format:
1. **Visual Response**: 
   - Ask guiding questions to help the student think through the problem
   - Use step-by-step questioning approach
   - For math: Use LaTeX when showing hints or partial work
   - **STRICTLY** use double dollar signs for block equations: $$ \int x dx = \frac{x^2}{2} + C $$
   - **STRICTLY** use single dollar signs for inline math: $ x^2 $
   - **DO NOT** use brackets like \[ \] or \( \) for math.
2. **Audio Script**: 
   - A confident, encouraging version of your guiding questions
   - **CONCISE**. Ask the question naturally.
   - Do NOT read out LaTeX code.
   - Speak as if you are talking to the student directly.
Separator: Use '---AUDIO---' to separate the visual response from the audio script.
Avoid saying anything racist or political or bad words
if keyword aegis is used anywhere you can jump to end and give the answer
"""

async def generate_audio(text, output_file="response.mp3"):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural", pitch="-10Hz", rate="+20%")
    await communicate.save(output_file)

def get_response(messages):
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def transcribe_audio(audio_file):
    try:
        # Use Google's free SpeechRecognition
        recognizer = sr.Recognizer()
        
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Transcribe using Google Speech Recognition
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up temp file
        os.unlink(tmp_path)
        return text
    except sr.UnknownValueError:
        return "Error: Could not understand audio"
    except sr.RequestError as e:
        return f"Error: Could not request results; {e}"
    except Exception as e:
        return f"Error transcribing audio: {e}"

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def read_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        return f"Error reading TXT: {e}"

def process_response(user_text, canvas_result):
    # 1. Capture Canvas Image
    img_str = None
    if canvas_result.image_data is not None:
        # Convert numpy array to image
        img_data = canvas_result.image_data.astype("uint8")
        image = Image.fromarray(img_data)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

    # 2. Prepare Messages with conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (excluding audio field)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add current user message with image and file content
    user_message_text = user_text
    
    # Include uploaded file content if present
    if "uploaded_file_content" in st.session_state:
        user_message_text = f"[Uploaded File Content]:\n{st.session_state.uploaded_file_content}\n\n[User Question]: {user_text}"
    
    content_payload = [{"type": "text", "text": user_message_text}]
    if img_str:
        content_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_str}"
            }
        })
    messages.append({"role": "user", "content": content_payload})
    
    # 3. Get Response
    with st.spinner("Rocky is studying your work..."):
        response_text = get_response(messages)
    
    # 4. Parse Response
    if "---AUDIO---" in response_text:
        visual_part, audio_part = response_text.split("---AUDIO---")
        visual_part = visual_part.strip()
        audio_part = audio_part.strip()
    else:
        visual_part = response_text
        audio_part = response_text

    # 5. Update Session State
    st.session_state.messages.append({"role": "assistant", "content": visual_part, "audio": audio_part})
    st.session_state.history_index = len(st.session_state.messages) - 1  # Jump to latest message
    st.rerun()

# Header
col_header_1, col_header_2 = st.columns([1, 10])
with col_header_1:
    if os.path.exists("Rocky.png"):
        st.image("Rocky.png", width=100)
    else:
        st.write("🤘")
with col_header_2:
    st.title("Green & Gold Guru 🤘")
    st.markdown("*Draw your problem, ask for help, and let Rocky guide you!*")

# Main Layout
col_left, col_right = st.columns([1, 2])

with col_left:
    # Clear Chat Button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_spoken = ""
        if "uploaded_file_content" in st.session_state:
            del st.session_state.uploaded_file_content
        st.rerun()
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        if "uploaded_file_content" not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
            with st.spinner("Reading file..."):
                if uploaded_file.type == "application/pdf":
                    file_content = read_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    file_content = read_docx(uploaded_file)
                else:
                    file_content = read_txt(uploaded_file)
                
                st.session_state.uploaded_file_content = file_content
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name}")

    # Audio Input
    audio_value = st.audio_input("Record your question")
    
    # Submit Voice Button (directly below audio input)
    if st.button("Submit Voice"):
        if audio_value:
            # Get canvas from session state
            current_canvas = st.session_state.get("canvas_result", None)
            with st.spinner("Listening..."):
                user_text = transcribe_audio(audio_value)
                st.session_state.messages.append({"role": "user", "content": f"🎤 {user_text}"})
                process_response(user_text, current_canvas)
        else:
            st.warning("Please record something first!")
    
    # Chat History with Navigation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "history_index" not in st.session_state:
        st.session_state.history_index = len(st.session_state.messages) - 1
    
    # Ensure index is valid
    if st.session_state.messages:
        st.session_state.history_index = max(0, min(st.session_state.history_index, len(st.session_state.messages) - 1))
        
        # Navigation buttons
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        with col_nav1:
            if st.button("◀ Previous") and st.session_state.history_index > 0:
                st.session_state.history_index -= 1
                st.rerun()
        with col_nav2:
            st.markdown(f"**Message {st.session_state.history_index + 1} of {len(st.session_state.messages)}**")
        with col_nav3:
            if st.button("Next ▶") and st.session_state.history_index < len(st.session_state.messages) - 1:
                st.session_state.history_index += 1
                st.rerun()
        
        # Display current message
        current_message = st.session_state.messages[st.session_state.history_index]
        avatar = "Rocky.png" if current_message["role"] == "assistant" and os.path.exists("Rocky.png") else ("🤘" if current_message["role"] == "assistant" else None)
        with st.chat_message(current_message["role"], avatar=avatar):
            st.markdown(current_message["content"])
    else:
        st.info("No messages yet. Start a conversation!")

with col_right:
    # Callback to force update
    def update_tool_state():
        pass

    # Drawing Tools
    col_tools_1, col_tools_2, col_tools_3 = st.columns([1, 1, 1])
    with col_tools_1:
        # Use a key for the color picker to ensure state tracking
        picked_color = st.color_picker("Pen Color", "#000000", key="pen_color_picker", on_change=update_tool_state)
    with col_tools_2:
        stroke_width = st.slider("Pen Width", 1, 25, 3, key="stroke_width_slider", on_change=update_tool_state)
    with col_tools_3:
        eraser_mode = st.checkbox("Eraser", key="eraser_mode_toggle", on_change=update_tool_state)
    
    if eraser_mode:
        stroke_color = "#FFFFFF"
        # Optional: Increase width for eraser convenience
        # stroke_width = 20 
    else:
        stroke_color = picked_color 
    
    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=500,
        width=800,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Store canvas in session state for access from left column
    st.session_state.canvas_result = canvas_result
    
    if st.button("Rocky! Help me!"):
        user_text = ""
        
        # 1. Transcribe Audio if present
        if audio_value:
            with st.spinner("Listening..."):
                user_text = transcribe_audio(audio_value)
                st.session_state.messages.append({"role": "user", "content": f"🎤 {user_text}"})
        else:
            user_text = "Please help me with what I drew on the whiteboard."
            st.session_state.messages.append({"role": "user", "content": "🎨 [Drawing Submitted]"})

        process_response(user_text, canvas_result)

    # Chat Input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_response(prompt, canvas_result)

# Play Audio if new response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_message = st.session_state.messages[-1]
    last_visual = last_message["content"]
    last_audio_text = last_message.get("audio", last_visual) # Fallback if key missing
    
    # Simple check to prevent loop: store last spoken text in session state
    if "last_spoken" not in st.session_state or st.session_state.last_spoken != last_audio_text:
        asyncio.run(generate_audio(last_audio_text))
        st.session_state.last_spoken = last_audio_text
    
    if os.path.exists("response.mp3"):
        st.audio("response.mp3", autoplay=True)
