import streamlit as st
import uuid
from PIL import Image

# Function to clear the input
def reset():
    st.session_state.keyword = ""
    st.session_state.upload_key = str(uuid.uuid4())  # Assign a new unique key
    st.session_state.flag = False  # Reset the flag
    st.session_state.stylized_image = None  # Reset the stylized image
    st.session_state.uploaded_image = None  # Reset the uploaded image
    st.session_state.original_prompt = ""
    st.session_state.modified_prompt = ""  # Reset the prompts

# Initialize session state
if "upload_key" not in st.session_state:
    st.session_state.upload_key = str(uuid.uuid4())
if "keyword" not in st.session_state:
    st.session_state.keyword = ""
if "model" not in st.session_state:
    st.session_state.model = None # Placeholder for model assignment
if "stylized_image" not in st.session_state:
    st.session_state.stylized_image = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "flag" not in st.session_state:
    st.session_state.flag = False
if "original_prompt" not in st.session_state:
    st.session_state.original_prompt = ""
if "modified_prompt" not in st.session_state:
    st.session_state.modified_prompt = ""

# Page setup
st.set_page_config(page_title="Image Vibe Transformer", layout="centered")
st.title("🖼️ Image to Different Vibe")

# Upload image
st.session_state.uploaded_image = st.file_uploader("Upload a file", key=st.session_state.upload_key)

# Keyword input
st.text_input("Optional keyword (e.g. 'vaporwave', 'cyberpunk')", key="keyword")

# Checkbox for showing prompt details
show_prompts = st.checkbox("Show Prompt Details", value=False)

def generate():
    image = Image.open(st.session_state.uploaded_image)
    
    # Simulate image transformation
    if not st.session_state.flag:
        st.session_state.flag = True
        st.session_state.original_prompt = "A serene angelic woman standing in a field of white flowers..."
        st.session_state.modified_prompt = f"A {st.session_state.keyword or '[vibe]'} warrior in flames with dark wings..."
        with st.spinner("Generating stylized image..."):
            st.session_state.stylized_image = image  # Replace with your real model output
        return

    # Prompt display
    if show_prompts:
        st.subheader("🔍 Prompt Details")
        st.code(st.session_state.original_prompt)
        st.code(st.session_state.modified_prompt)

    st.image(st.session_state.stylized_image, caption="Stylized Output", use_container_width=True)
    st.success("✅ Image generated!")

    # "Do it again" resets everything
    st.button("🔁 Do it again", on_click=reset)

# Generate button
if not st.session_state.flag and st.session_state.uploaded_image:
    st.button("Generate Stylized Image", on_click=generate)
elif st.session_state.flag:
    generate()

