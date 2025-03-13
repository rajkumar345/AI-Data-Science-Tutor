import os  
import streamlit as st  
from dotenv import load_dotenv  
from PIL import Image  
import pytesseract  
from langchain_google_genai import GoogleGenerativeAI  
from langchain.memory import ConversationBufferMemory  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  

# Load environment variables  
load_dotenv()

# Get the API key  
api_key = os.getenv("GOOGLE_API_KEY")  
if not api_key:  
    st.error("🚨 GOOGLE_API_KEY is missing! Please add it to `.env` file.")  
    st.stop()

# Initialize the model  
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)  

# Initialize memory  
if "memory" not in st.session_state:  
    st.session_state.memory = ConversationBufferMemory(return_messages=True)  
memory = st.session_state.memory  

# Streamlit UI  
st.title("🧠 AI Data Science Tutor")  
user_input = st.text_input("❓Ask your Data Science question:")  
uploaded_file = st.file_uploader("📤Upload an image with your question:", type=["jpg", "jpeg", "png"])  

# If image is uploaded, extract text using OCR  
if uploaded_file is not None:  
    image = Image.open(uploaded_file)  
    st.image(image, caption="Uploaded Image", use_container_width=True)  
    try:
        extracted_text = pytesseract.image_to_string(image).strip()  
        if extracted_text:
            user_input = extracted_text
            st.success("✅ Text extracted from image successfully.")
        else:
            st.warning("⚠️ No readable text found in the image.")
    except pytesseract.TesseractNotFoundError:
        st.error("❌ Tesseract-OCR is not installed or not added to PATH.")  

# Format conversation history  
history = "\n".join([  
    f"User: {msg.content}" if msg.type == "human" else f"Tutor: {msg.content}"  
    for msg in memory.chat_memory.messages  
])  

# Define AI prompt template  
prompt = PromptTemplate(  
    input_variables=["user_input", "history"],  
    template="""  
    You are an AI-powered Data Science tutor.  
    Provide a detailed and well-structured response to the user's question.  
    If relevant, include examples, definitions, and practical applications.  
    If the user question is unclear, ask follow-up questions for clarification.  
    **Conversation History:**  
    {history}  
    **User Question:**  
    {user_input}  
    **AI Response:**  
    """  
)  

# Create LLM chain  
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)  

if st.button("Submit"):  
    if user_input:  
        try:  
            # Generate response  
            response = chain.invoke({  
                "user_input": user_input,  
                "history": history  
            })  

            response_text = response.get("text") or response.get("output") or "Sorry, I couldn't process that."  

            # Display response  
            st.write("**Tutor:**")  
            st.write(response_text)  

        except Exception as e:  
            st.error(f"❌ Error: {e}")  
    else:  
        st.warning("⚠️ Please enter a question or upload an image.")  

# Display conversation history in the sidebar  
st.sidebar.subheader("🕒Conversation History")  
for msg in memory.chat_memory.messages:  
    if msg.type == "human":  
        st.sidebar.write(f"**User:** {msg.content}")  
    else:  
        st.sidebar.write(f"**Tutor:** {msg.content}")  
