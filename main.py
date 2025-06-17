import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load your environment variables (GOOGLE_API_KEY must be set)
load_dotenv()

# Initialize Gemini Pro model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Streamlit UI
st.set_page_config(page_title="Gemini PDF QA", page_icon="📄")
st.title("📄 Ask Questions from Your PDF (Powered by Gemini)")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# PDF upload
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Handle PDF processing
if pdf_file is not None:
    # Check if this is a new PDF or the same one
    pdf_name = pdf_file.name
    if "current_pdf" not in st.session_state or st.session_state.current_pdf != pdf_name:
        st.session_state.current_pdf = pdf_name
        st.session_state.messages = []  # Clear chat history for new PDF
        
        st.success("✅ PDF uploaded successfully.")
        
        # Extract all text
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    full_text += content + "\n"
        
        if not full_text.strip():
            st.warning("⚠️ Could not extract any text from the PDF.")
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_text = ""
        else:
            st.session_state.pdf_text = full_text
            st.session_state.pdf_uploaded = True
            st.success(f"✅ Text extracted successfully! ({len(full_text)} characters)")
    else:
        st.success("✅ PDF already loaded.")

# Display chat history
if st.session_state.messages:
    st.subheader("💬 Chat History")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**Question {i//2 + 1}:** {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer {i//2 + 1}:** {message['content']}")

# Chat input
query = st.chat_input("Ask a question based on the uploaded PDF")

# Handle query processing
if query:
    if st.session_state.pdf_uploaded and st.session_state.pdf_text:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**Question {len(st.session_state.messages)//2 + 1}:** {query}")
        
        # Show thinking message
        with st.chat_message("assistant"):
            with st.spinner("🤖 Generating response using Gemini..."):
                # Construct prompt for Gemini
                prompt = f"""
You are a helpful assistant. Based on the following PDF content, answer the user's question accurately and concisely. 
If the question is general and not from the PDF, say "Related info is not mentioned in the PDF" but answer if you know the answer to it.

PDF Content:
\"\"\"
{st.session_state.pdf_text[:15000]}  # Limit input to avoid token overflow
\"\"\"

Previous conversation context:
{chr(10).join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.messages[-6:] if msg['role'] == 'user'])}

Current Question: {query}

Answer:
"""
                
                try:
                    # Invoke Gemini
                    response = llm.invoke(prompt)
                    answer = response.content
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Display the answer
                    st.markdown(f"**Answer {len(st.session_state.messages)//2}:** {answer}")
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
        
        # Force a rerun to update the chat display
        st.rerun()
        
    else:
        st.warning("⚠️ Please upload a PDF before asking a question.")

# Sidebar with additional options
with st.sidebar:
    st.header("📋 Chat Options")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()
    
    if st.session_state.pdf_uploaded:
        st.success(f"📄 PDF loaded: {st.session_state.current_pdf}")
        st.info(f"📊 Text length: {len(st.session_state.pdf_text)} characters")
        
        # Show number of messages
        num_questions = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        st.info(f"💬 Questions asked: {num_questions}")
    
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("• Upload a PDF first")
    st.markdown("• Ask specific questions about the content")
    st.markdown("• Use 'Clear Chat History' to start fresh")
    st.markdown("• Previous context is maintained for follow-up questions")