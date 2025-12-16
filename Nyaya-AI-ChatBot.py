import streamlit as st
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent))
try:
    from NeuroSymbolic_Pipeline import NeurosymbolicLegalRetriever, LegalChatbot
    import logging
    logging.basicConfig(level=logging.WARNING)
except ImportError:
    st.error("Could not import chatbot modules.")
    st.stop()

st.set_page_config(
    page_title="Nyaya AI",
    layout="wide",
    initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        background-attachment: fixed;
    }    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(99, 102, 241, 0.08), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(99, 102, 241, 0.08), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(99, 102, 241, 0.05), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(99, 102, 241, 0.05), transparent),
            radial-gradient(3px 3px at 10% 80%, rgba(139, 92, 246, 0.06), transparent);
        background-size: 200% 200%;
        background-position: 0% 0%;
        animation: drift 20s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes drift {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }    
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            to bottom,
            transparent 50%,
            rgba(99, 102, 241, 0.02) 50%
        );
        background-size: 100% 4px;
        pointer-events: none;
        z-index: 1;
        opacity: 0.3;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }    
    .stChatMessage {
        background: rgba(20, 20, 20, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .stChatMessage::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(99, 102, 241, 0.1),
            transparent
        );
        transition: left 0.6s ease;
    }
    .stChatMessage:hover::before {
        left: 100%;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }
    [data-testid="stChatMessageContent"] {
        background: transparent;
    }    
    .stTextInput > div[data-baseweb="input"],
    div[data-baseweb="input"] {
        border: none !important;
        border-radius: 12px !important;
    } 
    .stTextInput > div[data-baseweb="input"] > div,
    div[data-baseweb="input"] > div {
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
    }
    .stTextInput > div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {
        border: 2px solid rgba(99, 102, 241, 0.8) !important;
        border-radius: 12px !important;
    }    
    .stTextInput > div > div > input,
    .stTextInput input,
    input[type="text"],
    div[data-baseweb="input"] input {
        border-radius: 12px !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: none !important;
        background: rgba(25, 25, 25, 0.8) !important;
        color: #FAFAFA !important;
        outline: none !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextInput input:focus,
    input[type="text"]:focus,
    div[data-baseweb="input"] input:focus {
        border-color: rgba(99, 102, 241, 0.8) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(99, 102, 241, 0.1) !important;
        background: rgba(30, 30, 30, 1) !important;
        transform: translateY(-1px);
    }    
    .stTextInput > div > div > input:invalid,
    .stTextInput > div > div > input:user-invalid,
    .stTextInput > div > div > input[aria-invalid="true"],
    .stTextInput input:invalid,
    .stTextInput input:user-invalid,
    input[type="text"]:invalid,
    input[type="text"]:user-invalid,
    div[data-baseweb="input"] input:invalid,
    div[data-baseweb="input"] input:user-invalid {
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div > input:focus:invalid,
    .stTextInput > div > div > input:focus:user-invalid,
    .stTextInput > div > div > input:focus[aria-invalid="true"],
    .stTextInput input:focus:invalid,
    input[type="text"]:focus:invalid {
        border: 2px solid rgba(99, 102, 241, 0.8) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(99, 102, 241, 0.1) !important;
    }
    .stTextInput input::placeholder {
        color: #606060 !important;
        transition: color 0.3s ease;
    }
    .stTextInput input:focus::placeholder {
        color: #808080 !important;
    }    
    .stTextInput button,
    .stTextInput [role="button"],
    .stTextInput [data-testid="baseButton-secondary"],
    .stTextInput [kind="secondary"],
    div[data-baseweb="input"] button,
    div[data-baseweb="input"] [role="button"],
    div[data-baseweb="input"] [data-testid="baseButton-secondary"],
    div[data-baseweb="input"] [kind="secondary"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }
    .stTextInput svg,
    .stTextInput path,
    div[data-baseweb="input"] svg,
    div[data-baseweb="input"] path {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }    
    .stTextInput > div > div > div:last-child,
    .stTextInput > div[data-baseweb="input"] > div:last-child,
    div[data-baseweb="input"] > div:last-child:not(input) {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }    
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 70px !important;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(25, 25, 25, 0.8) !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        min-height: 56px !important;
        width: 100% !important;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: rgba(99, 102, 241, 0.8) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(99, 102, 241, 0.1) !important;
        background: rgba(30, 30, 30, 1) !important;
    }    
    [data-testid="stChatInput"] textarea,
    .stChatInput textarea,
    [data-testid="stChatInput"] textarea:active,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:hover,
    [data-testid="stChatInput"] textarea[aria-invalid="true"],
    [data-testid="stChatInput"] textarea[aria-invalid="false"],
    [data-testid="stChatInput"] textarea:invalid,
    [data-testid="stChatInput"] textarea:user-invalid {
        background: transparent !important;
        color: #FAFAFA !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        outline: none !important;
        box-shadow: none !important;
        resize: none !important;
        flex: 1 !important;
        min-height: 52px !important;
        max-height: 52px !important;
        line-height: 1.5 !important;
        padding-top: 16px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #606060 !important;
        transition: color 0.3s ease;
    }
    [data-testid="stChatInput"] textarea:focus::placeholder {
        color: #808080 !important;
    }
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] [data-testid="baseButton-header"],
    [data-testid="stChatInput"] [data-testid="baseButton-headerNoPadding"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 12px 0 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        min-height: 40px !important;
        height: 40px !important;
        width: 40px !important;
        flex-shrink: 0 !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        align-self: center !important;
    }
    [data-testid="stChatInput"] > div > div:last-child {
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
    }
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInputSubmitButton"]:hover {
        background: rgba(99, 102, 241, 0.1) !important;
    }
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {
        color: rgba(99, 102, 241, 0.8) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"] button:hover svg,
    [data-testid="stChatInputSubmitButton"]:hover svg {
        color: rgba(99, 102, 241, 1) !important;
    }    
    .stTextInput > label > div[data-testid="stMarkdownContainer"] > p {
        display: none !important;
    }
    .stTextInput small {
        display: none !important;
    }
    .stTextInput > div[data-testid="InputInstructions"] {
        display: none !important;
    }
    .stSlider {
        padding: 10px 0 !important;
    }    
    .stSlider *,
    .stSlider > *,
    .stSlider > * > *,
    .stSlider > * > * > *,
    .stSlider div,
    .stSlider [data-testid="stTickBar"],
    .stSlider [data-testid="stThumbValue"] {
        background: transparent !important;
        background-color: transparent !important;
    }    
    [data-baseweb="slider"],
    [data-baseweb="slider"] > div,
    [data-baseweb="slider"] > div > div {
        background: transparent !important;
        background-color: transparent !important;
    }    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5) !important;
        cursor: grab !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"]:active {
        cursor: grabbing !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.7) !important;
    }    
    .stSlider [data-baseweb="slider"] > div:first-child > div:first-child {
        background: rgba(99, 102, 241, 0.2) !important;
    }    
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    .stButton button:hover::before {
        left: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
    }
    .stButton button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
    }    
    h1 {
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -1px;
        animation: titleGlow 3s ease-in-out infinite;
        text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
    }
    @keyframes titleGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    h2, h3 {
        color: #e0e7ff;
        font-weight: 600;
    }
    p, li, label {
        color: #cbd5e1;
    }
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.1) 100%);
        color: #e0e7ff;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 12px 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .streamlit-expanderHeader::after {
        content: '';
        position: absolute;
        right: -50px;
        top: -50px;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.2), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .streamlit-expanderHeader:hover::after {
        opacity: 1;
    }
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.2) 100%);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateX(4px);
    }    
    code {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'Fira Code', monospace;
        border: 1px solid rgba(16, 185, 129, 0.2);
        transition: all 0.3s ease;
        position: relative;
    }
    code:hover {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }    
    .stAlert {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        color: #cbd5e1;
        backdrop-filter: blur(10px);
    }    
    [data-testid="stMetricValue"] {
        color: #10b981;
        font-weight: 600;
    }    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 30px 0;
        position: relative;
    }
    hr::after {
        content: '';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 6px;
        height: 6px;
        background: #6366f1;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.8);
    }    
    .stSpinner > div {
        border-top-color: #6366f1 !important;
        border-right-color: rgba(99, 102, 241, 0.3) !important;
        border-bottom-color: rgba(99, 102, 241, 0.3) !important;
        border-left-color: rgba(99, 102, 241, 0.3) !important;
    }    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .stSpinner {
        animation: pulse 1.5s ease-in-out infinite;
    }    
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        color: #10b981;
        position: relative;
        overflow: hidden;
    }
    .stSuccess::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: #10b981;
        animation: slideDown 2s ease infinite;
    }
    @keyframes slideDown {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(300%); }
    }
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        color: #f59e0b;
        position: relative;
        overflow: hidden;
    }
    .stWarning::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: #f59e0b;
        animation: slideDown 2s ease infinite;
    }
    .case-card {
        background: rgba(30, 30, 30, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .case-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(180deg, #6366f1, #8b5cf6);
        border-radius: 12px 0 0 12px;
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    .case-card:hover::before {
        transform: scaleY(1);
    }
    .case-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(8px);
        background: rgba(35, 35, 35, 0.9);
    }
    .example-section {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .example-section:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
        transform: translateY(-4px);
        background: rgba(30, 30, 30, 0.7);
    }
    .example-section ul li {
        transition: color 0.2s ease, transform 0.2s ease;
        cursor: pointer;
        padding: 4px 0;
    }
    .example-section ul li:hover {
        color: #a5b4fc;
        transform: translateX(8px);
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.9em;
        padding: 20px;
        background: linear-gradient(180deg, transparent, rgba(99, 102, 241, 0.05));
        border-radius: 12px;
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .animate-in {
        animation: fadeInUp 0.6s ease;
    }
</style>
""", unsafe_allow_html=True)
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.retriever = None
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_details' not in st.session_state:
    st.session_state.show_details = True
@st.cache_resource

def initialize_chatbot():
    try:
        with st.spinner():
            retriever = NeurosymbolicLegalRetriever(
                gnn_data_dir="gnn_data",
                processed_dir="dataset_processed",
                rules_dir="official_documents")
            chatbot = LegalChatbot(
                retriever=retriever,
                llm_model="deepseek-r1:7b")
            return retriever, chatbot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None, None
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; font-size: 3.5em; margin-bottom: 10px;'>Nyaya AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1em; font-weight: 500; letter-spacing: 1px; margin-bottom: 40px;'>A Legal Neuro-Symbolic AI System</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### System Configuration")   
    top_k = st.slider("Results to retrieve", 3, 10, 5, 1,
                     help="Number of cases to retrieve",
                     key="top_k_slider")   
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")   
    st.markdown("#### System Status")
    if st.session_state.initialized:
        st.success("System Ready")
    else:
        st.warning("Initializing...")

alpha_text = 0.7
alpha_gat = 0.15
alpha_symbolic = 0.15
stage1_k = 100
show_thinking = True
show_scores = False
show_citations = False
chat_container = st.container()

if not st.session_state.initialized:
    retriever, chatbot = initialize_chatbot()
    if retriever and chatbot:
        st.session_state.retriever = retriever
        st.session_state.chatbot = chatbot
        st.session_state.initialized = True
        st.rerun()
    else:
        st.error("Failed to initialize.")
        st.stop()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])           
            if message["role"] == "assistant" and "thinking" in message and message["thinking"] and show_thinking:
                with st.expander("Reasoning Process"):
                    st.markdown(f"```\n{message['thinking']}\n```")           
            if message["role"] == "assistant" and "cases" in message and message["cases"]:
                with st.expander(f"Retrieved Cases ({len(message['cases'])})"):
                    for i, case in enumerate(message['cases'], 1):
                        doc_type = case.get('doc_type', 'case')
                        st.markdown(f"<div class='case-card'><strong>{i}. {case['title'][:80]}</strong>", unsafe_allow_html=True)                       
                        col1, col2 = st.columns(2)
                        with col1:
                            if doc_type == 'case':
                                st.text(f"Court: {case['court'].replace('_', ' ').title()}")
                                st.text(f"Date: {case['date']}")
                            elif doc_type == 'pdf_document':
                                st.text(f"Source: {case['metadata'].get('source_pdf', 'Unknown')}")
                        with col2:
                            if show_scores:
                                st.text(f"Overall: {case['score']:.3f}")                     
                        st.markdown("</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})   
    with st.chat_message("user"):
        st.markdown(prompt)   
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching cases and analyzing..."):
            try:
                result = st.session_state.chatbot.chat(
                    query=prompt, top_k=top_k,
                    stage1_k=stage1_k, alpha_text=alpha_text,
                    alpha_gat=alpha_gat, alpha_symbolic=alpha_symbolic,
                    return_thinking=True)
                message_placeholder.markdown(result['response'])
                st.session_state.messages.append({
                    "role": "assistant","content": result['response'],
                    "thinking": result.get('thinking', ''),
                    "cases": result.get('retrieved_cases', [])})
                if result.get('thinking') and show_thinking:
                    with st.expander("Reasoning Process"):
                        st.markdown(f"```\n{result['thinking']}\n```")               
                if result.get('retrieved_cases'):
                    with st.expander(f"Retrieved Cases ({len(result['retrieved_cases'])})"):
                        for i, case in enumerate(result['retrieved_cases'], 1):
                            doc_type = case.get('doc_type', 'case')
                            st.markdown(f"<div class='case-card'><strong>{i}. {case['title'][:80]}</strong>", unsafe_allow_html=True)                           
                            col1, col2 = st.columns(2)
                            with col1:
                                if doc_type == 'case':
                                    st.text(f"Court: {case['court'].replace('_', ' ').title()}")
                                    st.text(f"Date: {case['date']}")
                                elif doc_type == 'pdf_document':
                                    st.text(f"Source: {case['metadata'].get('source_pdf', 'Unknown')}")
                            with col2:
                                if show_scores:
                                    st.text(f"Overall: {case['score']:.3f}")
                            st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating response: {e}")

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<div class='footer-text'>"
        "Nyaya AI | Powered by DeepSeek-R1 | A Legal Neuro-Symbolic AI System"
        "</div>",
        unsafe_allow_html=True)
if not st.session_state.messages:
    st.markdown("<h3 style='text-align: center; margin-top: 40px; color: #e0e7ff;'>Example Queries</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='example-section'>
        <h4 style='color: #818cf8; margin-bottom: 15px;'>Trademark & IP</h4>
        <ul style='line-height: 1.8;'>
        <li>Find cases about trademark infringement in pharmaceuticals.</li>
        <li>Give copyright law cases.</li>
        </ul>
        <h4 style='color: #818cf8; margin-top: 20px; margin-bottom: 15px;'>Criminal Law</h4>
        <ul style='line-height: 1.8;'>
        <li>What punishment is there for murder under IPC?</li>
        <li>Show cases about Section 420 IPC.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='example-section'>
        <h4 style='color: #818cf8; margin-bottom: 15px;'>Constitutional Law</h4>
        <ul style='line-height: 1.8;'>
        <li>Find cases on Article 21.</li>
        <li>What are fundamental rights under Constitution?</li>
        </ul>
        <h4 style='color: #818cf8; margin-top: 20px; margin-bottom: 15px;'>Civil Law</h4>
        <ul style='line-height: 1.8;'>
        <li>Show cases about property disputes.</li>
        <li>Find contract law cases.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)