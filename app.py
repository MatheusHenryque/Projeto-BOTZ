import streamlit as st
from streamlit_chat import message
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.vector_stores import SimpleVectorStore  # Importação corrigida
from llama_index.core import Settings
from dotenv import load_dotenv
import os
import time

load_dotenv()

@st.cache_resource
def init_chat_engine():
    # Configuração do embedding
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    llm = Groq(model="llama3-70b-8192", api_key=st.secrets["API_KEY"])


    Settings.embed_model = embed_model
    Settings.llm = llm

    # Memória de conversa
    memory = ChatSummaryMemoryBuffer(llm=llm, token_limit=512)

    # Carregar documentos
    documents = SimpleDirectoryReader("./documentos").load_data()
    
    # Configurar vector store (versão atualizada)
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Criar índice
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    return index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "Você é um assistente especializado em IBM z17. "
            "Responda com precisão sobre arquitetura, IA embarcada, segurança, "
            "z/OS, CICS, IMS, JCL e casos de uso de mainframes modernos."
        )
    )

# Inicializar o motor de chat
chat_engine = init_chat_engine()

st.set_page_config(
    page_title="BOTZ - Assistente Z17",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_PALETTE = {
    "primary": "#0a192f",
    "secondary": "#64ffda",
    "background": "#0f2027",
    "text": "#ccd6f6",
    "accent": "#1e90ff",
    "dark": "#020c1b",
    "light": "#e6f1ff"
}

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        :root {{
            --primary: {COLOR_PALETTE["primary"]};
            --secondary: {COLOR_PALETTE["secondary"]};
            --background: {COLOR_PALETTE["background"]};
            --text: {COLOR_PALETTE["text"]};
            --accent: {COLOR_PALETTE["accent"]};
            --dark: {COLOR_PALETTE["dark"]};
            --light: {COLOR_PALETTE["light"]};
        }}
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        
        .main {{
            background: linear-gradient(135deg, var(--dark), var(--primary));
            color: var(--text);
        }}
        
        /* Sidebar moderna */
        .sidebar .sidebar-content {{
            background: var(--primary);
            border-right: 1px solid rgba(100, 255, 218, 0.1);
        }}
        
        /* Container do chat moderno */
        .chat-container {{
            display: flex;
            flex-direction: column;
            height: calc(100vh - 220px);
            background: rgba(10, 25, 47, 0.3);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 255, 218, 0.2);
            padding: 1rem;
            overflow-y: auto;
            margin-bottom: 1rem;
        }}
        
        /* Área de input moderna */
        .input-container {{
            background: rgba(10, 25, 47, 0.7);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(100, 255, 218, 0.2);
        }}
        
        /* Mensagens */
        .user-message {{
            background: rgba(30, 144, 255, 0.15) !important;
            border-left: 3px solid var(--accent) !important;
            margin-left: auto;
            margin-right: 0;
            max-width: 80%;
        }}
        
        .bot-message {{
            background: rgba(10, 25, 47, 0.5) !important;
            border-left: 3px solid var(--secondary) !important;
            margin-right: auto;
            margin-left: 0;
            max-width: 80%;
        }}
        
        /* Botões modernos */
        .stButton>button {{
            background-color: var(--secondary) !important;
            color: var(--primary) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(100, 255, 218, 0.3) !important;
        }}
        
        /* Cards modernos */
        .feature-card {{
            background: rgba(10, 25, 47, 0.5) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            border: 1px solid rgba(100, 255, 218, 0.2) !important;
            transition: all 0.3s ease !important;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px) !important;
            box-shadow: 0 8px 16px rgba(100, 255, 218, 0.2) !important;
            border: 1px solid var(--secondary) !important;
        }}
        
        /* Textarea moderno */
        .stTextArea textarea {{
            background: rgba(10, 25, 47, 0.8) !important;
            color: var(--text) !important;
            border: 1px solid rgba(100, 255, 218, 0.3) !important;
            border-radius: 8px !important;
            min-height: 60px !important;
        }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: {COLOR_PALETTE['secondary']}; 
            border-bottom: 2px solid {COLOR_PALETTE['secondary']}; 
            padding-bottom: 10px; display: inline-block;">BOTZ</h1>
        <p style="color: {COLOR_PALETTE['text']};">Assistente de Mainframe Z17</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "Navegação",
        ["🏠 Página Inicial", "💬 Chatbot"],
        label_visibility="collapsed",
        key="nav_radio"
    )

    if st.button("Chatbot"):
        st.switch_page(r"pages/page_1.py")
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="color: {COLOR_PALETTE['text']}; padding: 1rem;">
        <h4 style="color: {COLOR_PALETTE['secondary']};">Sobre o BOTZ</h4>
        <p>Assistente virtual especializado em IBM Z17 com base na documentação oficial.</p>
        <div style="margin-top: 1rem; padding: 0.5rem; 
            background: rgba(100, 255, 218, 0.1); 
            border-radius: 8px;">
            <small style="color: {COLOR_PALETTE['secondary']};">Versão 2.1</small><br>
            <small>Powered by Groq & LlamaIndex</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.link_button("Acesse o meu portfólio", "https://matheushenryque.github.io/Portfolio/")

if menu == "🏠 Página Inicial":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f"""
        <h1 style="color: {COLOR_PALETTE['secondary']}; margin-bottom: 1rem;">
            BOTZ <span style="color: {COLOR_PALETTE['text']};">- Assistente Z17</span>
        </h1>
        <div style="font-size: 1.1rem; line-height: 1.6; color: {COLOR_PALETTE['text']};">
            Seu especialista em <span style="color: {COLOR_PALETTE['secondary']};">IBM Z17</span>, 
            com conhecimento técnico avançado sobre arquitetura, operação e otimização de mainframes.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        st.markdown(f"""
        <h3 style="color: {COLOR_PALETTE['secondary']};">🚀 Como usar</h3>
        <ol style="color: {COLOR_PALETTE['text']}; line-height: 2;">
            <li>Acesse a aba <strong>Chatbot</strong> no menu lateral</li>
            <li>Digite suas perguntas técnicas</li>
            <li>Obtenha respostas precisas baseadas na documentação</li>
        </ol>
        """, unsafe_allow_html=True)

        if st.button("Iniciar Conversa →", key="start_button", use_container_width=True):
            menu == "💬 Chatbot"
            st.rerun()

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLOR_PALETTE['dark']}, {COLOR_PALETTE['primary']});
            height: 100%; min-height: 300px; border-radius: 12px; 
            display: flex; justify-content: center; align-items: center;
            border: 1px solid {COLOR_PALETTE['secondary']}; 
            box-shadow: 0 8px 32px rgba(100, 255, 218, 0.1);">
            <h3 style="color: {COLOR_PALETTE['secondary']};">IBM Z17 MAINFRAME</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown(f"""
    <h2 style="text-align: center; margin-bottom: 1.5rem; color: {COLOR_PALETTE['secondary']};">
        ✨ Recursos Técnicos
    </h2>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("📚", "Base de Conhecimento", "Documentação completa do Z17 indexada"),
        ("🤖", "IA Especializada", "Respostas técnicas precisas e contextualizadas"),
        ("⚡", "Performance", "Respostas rápidas com tecnologia Groq")
    ]
    
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2rem; color: {COLOR_PALETTE['secondary']}; 
                    margin-bottom: 0.5rem;">{icon}</div>
                <h3 style="color: {COLOR_PALETTE['secondary']};">{title}</h3>
                <p style="color: {COLOR_PALETTE['text']};">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

elif menu == "💬 Chatbot":
    st.header('🤖 Bem-vindo ao BOTZ', divider=True)

    chat_engine = st.session_state.get('chat_engine')
    if chat_engine is None:
        st.session_state.chat_engine = init_chat_engine()
        chat_engine = st.session_state.chat_engine
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for mensagem in st.session_state.chat_history:
        with st.chat_message(mensagem["type"]):
            st.markdown(mensagem["content"])
    
    input_usuario = st.chat_input("Fale com o BOTZ sobre IBM Z17")
    
    if input_usuario:
        with st.chat_message("human"):
            st.markdown(input_usuario)
        
        st.session_state.chat_history.append({
            "type": "human",
            "content": input_usuario
        })
        
        # Resposta do bot
        with st.chat_message("ai"):
            with st.spinner("Pensando..."):
                try:
                    start_time = time.time()
                    resposta = chat_engine.chat(input_usuario).response
                    processing_time = time.time() - start_time
                    
                    resposta_completa = f"{resposta}\n\n⏱️ {processing_time:.2f}s"
                    st.markdown(resposta_completa)
                    
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": resposta_completa
                    })
                except Exception as e:
                    erro = f"⚠️ Ocorreu um erro: {str(e)}"
                    st.error(erro)
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": erro
                    })


    if len(st.session_state['chat_history']) > 0:
        st.markdown("---")
        with st.expander("📚 Documentação Relacionada", expanded=False):
            st.markdown(f"""
            <div style="color: {COLOR_PALETTE['text']};">
                <h4 style="color: {COLOR_PALETTE['secondary']};">Recursos IBM</h4>
                <ul>
                    <li><strong>IBM Z17 Technical Guide</strong></li>
                    <li><strong>Redbook Z17</strong></li>
                    <li><strong>GDPS Configuration</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)