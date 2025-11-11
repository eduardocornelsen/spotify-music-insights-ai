import streamlit as st
import pandas as pd
import plotly.express as px
import io
import sys
import re
from pathlib import Path
from scipy import stats
from langchain.tools import tool
import time
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from collections import Counter

# --- Importações do LangChain (Tool Calling Agent) ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_agent
    from langchain_core.prompts import ChatPromptTemplate
    
    IA_DISPONIVEL = True
except ImportError:
    IA_DISPONIVEL = False
except Exception:
    IA_DISPONIVEL = False


# Read System Prompt from file
system_prompt = Path("./prompts/system.txt").read_text()


# --- PAGE CONFIG ---

st.set_page_config(
    page_title="MusicInsights AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

#--- FUNCTION FOR TOC ---
def set_viz(viz_name: str):
    """
    Callback function to update the visualization picker.
    """
    st.session_state.viz_picker = viz_name
    

# --- UTILITY FUNCTIONS FOR ANIMATIONS ---

def show_loading_animation(message="Loading...", duration=0.5):
    """Custom loading animation for operations"""
    with st.spinner(f'🎵 {message}'):
        time.sleep(duration)  # Simulated delay
        # Actual computation will replace the sleep
        
def animated_success(message, duration=2):
    """Animated success message that auto-disappears"""
    placeholder = st.empty()
    placeholder.success(f"✅ {message}")
    time.sleep(duration)
    placeholder.empty()

def animated_info(message, duration=3):
    """Animated info message"""
    placeholder = st.empty()
    placeholder.info(f"ℹ️ {message}")
    time.sleep(duration)
    placeholder.empty()

def animated_warning(message, duration=3):
    """Animated warning message"""
    placeholder = st.empty()
    placeholder.warning(f"⚠️ {message}")
    time.sleep(duration)
    placeholder.empty()

def progress_bar_loading(message="Processing...", steps=100):
    """Show progress bar for multi-step operations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(steps):
        progress_bar.progress((i + 1) / steps)
        status_text.text(f'{message} {i+1}%')
        time.sleep(0.01)  # Adjust speed as needed
    
    progress_bar.empty()
    status_text.empty()



# --- CSS AND STYLING FUNCTIONS ---
def apply_all_effects():
    """Apply all visual effects to the dashboard"""
    st.markdown("""
    <style>
        /* ============ SMOOTH SCROLLING ============ */
    
        /* Enable smooth scrolling globally */
        html, body, [data-testid="stAppViewContainer"] {
            scroll-behavior: smooth;
        }
        
        /* Main Content Area Scrolling */
        .main {
            scroll-behavior: smooth;
            position: relative;
        }
        
        /* Custom Scrollbar for Main Content */
        .main::-webkit-scrollbar {
            width: 12px;
            background: transparent;
        }
        
        .main::-webkit-scrollbar-track {
            background: rgba(24, 24, 24, 0.5);
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .main::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #1DB954, #1ED760);
            border-radius: 10px;
            border: 2px solid rgba(24, 24, 24, 0.5);
            transition: all 0.3s ease;
        }
        
        .main::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(to bottom, #1ED760, #22E868);
            box-shadow: 0 0 15px rgba(29, 185, 84, 0.5);
        }
        
        /* Scroll Progress Indicator */
        .scroll-progress {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1DB954, #1ED760);
            transform-origin: left;
            animation: scrollProgress linear;
            animation-timeline: scroll();
            z-index: 9999;
        }
        
        /* Parallax Effect for Headers */
        h1, h2, h3 {
            transition: transform 0.3s ease-out;
        }
        
        /* Fade In on Scroll */
        .element-container {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Back to Top Button */
        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #1DB954, #1ED760);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            opacity: 0;
            transform: translateY(100px);
            transition: all 0.3s ease;
            z-index: 999;
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        }
        
        .back-to-top.visible {
            opacity: 1;
            transform: translateY(0);
        }
        
        .back-to-top:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(29, 185, 84, 0.5);
        }
        
        /* Smooth Anchor Links */
        a[href^="#"] {
            transition: all 0.3s ease;
        }
        
        /* Section Transitions */
        .stTabs [data-baseweb="tab-panel"] {
            animation: tabFadeIn 0.5s ease;
        }
        
        @keyframes tabFadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Scroll Snap for Sections (Optional) */
        .main > div {
            scroll-snap-type: y proximity;
        }
        
        .main > div > div[data-testid="stVerticalBlock"] > div {
            scroll-snap-align: start;
            scroll-margin-top: 20px;
        }
                        
        /* ============ NAVIGATION BUTTON STYLING (FIXED) ============ */
        
        /* Grey buttons (default inactive state) - ONLY sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: #282828 !important;
            color: #B3B3B3 !important;
            border: 1px solid #404040 !important;
            border-radius: 500px !important;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        
        /* Hover for inactive sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #404040 !important;
            color: #FFFFFF !important;
            transform: translateX(5px);
            border-color: #606060 !important;
        }
        
        /* Green active buttons in sidebar (primary) */
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #1DB954 0%, #1ED760 100%) !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 500px !important;
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3) !important;
        }
        
        /* ============ MAIN CONTENT BUTTONS (INCLUDING AI CONSULTANT) ============ */
        
        /* All buttons in main area - GREEN by default */
        .main .stButton > button,
        [data-testid="stVerticalBlock"] .stButton > button,
        .element-container .stButton > button {
            background: linear-gradient(135deg, #1DB954 0%, #1ED760 100%) !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 500px !important;
            font-weight: 700 !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.2s ease !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.2) !important;
        }
        
        /* Hover for main area buttons */
        .main .stButton > button:hover,
        [data-testid="stVerticalBlock"] .stButton > button:hover,
        .element-container .stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 7px 25px rgba(29, 185, 84, 0.4) !important;
            background: linear-gradient(135deg, #1ED760 0%, #22E868 100%) !important;
        }
        
        /* Active/clicked state for main buttons */
        .main .stButton > button:active,
        [data-testid="stVerticalBlock"] .stButton > button:active {
            transform: translateY(0) scale(0.98) !important;
        }
        
        /* Ensure AI Consultant tab buttons are green and rounded */
        .main button[key*="btn_"],
        .main button[key*="custom_"] {
            background: linear-gradient(135deg, #1DB954 0%, #1ED760 100%) !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 500px !important;
            min-height: 45px !important;
            font-weight: 600 !important;
        }
        
        /* ============ SIDEBAR EFFECTS ============ */
    
        /* Animated Gradient Background */
        section[data-testid="stSidebar"] {
            background: linear-gradient(-45deg, #000000, #0a0a0a, #1a1a1a, #0f0f0f);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            border-right: 2px solid transparent;
            border-image: linear-gradient(to bottom, #1DB954, #1ED760, #1DB954);
            border-image-slice: 1;
            position: relative;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Glowing Border Effect */
        section[data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 2px;
            height: 100%;
            background: linear-gradient(to bottom, 
                transparent,
                #1DB954,
                #1ED760,
                #1DB954,
                transparent
            );
            animation: borderGlow 3s linear infinite;
        }
        
        @keyframes borderGlow {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        /* Sidebar Header Animation */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            background: linear-gradient(90deg, #1DB954, #1ED760, #1DB954);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textShine 3s linear infinite;
        }
        
        @keyframes textShine {
            to {
                background-position: 200% center;
            }
        }
        
        /* Sidebar Widgets Slide-in Effect */
        section[data-testid="stSidebar"] .element-container {
            animation: slideInLeft 0.5s ease-out;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"] .element-container:hover {
            transform: translateX(5px);
            background: rgba(29, 185, 84, 0.05);
            border-left: 3px solid #1DB954;
            padding-left: 10px;
            margin-left: -10px;
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Expandable Sections with Smooth Animation */
        section[data-testid="stSidebar"] .streamlit-expanderHeader {
            background: linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.1), transparent);
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            background: rgba(29, 185, 84, 0.2);
            transform: scale(1.02);
        }
        
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        
        /* Sidebar Scrollbar Custom */
        section[data-testid="stSidebar"] > div:first-child {
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar {
            width: 6px;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 3px;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #1DB954, #1ED760);
            border-radius: 3px;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: #1ED760;
            box-shadow: 0 0 10px rgba(29, 185, 84, 0.5);
        }
        
        /* Navigation Items Hover Effect */
        section[data-testid="stSidebar"] .stRadio > div {
            background: transparent;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"] .stRadio > div:hover {
            background: linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.1));
            border-radius: 10px;
        }
        
        /* ============ SCROLLBAR STYLING ============ */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(24, 24, 24, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #1DB954, #1ED760);
            border-radius: 10px;
            transition: all 0.2s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(to bottom, #1ED760, #22E868);
            box-shadow: 0 0 15px rgba(29, 185, 84, 0.5);
        }
        
        /* ============ TAB ANIMATIONS ============ */
        .stTabs [data-baseweb="tab"] {
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-2px);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
            animation: tabPulse 2s ease-in-out infinite;
        }
        
        @keyframes tabPulse {
            0%, 100% { box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(29, 185, 84, 0.5); }
        }
        
        /* ============ METRICS ANIMATIONS ============ */
        [data-testid="metric-container"] {
            transition: all 0.2s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 5px 20px rgba(29, 185, 84, 0.3);
        }
        
        /* ============ FADE IN ANIMATION (FAST) ============ */
        .element-container {
            animation: fadeInUp 0.3s ease-out;
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
        
        /* ============ SELECTBOX & RADIO EFFECTS ============ */
        .stSelectbox > div > div {
            background: rgba(24, 24, 24, 0.8);
            border: 2px solid transparent;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #1DB954;
            box-shadow: 0 0 20px rgba(29, 185, 84, 0.2);
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #1ED760;
            box-shadow: 0 0 30px rgba(29, 185, 84, 0.3);
            transform: scale(1.02);
        }
        
        .stRadio > div {
            display: flex;
            gap: 10px;
            padding: 5px;
            background: rgba(24, 24, 24, 0.5);
            border-radius: 25px;
        }
                
        .stRadio > div:hover {
            background: rgba(29, 185, 84, 0.1);
        }
                
        .stRadio > div > label {
            background: transparent;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
                
        .stRadio > div > label:hover {
            background: rgba(29, 185, 84, 0.1);
            transform: scale(1.05);
        }
                
        .stRadio > div > label[data-checked="true"] {
            background: linear-gradient(135deg, #1DB954, #1ED760);
            color: #000;
            font-weight: bold;
            animation: radioSelect 0.3s ease;
        }
        
        @keyframes radioSelect {
            0% { transform: scale(0.9); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
               
        /* ============ EXPANDER EFFECTS ============ */
        .streamlit-expanderHeader {
            transition: all 0.2s ease;
            background: linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.05), transparent);
            background-size: 200% 100%;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(29, 185, 84, 0.15);
            transform: scale(1.01);
        }
        
        /* ============ INFO/WARNING/ERROR BOXES ============ */
        .stAlert {
            animation: slideInLeft 0.3s ease;
            transition: all 0.2s ease;
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* ============ PLOTLY CHARTS CONTAINER ============ */
        [data-testid="stPlotlyChart"] {
            animation: fadeIn 0.4s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
                
    <!-- Add Scroll Progress Bar -->
    <div class="scroll-progress"></div>

    <!-- Add Back to Top Button -->
    <div class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        ⬆
    </div>

    <script>
        // Show/Hide Back to Top Button
        window.addEventListener('scroll', function() {
            const backToTop = document.querySelector('.back-to-top');
            if (window.scrollY > 300) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        });
        
        // Update scroll progress
        window.addEventListener('scroll', function() {
            const scrollProgress = document.querySelector('.scroll-progress');
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (window.scrollY / scrollHeight) * 100;
            if (scrollProgress) {
                scrollProgress.style.transform = `scaleX(${scrollPercent / 100})`;
            }
        });
    </script>
    
    <!-- Scroll to top button -->
    <a href="#" id="scrollTop" style="
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #1DB954, #1ED760);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: black;
        text-decoration: none;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        opacity: 0.8;
        transition: all 0.2s ease;
        z-index: 999;
    " onmouseover="this.style.transform='translateY(-5px)'; this.style.opacity='1';" 
       onmouseout="this.style.transform='translateY(0)'; this.style.opacity='0.8';">↑</a>
    """, unsafe_allow_html=True)

apply_all_effects()

# --- Cached data and animations with progress details ---
@st.cache_data
def load_data():
    """Load data - cached after first load"""
    try:
        # Load all datasets
        data_main = pd.read_csv('data/data.csv')
        data_by_artist = pd.read_csv('data/data_by_artist.csv')
        data_by_genres = pd.read_csv('data/data_by_genres.csv')
        data_by_year = pd.read_csv('data/data_by_year.csv')
        data_w_genres = pd.read_csv('data/data_w_genres.csv')
        
        # Basic cleaning
        data_main = data_main.dropna(subset=['popularity', 'energy', 'danceability'])
        
        return {
            'main': data_main,
            'by_artist': data_by_artist,
            'by_genres': data_by_genres,
            'by_year': data_by_year,
            'with_genres': data_w_genres
        }
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo não encontrado - {e}")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

# --- Loading with animation and session tracking ---
if 'initial_load_complete' not in st.session_state:
    # First time loading - show animation
    loading_container = st.container()
    
    with loading_container:
        st.markdown("### ⏳ Initializing...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Start loading
        status_text.text('📂 Locating data files...')
        progress_bar.progress(20)
        time.sleep(0.3)  # Brief pause for UX
        
        # Step 2: Load data
        status_text.text('📊 Loading 160,000+ tracks...')
        progress_bar.progress(40)
        
        with st.spinner('Processing music database...'):
            music_data = load_data()
        
        if music_data:
            # Step 3: Validate
            status_text.text('✅ Validating data integrity...')
            progress_bar.progress(60)
            time.sleep(0.3)
            
            # Step 4: Calculate stats
            status_text.text('📈 Calculating statistics...')
            progress_bar.progress(80)
            total_tracks = len(music_data['main'])
            total_artists = len(music_data['by_artist'])
            time.sleep(0.3)
            
            # Step 5: Complete
            status_text.text('🎉 Ready!')
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear loading UI
            #progress_bar.empty()
            #status_text.empty()
            
            
            # Show success metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tracks Loaded", f"{total_tracks:,}")
            with col2:
                st.metric("Artists", f"{total_artists:,}")
            with col3:
                st.metric("Years", "1921-2020")
            

            loading_container.empty()

            # Mark as loaded
            st.session_state.initial_load_complete = True
            
            # Auto-clear success message after 3 seconds
            success_placeholder = st.empty()
            success_placeholder.success("✅ MusicInsights AI is ready! Explore 100 years of music evolution.")
            time.sleep(3)
            success_placeholder.empty()
            st.rerun()

        else:
            # Error occurred
            progress_bar.empty()
            status_text.empty()
            st.error("❌ Failed to load data. Please check the data files.")
            st.stop()
else:
    # Subsequent runs - load instantly from cache
    music_data = load_data()
    # Ensure any stray success messages are cleared
    st.empty()


# After loading, show quick stats
if music_data and 'stats_shown' not in st.session_state:
        
    # Show the stats in a "toast" notification
    st.toast(
        f"""
        **Dataset Ready**
    
        
        - {len(music_data['main']):,} tracks
        - {len(music_data['by_artist']):,} artists 
        - {len(music_data['by_genres']):,} genres
        - 100 years of music ✨
        """,
        icon="✅"  # <--- This makes it look like a "success" toast
    )
    
    # Set the flag so it only runs once
    st.session_state.stats_shown = True

# --- Initialize Session State for Navigation ---
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"


# --- Sidebar Navigation ---

spotify_logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png"
dataset_url = "https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks"

header_html = f"""
<div style="
    display: flex;        
    align-items: center;     
    margin-bottom: 20px;     
">
    <img src="{spotify_logo_url}" width="80" style="margin-right: 15px;">
    
    <h2 style='color: #1DB954; margin: 0; padding: 0;'>
        MusicInsights AI
    </h2>
</div>
"""

st.sidebar.html(header_html) 

# Navigation Section
st.sidebar.markdown("### 🎵 Navigation")

tabs = ["AI Consultant", "Dashboard", "Insights", "Data Explorer"]

for tab in tabs:
    is_active = st.session_state.current_tab == tab
    
    # Set icon for each tab
    if tab == "AI Consultant":
        icon = "🧠"
        label = f"{icon} {tab} 💬"
    else:
        icon_map = {
            "Dashboard": "📊",
            "Insights": "💡",
            "Data Explorer": "📁"
        }
        icon = icon_map.get(tab, "📄")
        label = f"{icon} {tab}"
    
    # Create button - same style for all
    if st.sidebar.button(
        label,
        key=f"nav_{tab.replace(' ', '_')}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.current_tab = tab
        st.rerun()

# Show active indicator for AI Assistant
if st.session_state.current_tab == "AI Consultant":
    st.sidebar.markdown("""
    <div style='
        background: linear-gradient(135deg, #1DB954 0%, #1ED760 100%);
        padding: 10px; 
        border-radius: 8px; 
        margin: 10px 0;
        animation: pulse 2s infinite;
    '>
        <p style='color: black; font-weight: bold; margin: 0; text-align: center;'>
            🤖 AI Assistant Active
        </p>
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.02); opacity: 0.9; }
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
            

# ADD DATASET STATS 

def display_sidebar_stats(music_data, df_filtered=None):
    total_tracks = len(music_data['main'])
    current_df = df_filtered if df_filtered is not None else music_data['main']
    current_tracks = len(current_df)
    percentage = (current_tracks / total_tracks * 100) if total_tracks else 0

    if current_tracks:
        years = f"{int(current_df['year'].min())}-{int(current_df['year'].max())}"
        artists = current_df['artists'].nunique()
        avg_pop = current_df['popularity'].mean()
        explicit_count = int(current_df['explicit'].sum()) if 'explicit' in current_df.columns else 0
    else:
        years, artists, avg_pop, explicit_count = "N/A", 0, 0, 0

    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border: 2px solid #1DB954; border-radius: 12px; padding: 15px; margin: 10px 0;">
      <h4 style="color:#1DB954; margin:0 0 10px 0; text-align:center;">📊 Active Dataset</h4>
      <div style="text-align:center; padding:10px; background:#0a0a0a; border-radius:8px; margin-bottom:10px;">
        <div style="color:#1DB954; font-size:28px; font-weight:bold;">{current_tracks:,}</div>
        <div style="color:#888; font-size:12px;">tracks selected ({percentage:.0f}%)</div>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
        <div style="background:#0a0a0a; padding:8px; border-radius:6px; text-align:center;">
          <div style="color:#1DB954; font-weight:bold;">{years}</div>
          <div style="color:#666; font-size:12px;">Years</div>
        </div>
        <div style="background:#0a0a0a; padding:8px; border-radius:6px; text-align:center;">
          <div style="color:#1DB954; font-weight:bold;">{artists:,}</div>
          <div style="color:#666; font-size:12px;">Artists</div>
        </div>
        <div style="background:#0a0a0a; padding:8px; border-radius:6px; text-align:center;">
          <div style="color:#1DB954; font-weight:bold;">{avg_pop:.0f}</div>
          <div style="color:#666; font-size:12px;">Avg Pop</div>
        </div>
        <div style="background:#0a0a0a; padding:8px; border-radius:6px; text-align:center;">
          <div style="color:#1DB954; font-weight:bold;">{explicit_count:,}</div>
          <div style="color:#666; font-size:12px;">Explicit</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------
# CRIAR A FERRAMENTA CUSTOMIZADA COM IA
# --------------------------------------------------------
@tool
def PythonCodeExecutor(code: str) -> str:
    """
    Execute Python code for data analysis using the REAL datasets.
    IMPORTANT:
    - You MUST use the actual DataFrames below (do NOT fabricate data):
        - df         -> alias to df_tracks (tracks-level dataset)
        - df_tracks  -> data.csv (tracks-level)
        - df_year    -> data_by_year.csv (aggregated by year)
        - df_artist  -> data_by_artist.csv (aggregated by artist)
        - df_genres  -> data_by_genres.csv (aggregated by genre)
        - df_w_genres-> data_w_genres.csv (tracks with genres)
        - pd, np are available
    - Always verify results with the real data above.
    - Use print(...) to output your results. The tool captures stdout.

    Examples:
    - print(df['popularity'].mean())
    - print(df_year[['year','danceability','energy','valence']].corr()['popularity'])
    - subset = df_w_genres[df_w_genres['genres'].str.contains('hip hop', case=False, na=False)]
      print(subset[subset['year'].between(1990, 1999)][['energy','popularity']].corr())
    """
    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()

        if music_data is None:
            return "ERROR: Datasets not loaded."

        # Expose SAFE copies of DataFrames
        df_tracks = music_data["main"].copy()
        df_year = music_data["by_year"].copy()
        df_artist = music_data["by_artist"].copy()
        df_genres = music_data["by_genres"].copy()
        df_w_genres = music_data["with_genres"].copy()

        # Default 'df' alias to tracks-level dataset
        df = df_tracks

        # Basic anti-fabrication (disallow creating DataFrame from dict literal)
        if "pd.DataFrame" in code and "{" in code:
            return "ERROR: Do NOT create fake DataFrames. Use the provided DataFrames only (df, df_tracks, df_year, df_artist, df_genres, df_w_genres)."

        # Execute in a controlled namespace
        exec_env = {
            "pd": pd,
            "np": np,
            "df": df,
            "df_tracks": df_tracks,
            "df_year": df_year,
            "df_artist": df_artist,
            "df_genres": df_genres,
            "df_w_genres": df_w_genres,
        }
        exec(code, exec_env, {})

        sys.stdout = old_stdout
        output = redirected_output.getvalue()

        if not output.strip():
            return "ERROR: No output generated. Be sure to use print(...) to display results."
        return output

    except Exception as e:
        try:
            sys.stdout = old_stdout
        except Exception:
            pass
        return f"Erro: {e}"

tools = [PythonCodeExecutor]

# --- Main Content Area ---
if music_data is not None:
    
    # Get main dataframe
    df = music_data['main'].copy()
    df_year = music_data['by_year'].copy()
    df_genres = music_data['by_genres'].copy()
    df_artist = music_data['by_artist'].copy()
    
    # Add decade column globally
    df['decade'] = (df['year'] // 10) * 10
    
    # --- Key Mapping ---
    KEY_MAP = {0:'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',6:'F#',7:'G',8:'G#',9:'A',10:'A#',11:'B'}
    key_options = list(KEY_MAP.values())

    # ---------------------------------------------------
    # --- 1. INITIALIZE ALL FILTER STATES ---
    # This ensures all session_state keys exist on the first run.
    # ---------------------------------------------------

    # Main Filters
    if 'global_year_slider' not in st.session_state:
        st.session_state.global_year_slider = (int(df['year'].min()), int(df['year'].max()))
    if 'global_popularity_filter' not in st.session_state:
        st.session_state.global_popularity_filter = (0, 100)
    if 'global_explicit_filter' not in st.session_state:
        st.session_state.global_explicit_filter = "All"
    if 'global_key_filter' not in st.session_state:
        st.session_state.global_key_filter = key_options

    # Audio Feature Filters
    if 'f_dance' not in st.session_state:
        st.session_state.f_dance = (0, 100)
    if 'f_energy' not in st.session_state:
        st.session_state.f_energy = (0, 100)
    if 'f_valence' not in st.session_state:
        st.session_state.f_valence = (0, 100)
    if 'f_loud' not in st.session_state:
        st.session_state.f_loud = (-60.0, 0.0)
    if 'f_acoustic' not in st.session_state:
        st.session_state.f_acoustic = (0, 100)
    if 'f_instr' not in st.session_state:
        st.session_state.f_instr = (0, 100)
    if 'f_live' not in st.session_state:
        st.session_state.f_live = (0, 100)
    if 'f_speech' not in st.session_state:
        st.session_state.f_speech = (0, 100)

    # Comparison Mode
    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False
        
    # Dashboard Viz Picker
    if 'viz_picker' not in st.session_state:
        st.session_state.viz_picker = "🏠 Welcome & Table of Contents"

    # --- FUNCTION FOR CLEANING FILTERS ---
    def clear_filters():
        """
        Resets all session_state filter keys to their default values.
        """
        
        # --- 1. Main Filters ---
        # (Note: We use the actual min/max years here. 
        #  You can also hardcode them, e.g., (1921, 2020))
        st.session_state.global_year_slider = (int(df['year'].min()), int(df['year'].max()))
        st.session_state.global_popularity_filter = (0, 100)
        st.session_state.global_explicit_filter = "All"
        # Assumes KEY_MAP is a global constant
        st.session_state.global_key_filter = list(KEY_MAP.values()) 
        
        # --- 2. Audio Feature Filters ---
        st.session_state.f_dance = (0, 100)
        st.session_state.f_energy = (0, 100)
        st.session_state.f_valence = (0, 100)
        st.session_state.f_loud = (-60.0, 0.0)
        st.session_state.f_acoustic = (0, 100)
        st.session_state.f_instr = (0, 100)
        st.session_state.f_live = (0, 100)
        st.session_state.f_speech = (0, 100)
        
        # --- 3. Comparison Mode ---
        st.session_state.compare_mode = False
        
        # --- 4. Dashboard Viz Picker ---
        # This resets the selectbox back to the Welcome Page
        st.session_state.viz_picker = "🏠 Welcome & Table of Contents"

    # ---------------------------------------------------
    # GLOBAL FILTERS IN SIDEBAR
    # ---------------------------------------------------
    st.sidebar.markdown("### 🎛️ Global Filters")
    with st.sidebar.expander("Filter Options", expanded=True):

        st.button(
            "Clear All Filters 🔄",
            on_click=clear_filters,
            use_container_width=True
        )
        st.markdown("---")
        
        # Get the min and max year from the main dataframe
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        
        # Year Range Filter
        year_range = st.slider(
            "Year Range:",
            min_value=min_year,
            max_value=max_year,
            key="global_year_slider" # 'value' is gone!
        )
        
        # Popularity Range Filter
        popularity_range = st.slider(
            "Popularity Range:",
            min_value=0,
            max_value=100,
            key="global_popularity_filter" # 'value' is gone!
        )
        
        # Explicit Content Filter
        explicit_filter = st.radio(
            "Content Type:",
            options=["All", "Clean Only", "Explicit Only"],
            key="global_explicit_filter" # 'index' is gone!
        )
        
        # Musical Key Filter
        key_filter_names = st.multiselect(
            "Filter by Key:",
            options=key_options,
            key="global_key_filter" # 'default' is gone!
        )
        
        # ---------------------------------------------------
        # COMPARISON MODE SECTION (Final, Clean Structure)
        # ---------------------------------------------------
        st.sidebar.markdown("---")
        compare_mode = st.checkbox("Enable Comparison Mode", key="compare_mode")

        # --- Local function to render the comparison UI ---
        # This keeps the code clean and isolated.
        def _render_comparison_widgets():
            
            # 1. Decade List 
            decade_options_f = sorted(df_filtered['decade'].unique().tolist())
            
            # 2. Genre List 
            try:
                gframe_filtered = align_genre_frame(music_data["with_genres"], filters)
                df_genres_agg = aggregate_by_genre(gframe_filtered)
                genre_list_f = df_genres_agg.nlargest(20, 'popularity')['genres'].tolist()
            except:
                genre_list_f = []

            # 3. Artist List
            try:
                df_artist_agg = aggregate_by_artist(df_filtered)
                artist_list_f = df_artist_agg.nlargest(20, 'popularity')['artist_clean'].tolist()
            except:
                artist_list_f = []

            # --- RENDER SELECTBOXES ---
            compare_type = st.radio("Compare:", ["Decades", "Genres", "Artists"], key="compare_type")

            if compare_type == "Decades":
                if not decade_options_f:
                    st.info("Not enough data or decades in the current filter for comparison.")
                else:
                    decade1 = st.selectbox("First Decade", decade_options_f, index=0, key="decade1")
                    decade2 = st.selectbox("Second Decade", decade_options_f, index=min(1, len(decade_options_f) - 1), key="decade2")
                    st.session_state.compare_items = (decade1, decade2)
            
            elif compare_type == "Genres":
                if not genre_list_f:
                    st.info("No genres found in the current filter for comparison.")
                else:
                    genre1 = st.selectbox("First Genre", genre_list_f, index=0, key="genre1")
                    genre2 = st.selectbox("Second Genre", genre_list_f, index=min(1, len(genre_list_f) - 1), key="genre2")
                    st.session_state.compare_items = (genre1, genre2)
            
            else:  # Artists
                if not artist_list_f:
                    st.info("No artists found in the current filter for comparison.")
                else:
                    artist1 = st.selectbox("First Artist", artist_list_f, index=0, key="artist1")
                    artist2 = st.selectbox("Second Artist", artist_list_f, index=min(1, len(artist_list_f) - 1), key="second_artist2")
                    st.session_state.compare_items = (artist1, artist2)


        # --- Main Rendering Logic ---
        # Check if data and the checkbox are active
        if compare_mode:
            # Check if df_filtered is defined and not empty
            if 'df_filtered' in globals() and not df_filtered.empty:
                _render_comparison_widgets()
            else:
                # Data not ready OR filters returned no tracks
                st.warning("Please wait for the initial data load to complete before using comparison mode.")

        # ---------------------------------------------------
        # NEW: AUDIO FEATURE FILTERS (in 2 columns, as percentages)
        # ---------------------------------------------------
        st.sidebar.markdown("### 🔬 Audio Feature Filters")
        with st.sidebar.expander("Filter by Audio Features", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                dance_range = st.slider(
                    "Danceability:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_dance", format="%d%%"  # <-- FIX
                )
                energy_range = st.slider(
                    "Energy:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_energy", format="%d%%" # <-- FIX
                )
                valence_range = st.slider(
                    "Valence:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_valence", format="%d%%" # <-- FIX
                )
                # Loudness is NOT a percentage, so we leave it as-is
                loudness_range = st.slider(
                    "Loudness (dB):", -60.0, 0.0, (-60.0, 0.0), 0.1, key="f_loud"
                )
                
            with col2:
                acoustic_range = st.slider(
                    "Acousticness:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_acoustic", format="%d%%" # <-- FIX
                )
                instr_range = st.slider(
                    "Instrumentalness:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_instr", format="%d%%" # <-- FIX
                )
                live_range = st.slider(
                    "Liveness:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_live", format="%d%%" # <-- FIX
                )
                speech_range = st.slider(
                    "Speechiness:",
                    min_value=0, max_value=100, value=(0, 100), step=1,
                    key="f_speech", format="%d%%" # <-- FIX
                )
    
    selected_key_numbers = [k for k, v in KEY_MAP.items() if v in key_filter_names]

    # --- Global filtering primitives ---
    @dataclass(frozen=True)
    class FilterState:
        year_start: int
        year_end: int
        pop_min: int
        pop_max: int
        explicit: str
        keys: tuple[int, ...]
        
        # --- ADD THESE 8 NEW LINES ---
        dance_range: tuple[float, float]
        energy_range: tuple[float, float]
        valence_range: tuple[float, float]
        loudness_range: tuple[float, float]
        acoustic_range: tuple[float, float]
        instr_range: tuple[float, float]
        live_range: tuple[float, float]
        speech_range: tuple[float, float]

    @st.cache_data(show_spinner=False)
    def filter_tracks(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
        # These first two lines are from your existing code
        q = df[(df["year"] >= f.year_start) & (df["year"] <= f.year_end)]
        q = q[(q["popularity"] >= f.pop_min) & (q["popularity"] <= f.pop_max)]
        
        # --- FIX: Divide all percentage-based ranges by 100.0 ---
        q = q[(q["danceability"] >= f.dance_range[0] / 100.0) & (q["danceability"] <= f.dance_range[1] / 100.0)]
        q = q[(q["energy"] >= f.energy_range[0] / 100.0) & (q["energy"] <= f.energy_range[1] / 100.0)]
        q = q[(q["valence"] >= f.valence_range[0] / 100.0) & (q["valence"] <= f.valence_range[1] / 100.0)]
        q = q[(q["acousticness"] >= f.acoustic_range[0] / 100.0) & (q["acousticness"] <= f.acoustic_range[1] / 100.0)]
        q = q[(q["instrumentalness"] >= f.instr_range[0] / 100.0) & (q["instrumentalness"] <= f.instr_range[1] / 100.0)]
        q = q[(q["liveness"] >= f.live_range[0] / 100.0) & (q["liveness"] <= f.live_range[1] / 100.0)]
        q = q[(q["speechiness"] >= f.speech_range[0] / 100.0) & (q["speechiness"] <= f.speech_range[1] / 100.0)]
        
        # --- Loudness is NOT divided by 100 ---
        q = q[(q["loudness"] >= f.loudness_range[0]) & (q["loudness"] <= f.loudness_range[1])]

        # These are from your existing code
        if f.explicit == "Clean Only":
            q = q[q["explicit"] == 0]
        elif f.explicit == "Explicit Only":
            q = q[q["explicit"] == 1]
        if f.keys:
            q = q[q["key"].isin(f.keys)]
            
        return q

    # Build FilterState from sidebar widgets
    filters = FilterState(
        # Your existing 6 fields
        year_start=year_range[0],
        year_end=year_range[1],
        pop_min=popularity_range[0],
        pop_max=popularity_range[1],
        explicit=explicit_filter,
        keys=tuple(sorted(selected_key_numbers)),
        
        # --- ADD THESE 8 NEW LINES ---
        dance_range=dance_range,
        energy_range=energy_range,
        valence_range=valence_range,
        loudness_range=loudness_range,
        acoustic_range=acoustic_range,
        instr_range=instr_range,
        live_range=live_range,
        speech_range=speech_range
    )

    df_filtered = filter_tracks(df, filters)

    # --- CRITICAL FIX: Set the data ready flag ---
    # This must be done AFTER data loading and filtering are complete.
    if 'data_ready' not in st.session_state:
        st.session_state.data_ready = True

    # --- Aggregators (respect current filters) ---
    @st.cache_data(show_spinner=False)
    def aggregate_by_year(df_tracks: pd.DataFrame) -> pd.DataFrame:
        if df_tracks.empty:
            return pd.DataFrame(columns=["year","popularity","energy","danceability","valence","acousticness","instrumentalness","speechiness","liveness","loudness","tempo","duration_ms"])
        agg = (df_tracks.groupby("year", as_index=False)
            .agg(popularity=("popularity","mean"),
                energy=("energy","mean"),
                danceability=("danceability","mean"),
                valence=("valence","mean"),
                acousticness=("acousticness","mean"),
                instrumentalness=("instrumentalness","mean"),
                speechiness=("speechiness","mean"),
                liveness=("liveness","mean"),
                loudness=("loudness","mean"),
                tempo=("tempo","mean"),
                duration_ms=("duration_ms","mean")))
        return agg

    @st.cache_data(show_spinner=False)
    def aggregate_by_artist(df_tracks: pd.DataFrame) -> pd.DataFrame:
        if df_tracks.empty:
            return pd.DataFrame(columns=["artist_clean","popularity","energy","valence","count"])
        d = df_tracks.copy()
        d["artist_clean"] = (d["artists"]
            .str.replace(r"[\[\]'\"]", "", regex=True)
            .str.split(",").str[0].str.strip())
        agg = (d.groupby("artist_clean", as_index=False)
            .agg(popularity=("popularity","mean"),
                energy=("energy","mean"),
                valence=("valence","mean"),
                count=("name","count")))
        return agg

    @st.cache_data(show_spinner=False)
    def align_genre_frame(df_w_genres: pd.DataFrame, f: FilterState) -> pd.DataFrame:
        g = df_w_genres.copy()
        # Apply what we can (may not have explicit/key in this table)
        if "year" in g.columns:
            g = g[(g["year"] >= f.year_start) & (g["year"] <= f.year_end)]
        if "popularity" in g.columns:
            g = g[(g["popularity"] >= f.pop_min) & (g["popularity"] <= f.pop_max)]
        if "explicit" in g.columns and f.explicit != "All":
            want = 0 if f.explicit == "Clean Only" else 1
            g = g[g["explicit"] == want]
        return g

    @st.cache_data(show_spinner=False)
    def aggregate_by_genre(df_w_genres_filtered: pd.DataFrame) -> pd.DataFrame:
        if df_w_genres_filtered.empty:
            return pd.DataFrame(columns=["genres","popularity","energy","danceability","valence","acousticness","speechiness","tempo"])
        agg = (df_w_genres_filtered.groupby("genres", as_index=False)
            .agg(popularity=("popularity","mean"),
                energy=("energy","mean"),
                danceability=("danceability","mean"),
                valence=("valence","mean"),
                acousticness=("acousticness","mean"),
                speechiness=("speechiness","mean"),
                tempo=("tempo","mean")))
        return agg
    
    # --------------------------------------------------------
    # NEW WELCOME PAGE FUNCTION (with consistent font sizes)
    # --------------------------------------------------------
    def render_welcome_page():
        st.header("🏠 Welcome to MusicInsights AI")
        st.markdown("This dashboard provides a deep dive into over 100 years of music data from Spotify. Explore trends in audio features and understand what makes songs popular.")
        
         # HOW TO INTERACT
        with st.expander("How to interact", expanded=True):
            st.markdown("""
            - Use the Global Filters in the sidebar to subset the dataset.
            - Select a visualization from the dropdown below, or click a 'Go to' button in the Table of Contents.
            - Each visualization offers its own comparison mode when relevant.
            - On mobile, scroll horizontally when needed.
            - This page loads only the selected visualization for speed.
            """)

        st.subheader("About the Author & Data")
        st.markdown(f"""
        This dashboard was created by **Eduardo Cornelsen**.
        - **GitHub Repository:** [github.com/duducornelsen](https://github.com/duducornelsen)
        - **LinkedIn Profile:** [linkedin.com/in/eduardo-cornelsen](https://www.linkedin.com/in/eduardo-cornelsen-389b1b1a4/)
        
        The data for this project is sourced from the [Spotify Dataset 1921-2020](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks) on Kaggle.
        """)
        
        st.divider()
        
        st.subheader("📊 Table of Contents (Visualizations)")
        st.info("Click on any title to see its details, then click the 'Go to...' button to load the chart.")

        # --- DEFINE OUR NEW STYLES ---
        TOC_INFO_BOX = """
        <div style="
            background-color: rgba(0, 104, 225, 0.1); 
            border: 1px solid rgba(0, 104, 225, 0.5);
            border-radius: 6px; 
            padding: 10px 12px; 
            margin-bottom: 10px;
        ">
            <strong style="
                color: #FFFFFF; 
                text-align: center; 
                display: block; 
                /* FIX: font-size: 1.1em; REMOVED to match expander title */
            ">
                Key Problem
            </strong>
            
            <hr style="
                border: none; 
                border-top: 1px solid rgba(0, 104, 225, 0.5); 
                margin: 8px 0;
            ">
            
            <span style="
                color: #E0E0E0; 
                text-align: center; 
                display: block;
            ">
                {problem_text}
            </span>
        </div>
        """
        
        TOC_DETAILS_MD = """
        - **Columns Used:** `{columns}`
        - **Main `.groupby()`:** `{groupby}`
        """

        # --- LIST OF ALL VISUALIZATIONS ---
        # (This data is all the same)
        viz_toc = {
            "📈 Evolution of Features": {
                "problem": "How have the sonic qualities of music (like energy and acousticness) changed over the last 100 years?",
                "cols": "year, danceability, energy, valence, acousticness, instrumentalness, speechiness, liveness, loudness",
                "gb": "df_filtered.groupby('year', ...)"
            },
            "📊 Popularity vs Features": {
                "problem": "Is there a correlation between a song's audio features (like high energy) and its popularity? Which feature is the best predictor of a hit?",
                "cols": "popularity, energy, danceability, valence, acousticness, instrumentalness, speechiness",
                "gb": "No groupby; plots sampled raw data."
            },
            "🎸 Genre DNA": {
                "problem": "What is the unique audio 'signature' of different genres? How do genres compare in terms of energy, valence, etc.?",
                "cols": "genres, popularity, energy, danceability, valence, acousticness, speechiness",
                "gb": "gframe_filtered.groupby('genres', ...)"
            },
            "🔞 Explicit Strategy": {
                "problem": "Is explicit content a successful commercial strategy? How does it impact popularity and other audio features?",
                "cols": "explicit, popularity, energy, danceability, valence, speechiness",
                "gb": "No groupby; maps and plots data directly."
            },
            "📈 Explicit Over Time": {
                "problem": "When did explicit content become mainstream? Which genres pioneered it, and how has its popularity trended over time?",
                "cols": "year, explicit, genres, popularity",
                "gb": "df_filtered.groupby(['year','explicit']), gframe_filtered.groupby('genres')"
            },
            "🔗 Feature Relationships": {
                "problem": "Which audio features are most strongly correlated? Can we identify a 'formula' for success by analyzing feature averages for high-popularity songs?",
                "cols": "danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, popularity",
                "gb": "df_success.groupby('success_level')"
            },
            "🔗 Temporal Trends": {
                "problem": "How have song duration and tempo (BPM) changed over the decades? Can we project future trends?",
                "cols": "year, duration_ms, tempo",
                "gb": "df_filtered.groupby('year') (via aggregate_by_year)"
            },
            "👤 Artist Success Patterns": {
                "problem": "What separates consistent hitmakers from 'one-hit wonders'? This analyzes volume vs. quality, consistency, and the audio signature of top artists.",
                "cols": "artists, popularity, energy, valence, count",
                "gb": "df_filtered.groupby('artist_clean') (via aggregate_by_artist)"
            },
            "🔍 Feature Explorer": {
                "problem": "What interactive combinations of features (e.g., Energy vs. Danceability) lead to the most popular songs? Where are the 'sweet spots'?",
                "cols": "User-selected X/Y features (e.g., danceability, energy, popularity), decade",
                "gb": "No groupby; plots sampled raw data."
            },
            "🎵 Key & Mode": {
                "problem": "Do certain musical keys (C, G, A#) or modes (Major vs. Minor) correlate with higher popularity or specific emotions (valence)?",
                "cols": "key, mode, popularity, valence",
                "gb": "df_keys.groupby(['key_name', 'mode_name'])"
            },
            "📅 Decade Evolution": {
                "problem": "How does the distribution of a single feature change across decades? Are modern songs more or less diverse in their sound?",
                "cols": "decade, User-selected feature (e.g., valence)",
                "gb": "df_decades.groupby('decade')"
            },
            "💰 Genre Economics": {
                "problem": "Which genres have the highest popularity ('Power Rankings')? What is their 'Market Share' (volume of tracks)?",
                "cols": "genres, popularity",
                "gb": "gframe_filtered.groupby('genres')"
            },
            "⏱️ Tempo Zones": {
                "problem": "Is there an optimal BPM for hit songs? This analyzes popularity by 'BPM Zones' (Slow, Dance, Fast) and how those preferences have evolved.",
                "cols": "tempo, popularity, decade",
                "gb": "df_tempo.groupby('bpm_zone'), df_tempo.groupby(['decade', 'bpm_zone'])"
            },
            "🌟 Popularity Lifecycle": {
                "problem": "How has average popularity changed over 100 years (the 'Streaming Effect')? What audio features do 'timeless' songs have in common?",
                "cols": "year, popularity, era, energy, danceability, valence, acousticness",
                "gb": "df_filtered.groupby('year'), df_eras.groupby('era'), df_timeless.groupby('is_timeless')"
            },
            "🚀 Artist Evolution": {
                "problem": "Which artists dominated each era? This analyzes artist performance over time, career longevity, and 'Rising Stars' with high momentum.",
                "cols": "artists, year, decade, popularity, name",
                "gb": "df_artist_time.groupby(['time_period', 'artist_clean']), df_dominance.groupby('artist_clean'), df_longevity.groupby('artist_clean')"
            },
            "💬 Title Analytics": {
                "problem": "Do song titles affect popularity? This analyzes title length, most common words in hits, and the impact of patterns (like 'feat.', '()', or 'ALL CAPS').",
                "cols": "name, popularity",
                "gb": "df_titles.groupby(word_bins), df_titles.groupby('has_feat')"
            },
            "🤝 Collaboration Patterns": {
                "problem": "Do collaborations (songs with >1 artist) actually perform better? What is the optimal 'team size' for a hit song?",
                "cols": "artists, artist_count, popularity, year",
                "gb": "df_collab.groupby('is_collab'), df_collab.groupby('year'), df_collab.groupby('artist_count')"
            }
        }
        
        # --- CREATE 3-COLUMN LAYOUT ---
        cols = st.columns(3)
        
        for i, (viz_name, details) in enumerate(viz_toc.items()):
            with cols[i % 3]:
                with st.expander(f"**{i+1}. {viz_name}**", expanded=True):
                    
                    # Use st.html for the new info box
                    st.html(TOC_INFO_BOX.format(problem_text=details["problem"]))
                    
                    # This markdown remains left-aligned by default
                    st.markdown(TOC_DETAILS_MD.format(
                        columns=details["cols"],
                        groupby=details["gb"]
                    ))
                    
                    # Center the button
                    st.markdown("<br>", unsafe_allow_html=True) 
                    _, col_btn_mid, _ = st.columns([1, 2, 1])
                    with col_btn_mid:
                        st.button(
                            "Go to ➔", 
                            key=f"toc_{i+1}", 
                            on_click=set_viz, 
                            args=(viz_name,),
                            use_container_width=True
                        )

    # --------------------------------------------------------
    # NEW: ACTIVE DATASET HEADER (with corrected font sizes)
    # --------------------------------------------------------
    def render_active_dataset_header():
        """
        Displays a dynamic, horizontal header with custom HTML/CSS
        to match the sidebar style.
        """
        try:
            if df_filtered is None or len(df_filtered) == 0 or len(df_filtered) == len(music_data['main']):
                df_active = music_data['main']
                percentage = 100.0
            else:
                df_active = df_filtered
                total_tracks = len(music_data['main'])
                percentage = (len(df_active) / total_tracks) * 100 if total_tracks > 0 else 0
            
            active_tracks = len(df_active)
            year_range = f"{df_active['year'].min()} - {df_active['year'].max()}"
            artist_count = df_active['artists'].nunique()
            avg_pop = df_active['popularity'].mean()
            explicit_count = df_active['explicit'].sum()

            # --- NEW: Calculate the percentage features ---
            avg_dance_pct = df_active['danceability'].mean() * 100
            avg_energy_pct = df_active['energy'].mean() * 100

        except Exception as e:
            st.error(f"Error calculating header stats: {e}")
            return

        # --- Build the HTML string ---
        header_html = f"""
        <div style="background: linear-gradient(135deg, #1a1a1a, #2a1a1a); border: 2px solid #1DB954; border-radius: 12px; padding: 15px; margin-bottom: 20px;">
            <h4 style="color:#1DB954; margin:0 0 12px 0; text-align:left; font-size: 18px;">📊 Active Dataset</h4>
            <div style="display:grid; grid-template-columns: 2.5fr 1.5fr 1fr 1fr 1fr 1fr; gap:10px;">
                
                <div style="text-align:center; padding:10px; background:#0a0a0a; border-radius:8px;">
                    <div style="color:#1DB954; font-size:24px; font-weight:bold;">{active_tracks:,}</div>
                    <div style="color:#888; font-size:12px;">tracks selected ({percentage:.0f}%)</div>
                </div>
                
                <div style="background:#0a0a0a; padding: 12px 8px; border-radius:6px; text-align:center;">
                    <div style="color:#1DB954; font-weight:bold; font-size: 24px;">{year_range}</div>
                    <div style="color:#888; font-size:12px; margin-top: 5px;">Years</div>
                </div>
                
                <div style="background:#0a0a0a; padding: 12px 8px; border-radius:6px; text-align:center;">
                    <div style="color:#1DB954; font-weight:bold; font-size: 24px;">{artist_count:,}</div>
                    <div style="color:#888; font-size:12px; margin-top: 5px;">Artists</div>
                </div>
                
                <div style="background:#0a0a0a; padding: 12px 8px; border-radius:6px; text-align:center;">
                    <div style="color:#1DB954; font-weight:bold; font-size: 24px;">{avg_pop:.0f}</div>
                    <div style="color:#888; font-size:12px; margin-top: 5px;">Avg Pop</div>
                </div>

                <div style="background:#0a0a0a; padding: 12px 8px; border-radius:6px; text-align:center;">
                    <div style="color:#1DB954; font-weight:bold; font-size: 24px;">{avg_dance_pct:.0f}%</div>
                    <div style="color:#888; font-size:12px; margin-top: 5px;">Avg Dance</div>
                </div>
                
                <div style="background:#0a0a0a; padding: 12px 8px; border-radius:6px; text-align:center;">
                    <div style="color:#1DB954; font-weight:bold; font-size: 24px;">{avg_energy_pct:.0f}%</div>
                    <div style="color:#888; font-size:12px; margin-top: 5px;">Avg Energy</div>
                </div>
                
            </div>
        </div>
        """
        
        st.html(header_html)

    # --------------------------------------------------------
    # DASHBOARD SUMMARY CARDS (Corrected with 8 safe try/except blocks)
    # --------------------------------------------------------
    def render_dashboard_summary():
        """
        Renders the 8-card (2x4 grid) summary for the
        main dashboard, using custom HTML with colors AND consistent fonts.
        """
        st.subheader("📈 Dashboard Summary (Filtered)")

        # 1. Define the HTML template for a single metric box.
        METRIC_BOX_HTML = """
        <div style="
            background: {bg_color}; 
            border: 1px solid {border_color};
            padding: 12px 8px; 
            border-radius: 6px; 
            text-align: center;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="color:#FFFFFF; font-weight:bold; font-size: 24px; line-height: 1.2; margin-bottom: 5px;">
                {value}
            </div>
            <div style="color:#E0E0E0; font-size:12px;">
                {label}
            </div>
        </div>
        """
        
        # 2. Define our color palettes
        COLORS = {
            "info": {"bg": "rgba(0, 104, 225, 0.1)", "border": "rgba(0, 104, 225, 0.5)"},
            "success": {"bg": "rgba(9, 171, 59, 0.1)", "border": "rgba(9, 171, 59, 0.5)"},
            "warning": {"bg": "rgba(255, 193, 7, 0.1)", "border": "rgba(255, 193, 7, 0.5)"},
            "error": {"bg": "rgba(255, 4, 4, 0.1)", "border": "rgba(255, 4, 4, 0.5)"}
        }
        CARD_COLORS = [
            COLORS["info"], COLORS["success"], COLORS["warning"], COLORS["error"],
            COLORS["error"], COLORS["warning"], COLORS["success"], COLORS["info"]
        ]
        
        # 3. Define labels
        CARD_LABELS = [
            "🔥 Trending Feature", "📊 10-Year Growth", "🎵 Dominant Mode", "⏱️ Avg Duration",
            "🔞 Explicit Content", "🎤 Top Artist (Avg Pop)", "🎹 Top Key", "🎧 Overall Vibe"
        ]
        
        # 4. Set default values for all 8 stats
        stats_values = ["N/A"] * 8
        
        # 5. Run each calculation in its own safe try/except block
        if not df_filtered.empty:
            df_year_f = aggregate_by_year(df_filtered)
            df_summary = df_filtered.copy()

            # --- Card 1: Trending Feature ---
            try:
                if not df_year_f.empty:
                    stats_values[0] = df_year_f.iloc[-1][['energy', 'danceability', 'valence']].idxmax().capitalize()
            except:
                pass # Fails silently, leaves "N/A"

            # --- Card 2: 10-Year Growth ---
            try:
                if len(df_year_f) >= 10:
                    current_pop = df_year_f.iloc[-1]['popularity']
                    past_pop = df_year_f.iloc[-10]['popularity']
                    if past_pop > 0:
                        growth_rate = ((current_pop - past_pop) / past_pop) * 100
                        stats_values[1] = f"{growth_rate:.1f}%"
                    else:
                        stats_values[1] = "Inf"
                else:
                    stats_values[1] = "N/A (low data)"
            except:
                pass

            # --- Card 3: Dominant Mode ---
            try:
                stats_values[2] = "Major" if df_summary['mode'].mean() > 0.5 else "Minor"
            except:
                pass
                
            # --- Card 4: Avg Duration ---
            try:
                avg_duration = df_summary['duration_ms'].mean() / 60000
                stats_values[3] = f"{avg_duration:.1f} min"
            except:
                pass

            # --- Card 5: Explicit Content % ---
            try:
                explicit_pct = df_summary['explicit'].mean() * 100
                stats_values[4] = f"{explicit_pct:.1f}%"
            except:
                pass
                
            # --- Card 6: Top Artist ---
            try:
                df_summary['artist_clean'] = df_summary['artists'].str.replace(r"[\[\]'\"]", "", regex=True).str.split(',').str[0].str.strip()
                if not df_summary.groupby('artist_clean')['popularity'].mean().empty:
                    stats_values[5] = df_summary.groupby('artist_clean')['popularity'].mean().idxmax()
            except:
                pass # This was the most likely error source
                
            # --- Card 7: Top Musical Key ---
            try:
                top_key_index = df_summary['key'].mode()[0]
                stats_values[6] = KEY_MAP.get(top_key_index, "N/A")
            except:
                pass

            # --- Card 8: Overall Vibe ---
            try:
                vibe_val_num = df_summary['valence'].mean()
                vibe_en_num = df_summary['energy'].mean()
                vibe_mood = "Happy" if vibe_val_num > 0.5 else "Sad"
                vibe_energy = "Energetic" if vibe_en_num > 0.6 else "Mellow"
                stats_values[7] = f"{vibe_energy} & {vibe_mood}"
            except:
                pass
        
        # 6. Render the 8-card grid
        cols_row1 = st.columns(4)
        cols_row2 = st.columns(4)
        
        for i in range(8):
            col = cols_row1[i] if i < 4 else cols_row2[i-4]
            with col:
                st.html(METRIC_BOX_HTML.format(
                    value=str(stats_values[i])[:20], # Truncate long values like artist names
                    label=CARD_LABELS[i],
                    bg_color=CARD_COLORS[i]["bg"],
                    border_color=CARD_COLORS[i]["border"]
                ))

        
    # ========================================================
    #
    # VISUALIZATION FUNCTIONS
    #
    # ========================================================

    # --------------------------------------------------------
    # VIZ 1: EVOLUTION (Corrected: No arguments)
    # --------------------------------------------------------
    def render_evolution(df_filtered: pd.DataFrame):
        st.subheader("1. Evolution of ALL Audio Features Over Decades")

        # --- FIX: Get data inside the function ---
        # This function can "see" the 'df_filtered' and 'aggregate_by_year'
        # variables from your main script.
        try:
            df_year_f = aggregate_by_year(df_filtered) 
        except Exception as e:
            st.error(f"Error during aggregation: {e}")
            return

        # --- FIX: Add empty data check ---
        if df_year_f.empty:
            st.warning("No data available for the selected filters.")
            return

        features = ['danceability','energy','valence','acousticness','instrumentalness','speechiness','liveness','loudness']
        selected = st.multiselect("Select Audio Features:", features, default=features, key="evo_features")
        normalize = st.checkbox("Normalize", value=False, key="evo_normalize")
        compare = st.checkbox("Compare two decades", value=False, key="evo_compare")

        # --- FIX: Add check for empty selection ---
        if not selected:
            st.info("Please select at least one feature to display.")
            return

        base = df_year_f[["year"] + selected].copy()
        if normalize:
            for c in selected:
                rng = base[c].max() - base[c].min()
                base[c] = 0 if rng == 0 else (base[c] - base[c].min()) / rng

        if compare:
            decades = sorted(df_filtered["decade"].unique().tolist())
            if len(decades) < 2:
                st.info("Not enough decades in the current selection to compare.")
                return
            d1 = st.selectbox("First decade", decades, index=max(0, len(decades)-2), key="evo_d1")
            d2 = st.selectbox("Second decade", decades, index=max(0, len(decades)-1), key="evo_d2")

            summary = (df_filtered[df_filtered["decade"].isin([d1, d2])]
                    .assign(group=lambda x: x["decade"].astype(str))
                    .groupby(["group","year"], as_index=False)[selected].mean())
            melted = summary.melt(id_vars=["group","year"], var_name="Feature", value_name="Value")
            fig = px.line(melted, x="year", y="Value", color="Feature", line_dash="group",
                        title="Evolution (comparison by decade)", template="plotly_dark")
        else:
            melted = base.melt(id_vars=["year"], var_name="Feature", value_name="Value")
            fig = px.line(melted, x="year", y="Value", color="Feature",
                        title="Evolution of Audio Features (1920-2020)", template="plotly_dark")
            
        if 'loudness' not in selected:
            fig.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

                
    # --------------------------------------------------------
    # VIZ 2: CORRELATION (Corrected: No arguments)
    # --------------------------------------------------------
    def render_correlation(df_filtered: pd.DataFrame):
        st.subheader("2. Correlation Analysis: Popularity vs Audio Features")
        st.info("**🔍 Key Questions:** Which audio feature best predicts commercial success? ...")
        
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
        
        col_viz2_1, col_viz2_2, col_viz2_3 = st.columns(3)
        
        with col_viz2_1:
            selected_feature = st.selectbox(
                "Select Audio Feature:",
                ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'speechiness'],
                key="feature_selector"
            )
        
        with col_viz2_2:
            viz2_type = st.radio(
                "Visualization Type:",
                ["Scatter with Trend", "Hexbin Density", "Box Plot by Bins"],
                key="viz2_type"
            )
        
        with col_viz2_3:
            add_percentiles = st.checkbox("Show percentile lines", value=False, key="percentiles_viz2")
        
        # --- FIX: Get data inside the function ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
            
        df_sample = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)
        
        if viz2_type == "Scatter with Trend":
            fig_correlation = px.scatter(
                df_sample, x=selected_feature, y='popularity',
                title=f'Popularity vs {selected_feature.capitalize()}',
                opacity=0.6, trendline='ols',
                labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'}
            )
            if add_percentiles:
                p25 = df_sample[selected_feature].quantile(0.25)
                p75 = df_sample[selected_feature].quantile(0.75)
                fig_correlation.add_vline(x=p25, line_dash="dash", line_color="red", opacity=0.5)
                fig_correlation.add_vline(x=p75, line_dash="dash", line_color="green", opacity=0.5)
                
        elif viz2_type == "Hexbin Density":
            fig_correlation = px.density_heatmap(
                df_sample, x=selected_feature, y='popularity',
                title=f'Density: Popularity vs {selected_feature.capitalize()}',
                labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'},
                nbinsx=30, nbinsy=20
            )
        else:  # Box Plot by Bins
            df_sample['feature_bin'] = pd.cut(df_sample[selected_feature], bins=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            fig_correlation = px.box(
                df_sample, x='feature_bin', y='popularity',
                title=f'Popularity Distribution by {selected_feature.capitalize()} Levels',
                labels={'feature_bin': f'{selected_feature.capitalize()} Level', 'popularity': 'Popularity Score'}
            )
        
        features_to_format_as_percent = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness'
        ]
        
        # Check if the feature we plotted on the x-axis should be a %
        if selected_feature in features_to_format_as_percent:
            if viz2_type != "Box Plot by Bins":
                fig_correlation.update_layout(xaxis_tickformat=".0%")
                
        st.plotly_chart(fig_correlation, use_container_width=True)
        
        correlation = df_filtered[selected_feature].corr(df_filtered['popularity'])
        st.metric(f"Correlation Coefficient", f"{correlation:.3f}", 
                delta=f"{'Positive' if correlation > 0 else 'Negative'} correlation")
        
    # --------------------------------------------------------
    # VIZ 3: GENRE DNA (Corrected)
    # --------------------------------------------------------
    # We remove 'df_filtered' from the arguments, because this function
    # will now get the filtered data itself, just like you wanted.
    def render_genre_dna(df_filtered: pd.DataFrame):
        st.subheader("3. Genre DNA: Audio Feature Signatures")
        st.info("**🔍 Strategic Questions:** What is the unique 'DNA' of each genre? ...")

        # --- THIS IS THE FIX ---
        # 1. We get the DYNAMICALLY FILTERED data using your helper functions.
        #    This assumes 'df_genres' and 'filters' are defined in the global scope
        #    before this function is called (which they are in your app).
        try:
        # Use global helpers, but ensure df_filtered is present
            gframe_filtered = align_genre_frame(music_data["with_genres"], filters)
            df_genre_agg = aggregate_by_genre(gframe_filtered)
        except Exception as e:
            st.error(f"Error aggregating genre data: {e}")
            return

        # 2. Check if the filters returned any data
        if df_genre_agg.empty:
            st.warning("No genre data available for the selected filters.")
            return
        # --- END OF FIX ---

        col_genre1, col_genre2 = st.columns(2)
        
        with col_genre1:
            genre_feature = st.selectbox(
                "Select Audio Feature:",
                ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 
                'speechiness', 'liveness', 'loudness', 'tempo'],
                key="genre_feature_selector"
            )
        
        with col_genre2:
            genre_viz_type = st.radio(
                "Visualization:",
                ["Box Plot", "Radar Chart"],
                horizontal=True,
                key="genre_viz_type"
            )
        
        if genre_viz_type == "Box Plot":
            # Get top 15 genres *from the filtered data*
            top_genres_filtered = df_genre_agg.nlargest(15, 'popularity')
            
            fig_genre = px.box(
                top_genres_filtered, # Use the filtered data
                x='genres',
                y=genre_feature,
                color='genres',
                title=f'{genre_feature.capitalize()} Distribution Across Top 15 Genres (Filtered)',
                labels={'genres': 'Genre', genre_feature: f'{genre_feature.capitalize()}'}
            )
            fig_genre.update_layout(showlegend=False, xaxis_tickangle=-45)
            
        else:  # Radar Chart
            import plotly.graph_objects as go
            
            # Get top 8 genres *from the filtered data*
            top_8_genres_filtered = df_genre_agg.nlargest(8, 'popularity')
            features_for_radar = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
            
            fig_genre = go.Figure()
            
            for _, genre_row in top_8_genres_filtered.iterrows(): # Use the filtered data
                values = [genre_row[feat] for feat in features_for_radar]
                fig_genre.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features_for_radar,
                    fill='toself',
                    name=genre_row['genres'][:20]
                ))
            
            fig_genre.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Genre DNA: Audio Feature Signatures (Filtered)"
            )
        
        features_to_format_as_percent = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness'
        ]
        
        if genre_viz_type == "Box Plot":
            if genre_feature in features_to_format_as_percent:
                fig_genre.update_layout(yaxis_tickformat=".0%")
        else: # Radar Chart
            # The features are hardcoded as features_for_radar, which are all 0-1
            fig_genre.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"))
            )
            
        st.plotly_chart(fig_genre, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 4: EXPLICIT STRATEGY (Corrected: No arguments, local copy)
    # --------------------------------------------------------
    def render_explicit(df_filtered: pd.DataFrame):
        st.subheader("4. Explicit Content: Commercial Strategy Analysis")
        st.info("**🔍 Strategic Questions:** Is explicit content a successful commercial strategy? Which genres benefit most from explicit content? Should artists release both clean and explicit versions?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        # --- FIX: Create a local copy to avoid mutation errors ---
        df_explicit = df_filtered.copy()
        df_explicit['explicit_label'] = df_explicit['explicit'].map({0: 'Clean', 1: 'Explicit'})
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Popularity comparison
            fig_explicit = px.violin(
                df_explicit[df_explicit['popularity'] > 0], # Use local copy
                x='explicit_label',
                y='popularity',
                color='explicit_label',
                title='Popularity Distribution: Clean vs Explicit',
                labels={'explicit_label': 'Content Type', 'popularity': 'Popularity Score'}
            )
            st.plotly_chart(fig_explicit, use_container_width=True)
        
        with col_exp2:
            # Feature comparison
            explicit_feature = st.selectbox(
                "Compare by feature:",
                ['energy', 'danceability', 'valence', 'speechiness'],
                key="explicit_feature"
            )
            
            fig_explicit_feat = px.box(
                df_explicit, # Use local copy
                x='explicit_label',
                y=explicit_feature,
                color='explicit_label',
                title=f'{explicit_feature.capitalize()} by Content Type',
                labels={'explicit_label': 'Content Type', explicit_feature: explicit_feature.capitalize()}
            )

            features_to_format_as_percent = [
                'danceability', 'energy', 'valence', 'speechiness'
            ]
            if explicit_feature in features_to_format_as_percent:
                fig_explicit_feat.update_layout(yaxis_tickformat=".0%")
                
            st.plotly_chart(fig_explicit_feat, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 5: FEATURE RELATIONSHIPS (Corrected with % formatting)
    # --------------------------------------------------------
    def render_feature_relationships(df_filtered: pd.DataFrame):
        st.subheader("5. Feature Relationships: The Music Formula")
        st.info("**🔍 Strategic Questions:** Which features must be balanced together? Can we identify 'formula' combinations for different popularity levels? What trade-offs do producers make?")
        
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
            
        analysis_type = st.radio(
            "Analysis Type:",
            ["Correlation Matrix", "Feature Pairs Analysis", "Success Formula"],
            horizontal=True,
            key="analysis_type_viz5"
        )
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'popularity']
        
        # --- DEFINE YOUR FORMATTING LIST HERE ---
        features_to_format_as_percent = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'liveness', 'speechiness'
        ]

        if analysis_type == "Correlation Matrix":
            correlation_matrix = df_filtered[audio_features].corr()
            
            fig_heatmap = px.imshow(
                correlation_matrix,
                title='Correlation Between Audio Features',
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        elif analysis_type == "Feature Pairs Analysis":
            correlation_matrix = df_filtered[audio_features].corr()
            pop_corr = correlation_matrix['popularity'].drop('popularity').sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Correlations with Popularity:**")
                # --- FIX: We can format the text output right here ---
                for feat, corr in pop_corr.head(3).items():
                    if feat in features_to_format_as_percent:
                        st.write(f"• {feat}: {corr:.1%}")
                    else:
                        st.write(f"• {feat}: {corr:.3f}")
            
            with col2:
                st.markdown("**Top Negative Correlations with Popularity:**")
                for feat, corr in pop_corr.tail(3).items():
                    if feat in features_to_format_as_percent:
                        st.write(f"• {feat}: {corr:.1%}")
                    else:
                        st.write(f"• {feat}: {corr:.3f}")
                
        else:  # Success Formula
            df_success = df_filtered.copy()
            
            df_success['success_level'] = pd.cut(df_success['popularity'], 
                                                bins=[0, 30, 60, 100],
                                                labels=['Low', 'Medium', 'High'])
            
            # Slicing [:-1] to remove 'popularity'
            success_formula = df_success.groupby('success_level', observed=True)[audio_features[:-1]].mean()
            
            fig_formula = px.bar(
                success_formula.T,
                title="The Success Formula: Average Features by Popularity Level",
                labels={'index': 'Audio Feature', 'value': 'Average Value'},
                barmode='group'
            )
            
            # --- THIS IS THE FIX ---
            # The y-axis shows the feature means, which are 0-1
            # We apply the percentage format here
            fig_formula.update_layout(yaxis_tickformat=".0%")
            # --- END OF FIX ---

            st.plotly_chart(fig_formula, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 6: TEMPORAL TRENDS (Corrected: No arguments, local copy)
    # --------------------------------------------------------
    def render_temporal(df_filtered: pd.DataFrame):
        st.subheader("6. Temporal Trends: Predicting the Future of Music")
        st.info("**🔍 Strategic Questions:** Can we predict the next trend? Are songs converging to a standard formula? What features are becoming obsolete?")
        
        # --- FIX: Get data inside the function and make a local copy ---
        try:
            df_year_f = aggregate_by_year(df_filtered).copy()
        except Exception as e:
            st.error(f"Error during aggregation: {e}")
            return

        if df_year_f.empty:
            st.warning("No data available for the selected filters.")
            return

        # --- FIX: Check for empty data ---
        if df_year_f.empty:
            st.warning("No data available for the selected filters.")
            return
            
        col_tempo1, col_tempo2 = st.columns(2)
        
        # Convert duration to minutes
        df_year_f["duration_min"] = df_year_f["duration_ms"] / 60000.0

        with col_tempo1:
            # Add trend prediction toggle
            show_prediction = st.checkbox("Show trend projection", value=False, key="prediction_duration")
            
            fig_duration = px.line(
                df_year_f[df_year_f["year"] >= 1960],
                x="year", y="duration_min",
                title="Song Duration: The Attention Span Crisis",
                labels={"duration_min": "Duration (minutes)", "year": "Year"}
            )
            
            if show_prediction:
                # Simple linear projection
                try:
                    from scipy import stats # Import is safe inside the try block
                    
                    # Use the already calculated df_year_f
                    recent_years_agg = df_year_f[df_year_f['year'] >= 2000]
                    
                    # Check if we have valid data
                    if len(recent_years_agg) > 1 and not recent_years_agg['duration_min'].isna().all():
                        slope, intercept, _, _, _ = stats.linregress(recent_years_agg['year'], recent_years_agg['duration_min'])
                        future_years = list(range(2020, 2031))
                        future_duration = [slope * year + intercept for year in future_years]
                        fig_duration.add_scatter(x=future_years, y=future_duration, mode='lines', 
                                                name='Projection', line=dict(dash='dash', color='red'))
                except ImportError:
                    st.warning("Scipy module not found. Skipping trend projection.")
                except Exception as e:
                    st.warning(f"Could not calculate trend: {e}")
            
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col_tempo2:
            fig_tempo = px.line(
                df_year_f[df_year_f['year'] >= 1960],
                x='year',
                y='tempo',
                title='Tempo Evolution: The BPM Arms Race',
                labels={'tempo': 'Tempo (BPM)', 'year': 'Year'}
            )
            st.plotly_chart(fig_tempo, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 7: ARTIST SUCCESS PATTERNS (Corrected: No arguments)
    # --------------------------------------------------------
    def render_artists(df_filtered: pd.DataFrame):
        st.subheader("7. Artist Success Patterns: One-Hit Wonder vs Consistency")
        st.info("**🔍 Strategic Questions:** What separates consistent hitmakers from one-hit wonders? Is it better to release many average songs or few excellent ones? How do successful artists maintain their sound signature?")
        
        # --- FIX: Get DYNAMICALLY FILTERED artist data first ---
        try:
            df_artist_f = aggregate_by_artist(df_filtered)
        except Exception as e:
            st.error(f"Error during artist aggregation: {e}")
            return

        if df_artist_f.empty:
            st.warning("No artist data available for the selected filters.")
            return
        # --- END OF FIX ---

        artist_strategy = st.radio(
            "Analysis Focus:",
            ["Top 50 Artists Overview", "Consistency Analysis", "Feature Signature (Top 5)"],
            horizontal=True,
            key="artist_strategy"
        )
        
        # --- FIX: Use the filtered artist dataframe ---
        top_artists_filtered = df_artist_f.nlargest(50, 'popularity')
        
        if artist_strategy == "Top 50 Artists Overview":
            fig_artists = px.scatter(
                top_artists_filtered, # Use filtered data
                x='count',
                y='popularity',
                size='energy',
                color='valence',
                hover_data=['artist_clean'], # Use the clean artist name
                title='Artist Strategy: Volume vs Quality (Filtered)',
                labels={'count': 'Number of Tracks', 'popularity': 'Average Popularity', 'valence': 'Valence'}
            )
            
            # Add quadrant lines based on filtered data
            median_count = top_artists_filtered['count'].median()
            median_pop = top_artists_filtered['popularity'].median()
            fig_artists.add_hline(y=median_pop, line_dash="dash", line_color="gray", opacity=0.5)
            fig_artists.add_vline(x=median_count, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant labels
            fig_artists.add_annotation(x=median_count*0.3, y=median_pop*1.2, text="Quality over Quantity", 
                                    showarrow=False, font=dict(color="green"))
            fig_artists.add_annotation(x=median_count*1.7, y=median_pop*1.2, text="Consistent Hitmakers", 
                                    showarrow=False, font=dict(color="#89ccff"))
            
            st.plotly_chart(fig_artists, use_container_width=True)

        elif artist_strategy == "Consistency Analysis":
            # Calculate coefficient of variation (std/mean) for each artist
            artist_consistency = []
            
            # --- FIX: Loop over filtered artists and search in df_filtered ---
            for artist_name in top_artists_filtered['artist_clean']:
                # Search for this artist's songs *within the filtered tracklist*
                # Note: str.contains is not perfect, but it's what your original code used.
                artist_songs = df_filtered[df_filtered['artists'].str.contains(artist_name, na=False, case=True)]['popularity']
                
                if len(artist_songs) > 1:
                    cv = artist_songs.std() / artist_songs.mean() if artist_songs.mean() > 0 else 0
                    artist_consistency.append({'artist': artist_name[:20], 'consistency': 1 - cv, 
                                            'avg_popularity': artist_songs.mean()})
            
            consistency_df = pd.DataFrame(artist_consistency)
            
            if consistency_df.empty:
                st.info("Could not calculate artist consistency. (No artist matches with >1 song found in the filtered data).")
                fig_artists = go.Figure() # Create empty figure
                fig_artists.update_layout(title='Artist Consistency Score (No data to display)')
            else:
                fig_artists = px.bar(
                    consistency_df.sort_values('consistency', ascending=True),
                    x='consistency',
                    y='artist',
                    orientation='h',
                    color='avg_popularity',
                    title='Artist Consistency Score (Higher = More Consistent)',
                    labels={'consistency': 'Consistency Score', 'artist': 'Artist', 
                            'avg_popularity': 'Avg Popularity'}
                )
            
            st.plotly_chart(fig_artists, use_container_width=True)
                
        else:  # Feature Signature
            # --- FIX: Use the filtered top 5 artists ---
            top_5_artists_filtered = top_artists_filtered.head(5)
            features_for_signature = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
            
            fig_artists = go.Figure()
            
            for _, artist_row in top_5_artists_filtered.iterrows(): # Use filtered data
                values = [artist_row[feat] for feat in features_for_signature]
                fig_artists.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features_for_signature,
                    fill='toself',
                    name=artist_row['artist_clean'][:20] # Use clean name
                ))
            
            fig_artists.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
                showlegend=True,
                title="Artist Sound Signatures: What Makes Them Unique"
            )

            st.plotly_chart(fig_artists, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 8: FEATURE EXPLORER (Corrected: No arguments)
    # --------------------------------------------------------
    def render_explorer(df_filtered: pd.DataFrame):
        st.subheader("8. Feature Interaction Explorer: Finding the Sweet Spot")
        st.info("**🔍 Strategic Questions:** What feature combinations create viral hits? Are there 'dead zones' to avoid? How do different eras prefer different feature combinations?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
            
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            feature_x = st.selectbox(
                "X-axis Feature:",
                ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 
                'speechiness', 'liveness', 'loudness', 'tempo', 'popularity'],
                index=0,
                key="density_x_feature"
            )
        
        with col_exp2:
            feature_y = st.selectbox(
                "Y-axis Feature:",
                ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 
                'speechiness', 'liveness', 'loudness', 'tempo', 'popularity'],
                index=1, # Set a different default index from X
                key="density_y_feature"
            )
        
        with col_exp3:
            viz_style = st.radio(
                "Style:",
                ["Density Heatmap", "Scatter with Size", "Contour Plot"],
                key="viz_style_8"
            )
        
        features_to_format_as_percent = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness'
        ]

        # This filter logic is excellent:
        # Options are from the full 'df' (good!)
        # Data is filtered from 'df_filtered' (also good!)
        decade_filter_8 = st.select_slider(
            "Filter by Decade Range (for this viz):",
            options=sorted(df['decade'].unique()), # Uses original 'df'
            value=(1990, 2020),
            key="decade_filter_8"
        )
        
        # Filter data
        df_density = df_filtered[(df_filtered['decade'] >= decade_filter_8[0]) & 
                                (df_filtered['decade'] <= decade_filter_8[1])]
        
        # Check if the *local* filter returned data
        if df_density.empty:
            st.warning("No data available for the selected *local decade filter*.")
            return

        df_density_sample = df_density.sample(min(10000, len(df_density)), random_state=42)
        
        if viz_style == "Density Heatmap":
            fig_density = px.density_heatmap(
                df_density_sample,
                x=feature_x,
                y=feature_y,
                title=f'Feature Sweet Spots: {feature_x.capitalize()} vs {feature_y.capitalize()}',
                labels={feature_x: feature_x.capitalize(), feature_y: feature_y.capitalize()},
                nbinsx=30,
                nbinsy=30,
                color_continuous_scale='Viridis'
            )
            
        elif viz_style == "Scatter with Size":
            fig_density = px.scatter(
                df_density_sample.sample(min(2000, len(df_density_sample))),
                x=feature_x,
                y=feature_y,
                size='popularity',
                color='decade',
                title=f'Feature Combinations: Size = Popularity',
                labels={feature_x: feature_x.capitalize(), feature_y: feature_y.capitalize()},
                opacity=0.6
            )
            
        else:  # Contour Plot
            fig_density = px.density_contour(
                df_density_sample,
                x=feature_x,
                y=feature_y,
                title=f'Density Contours: {feature_x.capitalize()} vs {feature_y.capitalize()}',
                labels={feature_x: feature_x.capitalize(), feature_y: feature_y.capitalize()}
            )
            fig_density.update_traces(contours_coloring="fill", contours_showlabels=True)
        
        x_format = ".0%" if feature_x in features_to_format_as_percent else ""
        y_format = ".0%" if feature_y in features_to_format_as_percent else ""
        fig_density.update_layout(xaxis_tickformat=x_format, yaxis_tickformat=y_format)
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Show insights
        high_pop_threshold = df_density['popularity'].quantile(0.75)
        sweet_spot_df = df_density[(df_density['popularity'] > high_pop_threshold)]
        
        if len(sweet_spot_df) > 0:
            col_insight1, col_insight2 = st.columns(2)
            with col_insight1:
                # --- THIS IS THE FIX (Part 3: The Metrics) ---
                val_x = sweet_spot_df[feature_x].mean()
                delta_x = sweet_spot_df[feature_x].std()
                if feature_x in features_to_format_as_percent:
                    st.metric(f"Sweet Spot {feature_x.capitalize()}", 
                            f"{val_x:.1%}", f"±{delta_x:.1%}")
                else:
                    st.metric(f"Sweet Spot {feature_x.capitalize()}", 
                            f"{val_x:.2f}", f"±{delta_x:.2f}")
            with col_insight2:
                val_y = sweet_spot_df[feature_y].mean()
                delta_y = sweet_spot_df[feature_y].std()
                if feature_y in features_to_format_as_percent:
                    st.metric(f"Sweet Spot {feature_y.capitalize()}", 
                            f"{val_y:.1%}", f"±{delta_y:.1%}")
                else:
                    st.metric(f"Sweet Spot {feature_y.capitalize()}", 
                            f"{val_y:.2f}", f"±{delta_y:.2f}")

    # --------------------------------------------------------
    # VIZ 9: KEY & MODE (Corrected: No arguments, local copy)
    # --------------------------------------------------------
    def render_keys(df_filtered: pd.DataFrame):
        st.subheader("9. Musical Key & Mode: The Emotional Blueprint")
        st.info("**🔍 Strategic Questions:** Do certain keys resonate better with audiences? Should artists favor major (happy) or minor (sad) keys? Is there a 'golden key' for hits?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        # --- FIX: Create a local copy to avoid mutation errors ---
        df_keys = df_filtered.copy()
        
        # Map keys to musical notation
        key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
        
        # Add new columns to the local copy
        df_keys['key_name'] = df_keys['key'].map(key_mapping)
        df_keys['mode_name'] = df_keys['mode'].map({0: 'Minor', 1: 'Major'})
        
        key_analysis = st.radio(
            "Analysis Type:",
            ["Distribution", "Popularity by Key", "Emotional Impact"],
            horizontal=True,
            key="key_analysis"
        )
        
        if key_analysis == "Distribution":
            # Count combinations using the local copy
            key_mode_counts = df_keys.groupby(['key_name', 'mode_name']).size().reset_index(name='count')
            
            fig_keys = px.bar(
                key_mode_counts,
                x='key_name',
                y='count',
                color='mode_name',
                title='Distribution of Musical Keys (Major vs Minor)',
                labels={'key_name': 'Musical Key', 'count': 'Number of Songs', 'mode_name': 'Mode'},
                category_orders={'key_name': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']}
            )
            
        elif key_analysis == "Popularity by Key":
            # Average popularity by key using the local copy
            key_popularity = df_keys.groupby(['key_name', 'mode_name'])['popularity'].mean().reset_index()
            
            fig_keys = px.bar(
                key_popularity,
                x='key_name',
                y='popularity',
                color='mode_name',
                title='Average Popularity by Musical Key',
                labels={'key_name': 'Musical Key', 'popularity': 'Average Popularity', 'mode_name': 'Mode'},
                category_orders={'key_name': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']},
                barmode='group'
            )
            
        else:  # Emotional Impact
            # Compare valence (happiness) using the local copy
            fig_keys = px.box(
                df_keys,
                x='mode_name',
                y='valence',
                color='mode_name',
                title='Emotional Impact: Valence in Major vs Minor Keys',
                labels={'mode_name': 'Mode', 'valence': 'Valence (Happiness)'}
            )

            fig_keys.update_layout(yaxis_tickformat=".0%")
        
        st.plotly_chart(fig_keys, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 10: DECADE EVOLUTION (Corrected: No arguments, safe checks)
    # --------------------------------------------------------
    def render_decades(df_filtered: pd.DataFrame):
        st.subheader("10. Decade Evolution: The Sound of Generations")
        st.info("**🔍 Strategic Questions:** How homogeneous is modern music? Which decades had the most experimental music? Can we identify the 'signature sound' of each generation?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
            
        col_decade1, col_decade2 = st.columns(2)
        
        with col_decade1:
            decade_feature = st.selectbox(
                "Select Feature:",
                ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 
                'speechiness', 'liveness', 'loudness'],
                key="decade_feature_selector"
            )
        
        with col_decade2:
            show_variance = st.checkbox("Show variance analysis", value=False, key="variance_decade")
        
        # Filter for relevant decades (this .copy() is perfect)
        df_decades = df_filtered['decade'].copy()
        
        # --- FIX: Add second check for locally filtered data ---
        if df_decades.empty:
            st.warning("No data available for decades in your current selection.")
            return

        if not show_variance:
            fig_decade_box = px.box(
                df_decades,
                x='decade',
                y=decade_feature,
                color='decade',
                title=f'{decade_feature.capitalize()} Evolution by Decade',
                labels={'decade': 'Decade', decade_feature: f'{decade_feature.capitalize()}'}
            )
            fig_decade_box.update_layout(showlegend=False)
            
        else:
            # Calculate variance by decade
            variance_by_decade = df_decades.groupby('decade')[decade_feature].agg(['mean', 'std']).reset_index()
            
            # --- FIX: Prevent ZeroDivisionError ---
            variance_by_decade['cv'] = variance_by_decade.apply(
                lambda row: row['std'] / row['mean'] if row['mean'] != 0 else 0, axis=1
            )
            
            fig_decade_box = px.line(
                variance_by_decade,
                x='decade',
                y='cv',
                title=f'Musical Diversity: {decade_feature.capitalize()} Variance Over Time',
                labels={'decade': 'Decade', 'cv': 'Coefficient of Variation'},
                markers=True
            )
            fig_decade_box.add_hline(y=variance_by_decade['cv'].mean(), 
                                line_dash="dash", line_color="red", opacity=0.5)
        
        # Define our percentage list
        features_to_format_as_percent = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness'
        ]
        
        if not show_variance and decade_feature in features_to_format_as_percent:
            # Format the main box plot
            fig_decade_box.update_layout(yaxis_tickformat=".0%")
        elif show_variance:
            # Format the variance line plot (CV is a ratio)
            fig_decade_box.update_layout(yaxis_tickformat=".1%")
            
        st.plotly_chart(fig_decade_box, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 11: GENRE ECONOMICS (Corrected: No arguments, fully dynamic)
    # --------------------------------------------------------
    def render_genre_econ(df_filtered: pd.DataFrame):
        st.subheader("11. Genre Economics: Market Share & Commercial Viability")
        st.info("**🔍 Strategic Questions:** Which genres dominate the market? Are niche genres more loyal? Where should new artists position themselves for maximum impact?")
        
        # --- FIX: Get DYNAMICALLY FILTERED data ONCE at the top ---
        try:
            # Use global helpers, but rely on df_filtered to derive data
            gframe_filtered = align_genre_frame(music_data["with_genres"], filters)
            df_genres_f = aggregate_by_genre(gframe_filtered) 
        except Exception as e:
            st.error(f"Error aggregating genre data: {e}")
            return

        if gframe_filtered.empty or df_genres_f.empty:
            st.warning("No genre data available for the selected filters.")
            return
            
        genre_view = st.radio(
            "View:",
            ["Top Genres by Popularity", "Genre Market Share", "Genre Loyalty Index"],
            horizontal=True,
            key="genre_view"
        )
        
        if genre_view == "Top Genres by Popularity":
            # This was already correct, just uses our new variable
            top_genres_pop = df_genres_f.nlargest(20, "popularity")[["genres","popularity"]]
            
            fig_top_genres = px.bar(
                top_genres_pop,
                x='popularity',
                y='genres',
                orientation='h',
                color='popularity',
                color_continuous_scale='Viridis',
                title='Genre Power Rankings: Commercial Appeal (Filtered)',
                labels={'genres': 'Genre', 'popularity': 'Average Popularity'}
            )
            fig_top_genres.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            
        elif genre_view == "Genre Market Share":
            # --- FIX: Use the filtered gframe_filtered ---
            genre_counts = gframe_filtered['genres'].value_counts().head(15)
            
            fig_top_genres = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title='Genre Market Share by Track Volume (Filtered)'
            )
            
        else:  # Genre Loyalty Index
            genre_loyalty = []
            
            # --- FIX 1: Iterate over the filtered genre list ---
            for genre in df_genres_f.nlargest(15, 'popularity')['genres']:
                
                # --- FIX 2: Search for songs in the filtered track list ---
                genre_data = gframe_filtered[gframe_filtered['genres'] == genre]
                
                if len(genre_data) > 10:
                    loyalty = 1 / (genre_data['popularity'].std() + 1)  # +1 to avoid division by zero
                    genre_loyalty.append({'genre': genre[:20], 'loyalty_index': loyalty * 100,
                                        'avg_popularity': genre_data['popularity'].mean()})
            
            if not genre_loyalty:
                st.info("Not enough data to calculate genre loyalty for the current filter.")
                fig_top_genres = go.Figure().update_layout(title='Genre Loyalty vs Popularity: Finding Your Niche')
            else:
                loyalty_df = pd.DataFrame(genre_loyalty).sort_values('loyalty_index', ascending=True)
                
                fig_top_genres = px.scatter(
                    loyalty_df,
                    x='avg_popularity',
                    y='loyalty_index',
                    text='genre',
                    title='Genre Loyalty vs Popularity: Finding Your Niche (Filtered)',
                    labels={'loyalty_index': 'Loyalty Index', 'avg_popularity': 'Average Popularity'}
                )
                fig_top_genres.update_traces(textposition='top center')
            
        st.plotly_chart(fig_top_genres, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 12: TEMPO ZONES (Corrected: No arguments, safe checks)
    # --------------------------------------------------------
    def render_tempo(df_filtered: pd.DataFrame):
        st.subheader("12. The Tempo Formula: BPM Success Zones")
        st.info("**🔍 Strategic Questions:** Is there an optimal BPM for chart success? Do different decades prefer different tempos? Should producers target specific BPM ranges?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        # --- This .copy() is perfect! ---
        df_tempo = df_filtered.copy()
        
        # --- FIX: Use np.inf for the last bin to include all tempos ---
        df_tempo['bpm_zone'] = pd.cut(df_tempo['tempo'], 
                                    bins=[0, 80, 100, 120, 140, np.inf],
                                    labels=['Slow (<80)', 'Moderate (80-100)', 
                                            'Dance (100-120)', 'Fast (120-140)', 'Very Fast (>140)'])
        
        tempo_analysis = st.radio(
            "Analysis:",
            ["Density Map", "Success Zones", "Evolution"],
            horizontal=True,
            key="tempo_analysis"
        )
        
        if tempo_analysis == "Density Map":
            df_tempo_sample = df_tempo.sample(min(10000, len(df_tempo)))
            
            if df_tempo_sample.empty:
                st.warning("No data to display for Density Map.")
                return

            fig_tempo_density = px.density_heatmap(
                df_tempo_sample,
                x='tempo',
                y='popularity',
                title='Tempo-Popularity Heat Map: Where Success Lives',
                labels={'tempo': 'Tempo (BPM)', 'popularity': 'Popularity Score'},
                nbinsx=40,
                nbinsy=30
            )
            
        elif tempo_analysis == "Success Zones":
            bpm_success = df_tempo.groupby('bpm_zone', observed=True)['popularity'].agg(['mean', 'std', 'count']).reset_index()
            
            if bpm_success.empty:
                st.warning("No data to display for Success Zones.")
                return

            fig_tempo_density = px.bar(
                bpm_success,
                x='bpm_zone',
                y='mean',
                error_y='std',
                title='Popularity by BPM Zone',
                labels={'bpm_zone': 'BPM Zone', 'mean': 'Average Popularity'},
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            
        else:  # Evolution
            tempo_evolution = df_tempo.groupby(['decade', 'bpm_zone'], observed=True).size().reset_index(name='count')
            tempo_evolution = tempo_evolution[tempo_evolution['decade'] >= 1960]
            
            if tempo_evolution.empty:
                st.warning("No data to display for Tempo Evolution.")
                return

            fig_tempo_density = px.bar(
                tempo_evolution,
                x='decade',
                y='count',
                color='bpm_zone',
                title='Evolution of Tempo Preferences Over Decades',
                labels={'decade': 'Decade', 'count': 'Number of Tracks'},
                barnorm='percent'
            )
        
        st.plotly_chart(fig_tempo_density, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 13: EXPLICIT OVER TIME (Corrected: No arguments, fully dynamic)
    # --------------------------------------------------------
    def render_explicit_time(df_filtered: pd.DataFrame):
        st.subheader("13. Explicit Content Evolution: Cultural Shifts & Commercial Impact")
        st.info("**🔍 Strategic Questions:** When did explicit content become mainstream? Which genres pioneered explicit content? Is the trend reversing or accelerating?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return
            
        explicit_view = st.radio(
            "View:",
            ["Timeline", "By Genre", "Commercial Impact"],
            horizontal=True,
            key="explicit_view"
        )
        
        if explicit_view == "Timeline":
            # This was already correct, as it uses df_filtered.
            explicit_years = (df_filtered[df_filtered["year"] >= 1960]
                            .groupby(["year","explicit"])
                            .size().reset_index(name="count"))
            
            if explicit_years.empty:
                st.warning("No data for timeline view.")
                return

            explicit_years["explicit_label"] = explicit_years["explicit"].map({0:"Clean",1:"Explicit"})
            
            fig_explicit_years = px.area(
                explicit_years,
                x='year',
                y='count',
                color='explicit_label',
                title='The Rise of Explicit Content (1960-2020)',
                labels={'year': 'Year', 'count': 'Number of Tracks', 'explicit_label': 'Content Type'},
                color_discrete_map={'Clean': '#2E7D32', 'Explicit': '#D32F2F'}
            )
            
        elif explicit_view == "By Genre":
            # --- FIX: Use helper functions to get filtered genre data ---
            try:
                gframe_filtered = align_genre_frame(music_data["with_genres"], filters)
                df_genres_f = aggregate_by_genre(gframe_filtered)
            except Exception as e:
                st.error(f"Error aggregating genre data: {e}")
                return
            
            if gframe_filtered.empty or df_genres_f.empty:
                st.warning("No genre data available for this filter to analyze 'By Genre'.")
                return
                
            # --- FIX: Get top genres from *filtered* data ---
            top_genres_list = df_genres_f.nlargest(10, 'popularity')['genres'].tolist()
            
            explicit_by_genre = []
            for genre in top_genres_list:
                # --- FIX: Search in the *filtered* track list ---
                genre_data = gframe_filtered[gframe_filtered['genres'] == genre]
                if len(genre_data) > 0:
                    # Ensure 'explicit' column exists, default to 0 (Clean) if not
                    explicit_pct = (genre_data.get('explicit', 0).sum() / len(genre_data)) * 100
                    explicit_by_genre.append({'genre': genre[:20], 'explicit_percentage': explicit_pct})
            
            if not explicit_by_genre:
                st.info("No explicit content found in the top genres for this filter.")
                fig_explicit_years = go.Figure().update_layout(title='Explicit Content by Genre (%)')
            else:
                explicit_genre_df = pd.DataFrame(explicit_by_genre).sort_values('explicit_percentage')
                
                fig_explicit_years = px.bar(
                    explicit_genre_df,
                    x='explicit_percentage',
                    y='genre',
                    orientation='h',
                    title='Explicit Content by Genre (%) (Filtered)',
                    labels={'genre': 'Genre', 'explicit_percentage': 'Explicit Content (%)'},
                    color='explicit_percentage',
                    color_continuous_scale='Reds'
                )
            
        else:  # Commercial Impact
            # --- FIX: Use df_filtered, not df. Make a copy. ---
            df_impact = df_filtered[df_filtered['year'] >= 1980].copy()
            
            if df_impact.empty:
                st.warning("No data from 1980 onwards to calculate commercial impact.")
                return

            explicit_impact = df_impact.groupby(['year', 'explicit'])['popularity'].mean().reset_index()
            explicit_impact['explicit_label'] = explicit_impact['explicit'].map({0: 'Clean', 1: 'Explicit'})
            
            fig_explicit_years = px.line(
                explicit_impact,
                x='year',
                y='popularity',
                color='explicit_label',
                title='Commercial Performance: Clean vs Explicit Over Time (Filtered)',
                labels={'year': 'Year', 'popularity': 'Average Popularity', 'explicit_label': 'Content Type'},
                markers=True
            )
        
        st.plotly_chart(fig_explicit_years, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 14: POPULARITY LIFECYCLE (Corrected: No arguments, fully dynamic)
    # --------------------------------------------------------
    def render_pop_lifecycle(df_filtered: pd.DataFrame):
        st.subheader("14. The Popularity Lifecycle: Understanding Music Relevance")
        st.info("**🔍 Strategic Questions:** How long do songs stay relevant? Are older songs experiencing a streaming renaissance? What makes a song 'timeless'?")
        
        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        popularity_view = st.radio(
            "Analysis:",
            ["Overall Trend", "By Era", "Timeless Features"],
            horizontal=True,
            key="popularity_view"
        )
        
        if popularity_view == "Overall Trend":
            # --- FIX: Get aggregated data inside the function ---
            try:
                # Assumes 'aggregate_by_year' and 'df_filtered' are in global scope
                popularity_trend = aggregate_by_year(df_filtered)
                popularity_trend = popularity_trend[popularity_trend['year'] >= 1920][['year', 'popularity']]
            except Exception as e:
                st.error(f"Error during aggregation: {e}")
                return

            if popularity_trend.empty:
                st.warning("No popularity trend data for this filter.")
                return
                
            fig_pop_trend = px.line(
                popularity_trend,
                x='year',
                y='popularity',
                title='The Streaming Effect: Popularity Trend (Filtered)',
                labels={'year': 'Year', 'popularity': 'Average Popularity'},
                markers=True
            )
            
            # Add streaming era marker
            fig_pop_trend.add_vline(x=2006, line_dash="dash", line_color="green", opacity=0.5)
            fig_pop_trend.add_annotation(x=2006, y=popularity_trend['popularity'].max()*0.9, 
                                        text="Spotify Launch", showarrow=True)
            
            # Add trend line
            fig_pop_trend.add_scatter(
                x=popularity_trend['year'],
                y=popularity_trend['popularity'].rolling(window=10, center=True).mean(),
                mode='lines',
                name='10-Year Moving Average',
                line=dict(color='red', dash='dash')
            )
            
        elif popularity_view == "By Era":
            # --- FIX: Use df_filtered.copy() ---
            df_eras = df_filtered.copy()
            df_eras['era'] = pd.cut(df_eras['year'], 
                                    bins=[0, 1960, 1980, 1990, 2000, 2010, 2025],
                                    labels=['Pre-1960', '1960s-70s', '1980s', '1990s', '2000s', '2010s+'])
            
            # Use observed=True to ignore empty bins
            era_popularity = df_eras.groupby('era', observed=True)['popularity'].agg(['mean', 'std']).reset_index()
            
            if era_popularity.empty:
                st.warning("Not enough data to group by era.")
                return

            fig_pop_trend = px.bar(
                era_popularity,
                x='era',
                y='mean',
                error_y='std',
                title='Popularity by Musical Era (Filtered)',
                labels={'era': 'Era', 'mean': 'Average Popularity'},
                color='mean',
                color_continuous_scale='Blues'
            )
            
        else:  # Timeless Features
            # --- FIX: Use df_filtered, not df ---
            timeless_threshold = df_filtered['popularity'].quantile(0.7)
            df_timeless = df_filtered.copy()
            df_timeless['is_timeless'] = df_timeless['popularity'] > timeless_threshold
            
            features_compare = ['energy', 'danceability', 'valence', 'acousticness']
            
            # Use observed=True to handle cases where one group might be empty
            timeless_features = df_timeless.groupby('is_timeless', observed=True)[features_compare].mean().T
            
            # Check if both columns (True/False) exist
            if True not in timeless_features.columns or False not in timeless_features.columns:
                st.info("Not enough data to compare 'Timeless' vs 'Regular' songs with current filters.")
                fig_pop_trend = go.Figure().update_layout(title='The DNA of Timeless Songs (Filtered)')
            else:
                timeless_features.columns = ['Regular', 'Timeless']
                fig_pop_trend = px.bar(
                    timeless_features,
                    title='The DNA of Timeless Songs (Filtered)',
                    labels={'index': 'Feature', 'value': 'Average Value'},
                    barmode='group'
                )
            
            fig_pop_trend.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig_pop_trend, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 15: ARTIST EVOLUTION (Corrected: No arguments, empty check)
    # --------------------------------------------------------
    def render_artist_evo(df_filtered: pd.DataFrame):
        st.subheader("15. Artist Evolution: The Rise and Fall of Music Icons")
        st.info("**🔍 Strategic Questions:** Which artists dominated different eras? How has artist longevity changed? Can we identify 'comeback' artists or one-era wonders?")

        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        # Prepare the data
        artist_time_analysis = st.radio(
            "Analysis Type:",
            ["Top Artists Timeline", "Artist Dominance by Decade", "Longevity Analysis", "Rising Stars"],
            horizontal=True,
            key="artist_time_analysis"
        )

        if artist_time_analysis == "Top Artists Timeline":
            # Sub-options for timeline view
            col_art_time1, col_art_time2, col_art_time3 = st.columns(3)
            
            with col_art_time1:
                metric_choice = st.selectbox(
                    "Metric:",
                    ["Average Popularity", "Track Count", "Maximum Popularity"],
                    key="artist_timeline_metric"
                )
            
            with col_art_time2:
                time_granularity = st.selectbox(
                    "Time Period:",
                    ["By Year", "By Decade", "By 5-Year Period"],
                    key="time_granularity"
                )
            
            with col_art_time3:
                top_n_artists = st.slider(
                    "Number of Artists:",
                    min_value=5, max_value=30, value=15, step=5,
                    key="top_n_timeline"
                )
            
            # --- This is all correct: uses df_filtered and .copy() ---
            df_artist_time = df_filtered[df_filtered['year'] >= 1960].copy()
            
            if df_artist_time.empty:
                st.warning("No data from 1960 onwards for this analysis.")
                return

            # ... (rest of your "Top Artists Timeline" logic is perfect) ...
            # Determine time column based on granularity
            if time_granularity == "By Decade":
                df_artist_time['time_period'] = df_artist_time['decade']
                time_col = 'time_period'
            elif time_granularity == "By 5-Year Period":
                df_artist_time['time_period'] = (df_artist_time['year'] // 5) * 5
                time_col = 'time_period'
            else:  # By Year
                time_col = 'year'
                df_artist_time['time_period'] = df_artist_time['year']
            
            df_artist_time['artist_clean'] = df_artist_time['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            df_artist_time['artist_clean'] = df_artist_time['artist_clean'].str.split(',').str[0].str.strip()
            
            if metric_choice == "Average Popularity":
                artist_metrics = df_artist_time.groupby(['time_period', 'artist_clean'])['popularity'].mean().reset_index()
                metric_col = 'popularity'
                metric_label = 'Average Popularity'
            elif metric_choice == "Track Count":
                artist_metrics = df_artist_time.groupby(['time_period', 'artist_clean']).size().reset_index(name='track_count')
                metric_col = 'track_count'
                metric_label = 'Number of Tracks'
            else:  # Maximum Popularity
                artist_metrics = df_artist_time.groupby(['time_period', 'artist_clean'])['popularity'].max().reset_index()
                metric_col = 'popularity'
                metric_label = 'Maximum Popularity'
            
            top_artists_overall = artist_metrics.groupby('artist_clean')[metric_col].mean().nlargest(top_n_artists).index.tolist()
            artist_metrics_filtered = artist_metrics[artist_metrics['artist_clean'].isin(top_artists_overall)]
            
            if artist_metrics_filtered.empty:
                st.warning("No data found for the top artists in this time period.")
                return

            fig_artist_timeline = px.line(
                artist_metrics_filtered,
                x='time_period', y=metric_col, color='artist_clean',
                title=f'Top {top_n_artists} Artists: {metric_label} Over Time',
                labels={'time_period': time_granularity.replace('By ', ''), 
                        metric_col: metric_label, 'artist_clean': 'Artist'},
                markers=True, template='plotly_dark'
            )
            fig_artist_timeline.update_layout(
                plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3',
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
                height=600
            )
            fig_artist_timeline.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_artist_timeline, use_container_width=True)
            
            # ... (rest of your metric columns logic is also correct) ...
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                if artist_metrics_filtered.empty:
                    st.metric("Most Consistent Artist", "N/A", "No data")
                else:
                    counts_by_artist = artist_metrics_filtered.groupby('artist_clean').size()
                    if counts_by_artist.empty:
                        st.metric("Most Consistent Artist", "N/A", "No data")
                    else:
                        most_consistent = counts_by_artist.idxmax()
                        appearances = int(counts_by_artist.max())
                        st.metric("Most Consistent Artist", most_consistent[:25], f"{appearances} periods")
            with col_stat2:
                if artist_metrics_filtered.empty or artist_metrics_filtered[metric_col].dropna().empty:
                    st.metric(f"Peak {metric_label}", "N/A", "No data")
                else:
                    peak_row = artist_metrics_filtered.loc[artist_metrics_filtered[metric_col].idxmax()]
                    peak_artist = str(peak_row['artist_clean'])
                    peak_value = float(peak_row[metric_col])
                    st.metric(f"Peak {metric_label}", peak_artist[:25], f"{peak_value:.1f}")
            with col_stat3:
                if artist_metrics_filtered.empty or artist_metrics_filtered['time_period'].dropna().empty:
                    st.metric("Current Leader", "N/A", "")
                else:
                    recent_period = artist_metrics_filtered['time_period'].max()
                    recent_df = artist_metrics_filtered[artist_metrics_filtered['time_period'] == recent_period]
                    if recent_df.empty or recent_df[metric_col].dropna().empty:
                        st.metric("Current Leader", "N/A", "")
                    else:
                        recent_top = recent_df.nlargest(1, metric_col)['artist_clean'].values[0]
                        st.metric("Current Leader", str(recent_top)[:25], f"In {recent_period}")

        elif artist_time_analysis == "Artist Dominance by Decade":
            # --- This is all correct: uses 'df' for options, 'df_filtered' for data ---
            decade_selection = st.select_slider(
                "Select Decade Range:",
                options=sorted(df['decade'].unique()),
                value=(1970, 2020),
                key="dominance_decade_range"
            )
            
            df_dominance = df_filtered[(df_filtered['decade'] >= decade_selection[0]) & 
                                    (df_filtered['decade'] <= decade_selection[1])].copy()
            
            if df_dominance.empty:
                st.warning("No data available for the selected decade range.")
                return

            # ... (rest of your "Artist Dominance" logic is perfect) ...
            df_dominance['artist_clean'] = df_dominance['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            df_dominance['artist_clean'] = df_dominance['artist_clean'].str.split(',').str[0].str.strip()
            
            dominance_by_decade = []
            for decade in range(decade_selection[0], decade_selection[1] + 10, 10):
                decade_data = df_dominance[df_dominance['decade'] == decade]
                if len(decade_data) > 0:
                    artist_stats = decade_data.groupby('artist_clean').agg({
                        'popularity': ['mean', 'max', 'count']
                    }).round(2)
                    artist_stats.columns = ['avg_popularity', 'max_popularity', 'track_count']
                    artist_stats['dominance_score'] = (
                        artist_stats['avg_popularity'] * 0.5 + 
                        artist_stats['max_popularity'] * 0.3 + 
                        artist_stats['track_count'] * 0.2
                    )
                    top_decade_artists = artist_stats.nlargest(10, 'dominance_score')
                    for artist, row in top_decade_artists.iterrows():
                        dominance_by_decade.append({
                            'decade': decade, 'artist': artist[:30],
                            'dominance_score': row['dominance_score'],
                            'avg_popularity': row['avg_popularity'],
                            'track_count': row['track_count']
                        })
            
            dominance_df = pd.DataFrame(dominance_by_decade)
            
            if not dominance_df.empty:
                heatmap_data = dominance_df.pivot_table(
                    index='artist', columns='decade', values='dominance_score', fill_value=0
                )
                heatmap_data['total'] = heatmap_data.sum(axis=1)
                heatmap_data = heatmap_data.sort_values('total', ascending=False).drop('total', axis=1).head(20)
                
                fig_dominance = px.imshow(
                    heatmap_data,
                    title='Artist Dominance Heatmap by Decade',
                    labels=dict(x="Decade", y="Artist", color="Dominance Score"),
                    aspect="auto", color_continuous_scale="Viridis", template='plotly_dark'
                )
                fig_dominance.update_layout(
                    plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3', height=700
                )
                st.plotly_chart(fig_dominance, use_container_width=True)
                
                st.markdown("### 👑 Decade Leaders")
                decade_leaders = dominance_df.sort_values(['decade', 'dominance_score'], ascending=[True, False])
                decade_leaders = decade_leaders.groupby('decade').first().reset_index()
                
                cols = st.columns(len(decade_leaders))
                for idx, (_, leader) in enumerate(decade_leaders.iterrows()):
                    with cols[idx]:
                        st.metric(
                            f"{int(leader['decade'])}s",
                            leader['artist'][:20],
                            f"Score: {leader['dominance_score']:.1f}"
                        )
            else:
                st.info("No data available for the selected decade range.")

        elif artist_time_analysis == "Longevity Analysis":
            # --- This is all correct: uses df_filtered.copy() ---
            st.markdown("### 🏆 Artist Career Longevity & Consistency")
            min_tracks = st.slider(
                "Minimum tracks to be included:",
                min_value=5, max_value=50, value=10, step=5,
                key="longevity_min_tracks"
            )
            
            df_longevity = df_filtered.copy()
            
            if df_longevity.empty:
                st.warning("No data for longevity analysis.")
                return

            # ... (rest of your "Longevity Analysis" logic is perfect) ...
            df_longevity['artist_clean'] = df_longevity['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            df_longevity['artist_clean'] = df_longevity['artist_clean'].str.split(',').str[0].str.strip()
            
            longevity_stats = df_longevity.groupby('artist_clean').agg({
                'year': ['min', 'max', 'nunique'],
                'popularity': ['mean', 'std', 'max'],
                'name': 'count'
            }).round(2)
            longevity_stats.columns = ['first_year', 'last_year', 'active_years', 
                                    'avg_popularity', 'std_popularity', 'max_popularity', 'track_count']
            longevity_stats = longevity_stats[longevity_stats['track_count'] >= min_tracks]
            
            if longevity_stats.empty:
                st.warning(f"No artists found with at least {min_tracks} tracks in this filter.")
                return

            longevity_stats['career_span'] = longevity_stats['last_year'] - longevity_stats['first_year']
            longevity_stats['consistency_score'] = (longevity_stats['avg_popularity'] / 
                                                (longevity_stats['std_popularity'] + 1)) * (longevity_stats['active_years'] / longevity_stats['career_span'].clip(lower=1))
            longevity_stats = longevity_stats.reset_index()
            
            col_long1, col_long2 = st.columns(2)
            with col_long1:
                fig_career_span = px.scatter(
                    longevity_stats.nlargest(30, 'career_span'),
                    x='career_span', y='avg_popularity', size='track_count', color='consistency_score',
                    hover_data=['artist_clean', 'first_year', 'last_year'],
                    title='Longest Career Spans (Top 30)',
                    labels={'career_span': 'Career Span (Years)', 'avg_popularity': 'Average Popularity', 'consistency_score': 'Consistency'},
                    template='plotly_dark', color_continuous_scale='Viridis'
                )
                fig_career_span.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                st.plotly_chart(fig_career_span, use_container_width=True)
            
            with col_long2:
                fig_consistency = px.bar(
                    longevity_stats.nlargest(15, 'consistency_score'),
                    x='consistency_score', y='artist_clean', orientation='h', color='avg_popularity',
                    title='Most Consistent Artists (Top 15)',
                    labels={'consistency_score': 'Consistency Score', 'artist_clean': 'Artist', 'avg_popularity': 'Avg Popularity'},
                    template='plotly_dark', color_continuous_scale='Viridis'
                )
                fig_consistency.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3', yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_consistency, use_container_width=True)
            
            st.markdown("### 🎭 Artist Career Types")
            longevity_stats['career_type'] = 'Standard'
            longevity_stats.loc[(longevity_stats['career_span'] > 20) & (longevity_stats['avg_popularity'] > 50), 'career_type'] = 'Legend'
            longevity_stats.loc[(longevity_stats['career_span'] < 5) & (longevity_stats['max_popularity'] > 70), 'career_type'] = 'One-Hit Wonder'
            longevity_stats.loc[(longevity_stats['career_span'] > 10) & (longevity_stats['consistency_score'] > longevity_stats['consistency_score'].quantile(0.75)), 'career_type'] = 'Steady Performer'
            
            career_distribution = longevity_stats['career_type'].value_counts()
            cols = st.columns(min(4, len(career_distribution)))
            for idx, (career_type, count) in enumerate(career_distribution.items()):
                with cols[idx % 4]:
                    emoji = {'Legend': '👑', 'One-Hit Wonder': '🌟', 'Steady Performer': '📊', 'Standard': '🎵'}.get(career_type, '🎵')
                    st.metric(f"{emoji} {career_type}", f"{count} artists", f"{(count/len(longevity_stats)*100):.1f}%")

        else:  # Rising Stars
            # --- This is all correct: uses df_filtered.copy() ---
            st.markdown("### 🚀 Rising Stars & Trending Artists")
            window_years = st.slider(
                "Analysis window (recent years):",
                min_value=5, max_value=20, value=10, step=5,
                key="rising_window"
            )
            
            current_year = df_filtered['year'].max()
            cutoff_year = current_year - window_years
            
            df_recent = df_filtered[df_filtered['year'] > cutoff_year].copy()
            df_previous = df_filtered[(df_filtered['year'] <= cutoff_year) & 
                                    (df_filtered['year'] > cutoff_year - window_years)].copy()
            
            if df_recent.empty:
                st.warning("No data found in the recent analysis window.")
                return

            # ... (rest of your "Rising Stars" logic is perfect) ...
            for data in [df_recent, df_previous]:
                data['artist_clean'] = data['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                data['artist_clean'] = data['artist_clean'].str.split(',').str[0].str.strip()
            
            recent_stats = df_recent.groupby('artist_clean').agg({'popularity': 'mean', 'name': 'count'}).rename(columns={'popularity': 'recent_pop', 'name': 'recent_tracks'})
            previous_stats = df_previous.groupby('artist_clean').agg({'popularity': 'mean', 'name': 'count'}).rename(columns={'popularity': 'previous_pop', 'name': 'previous_tracks'})
            
            growth_analysis = pd.merge(recent_stats, previous_stats, left_index=True, right_index=True, how='left').fillna(0)
            growth_analysis['pop_growth'] = ((growth_analysis['recent_pop'] - growth_analysis['previous_pop']) / (growth_analysis['previous_pop'] + 1)) * 100
            growth_analysis['track_growth'] = ((growth_analysis['recent_tracks'] - growth_analysis['previous_tracks']) / (growth_analysis['previous_tracks'] + 1)) * 100
            growth_analysis['momentum_score'] = (growth_analysis['pop_growth'] * 0.6 + growth_analysis['track_growth'] * 0.4)
            growth_analysis = growth_analysis.reset_index()
            growth_analysis = growth_analysis[growth_analysis['recent_tracks'] >= 3]

            if growth_analysis.empty:
                st.warning("No artists with >= 3 recent tracks found for momentum analysis.")
                return

            col_rise1, col_rise2 = st.columns(2)
            with col_rise1:
                rising_stars = growth_analysis.nlargest(15, 'momentum_score')
                if not rising_stars.empty:
                    fig_rising = px.scatter(
                        rising_stars,
                        x='previous_pop', y='recent_pop', size='recent_tracks', color='momentum_score',
                        hover_data=['artist_clean'],
                        title='Rising Stars: Previous vs Current Popularity',
                        labels={'previous_pop': f'Popularity ({cutoff_year-window_years}-{cutoff_year})',
                                'recent_pop': f'Popularity ({cutoff_year}-{current_year})',
                                'momentum_score': 'Momentum'},
                        template='plotly_dark', color_continuous_scale='RdYlGn'
                    )
                    max_val = max(rising_stars['previous_pop'].max(), rising_stars['recent_pop'].max())
                    fig_rising.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color='gray', dash='dash', width=1))
                    fig_rising.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                    st.plotly_chart(fig_rising, use_container_width=True)
                else:
                    st.info("No rising stars found.")

            with col_rise2:
                new_artists = growth_analysis[growth_analysis['previous_tracks'] == 0].nlargest(10, 'recent_pop')
                if not new_artists.empty:
                    fig_new = px.bar(
                        new_artists, x='recent_pop', y='artist_clean', orientation='h', color='recent_tracks',
                        title='Breakthrough Artists (New Entrants)',
                        labels={'recent_pop': 'Current Popularity', 'artist_clean': 'Artist', 'recent_tracks': 'Track Count'},
                        template='plotly_dark', color_continuous_scale='Viridis'
                    )
                    fig_new.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3', yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_new, use_container_width=True)
                else:
                    st.info("No breakthrough artists found in the selected time window.")
            
            st.markdown("### 📊 Movement Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                biggest_gainer = growth_analysis.nlargest(1, 'pop_growth')
                if not biggest_gainer.empty:
                    st.metric("🎯 Biggest Popularity Gain", biggest_gainer['artist_clean'].values[0][:25], f"+{biggest_gainer['pop_growth'].values[0]:.1f}%")
            with col_sum2:
                most_productive = growth_analysis.nlargest(1, 'recent_tracks')
                if not most_productive.empty:
                    st.metric("🎵 Most Productive", most_productive['artist_clean'].values[0][:25], f"{most_productive['recent_tracks'].values[0]:.0f} tracks")
            with col_sum3:
                highest_momentum = growth_analysis.nlargest(1, 'momentum_score')
                if not highest_momentum.empty:
                    st.metric(
                        "🚀 Highest Momentum",
                        highest_momentum['artist_clean'].values[0][:25],
                        f"Score: {highest_momentum['momentum_score'].values[0]:.1f}"
                    )

    # --------------------------------------------------------
    # VIZ 16: TITLE ANALYTICS (Corrected: No arguments, safe checks)
    # --------------------------------------------------------
    def render_titles(df_filtered: pd.DataFrame):
        st.subheader("16. The Power of Words: Song Title Analysis")
        st.info("**🔍 Strategic Questions:** Do shorter titles perform better? What words appear most in hit songs? Can title sentiment predict success?")

        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        title_analysis_type = st.radio(
            "Analysis Type:",
            ["Title Length vs Popularity", "Most Common Words", "Title Patterns"],
            horizontal=True,
            key="title_analysis"
        )

        # --- This .copy() is perfect! ---
        df_titles = df_filtered.copy()
        
        # Check for NaNs in 'name' column, which can break string ops
        if df_titles['name'].isna().any():
            df_titles = df_titles.dropna(subset=['name'])

        df_titles['title_length'] = df_titles['name'].str.len()
        df_titles['title_word_count'] = df_titles['name'].str.split().str.len()

        if title_analysis_type == "Title Length vs Popularity":
            col_title1, col_title2 = st.columns(2)
            
            with col_title1:
                fig_title_len = px.scatter(
                    df_titles.sample(min(5000, len(df_titles))),
                    x='title_length',
                    y='popularity',
                    trendline='lowess',
                    title='Title Length (Characters) vs Popularity',
                    labels={'title_length': 'Title Length (chars)', 'popularity': 'Popularity'},
                    opacity=0.6,
                    template='plotly_dark'
                )
                fig_title_len.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                st.plotly_chart(fig_title_len, use_container_width=True)
            
            with col_title2:
                # --- FIX: Use np.inf for the last bin ---
                word_bins = pd.cut(df_titles['title_word_count'], 
                                bins=[0, 1, 2, 3, 4, np.inf], 
                                labels=['1 word', '2 words', '3 words', '4 words', '5+ words'])
                
                # --- FIX: Add observed=True ---
                word_popularity = df_titles.groupby(word_bins, observed=True)['popularity'].mean().reset_index()
                
                fig_word_count = px.bar(
                    word_popularity,
                    x='title_word_count',
                    y='popularity',
                    title='Average Popularity by Title Word Count',
                    labels={'title_word_count': 'Word Count', 'popularity': 'Average Popularity'},
                    color='popularity',
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
                fig_word_count.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                st.plotly_chart(fig_word_count, use_container_width=True)

        elif title_analysis_type == "Most Common Words":
            df_popular_titles = df_titles[df_titles['popularity'] > 50]
            
            if df_popular_titles.empty:
                st.warning("No songs with >50 popularity in this filter to analyze common words.")
                return

            all_words = ' '.join(df_popular_titles['name'].str.lower()).split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'been'}
            filtered_words = [re.sub(r'\W+', '', word) for word in all_words if word not in stop_words and len(word) > 2]
            
            word_freq = Counter(filtered_words)
            top_words = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
            
            if top_words.empty:
                st.info("No common words found.")
                return

            fig_words = px.bar(
                top_words,
                x='count',
                y='word',
                orientation='h',
                title='Most Common Words in Popular Songs (>50 popularity)',
                labels={'count': 'Frequency', 'word': 'Word'},
                color='count',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig_words.update_layout(
                plot_bgcolor='#181818', paper_bgcolor='#181818',
                font_color='#B3B3B3', height=600,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_words, use_container_width=True)

        else:  # Title Patterns
            df_titles['has_feat'] = df_titles['name'].str.contains('feat\.|ft\.|featuring', case=False, na=False)
            df_titles['has_parentheses'] = df_titles['name'].str.contains('\(|\)', na=False)
            df_titles['has_numbers'] = df_titles['name'].str.contains('\d', na=False)
            df_titles['is_uppercase'] = df_titles['name'].str.isupper()
            
            patterns = {
                'Features': df_titles.groupby('has_feat')['popularity'].mean(),
                'Parentheses': df_titles.groupby('has_parentheses')['popularity'].mean(),
                'Numbers': df_titles.groupby('has_numbers')['popularity'].mean(),
                'All Caps': df_titles.groupby('is_uppercase')['popularity'].mean()
            }
            
            pattern_results = []
            for pattern, data in patterns.items():
                # This 'if len(data) == 2' check is perfect!
                if len(data) == 2 and True in data.index and False in data.index:
                    pattern_results.append({
                        'Pattern': pattern,
                        'Without': data[False],
                        'With': data[True],
                        'Impact': data[True] - data[False]
                    })
            
            if not pattern_results:
                st.info("Not enough data to compare title patterns.")
                return

            pattern_df = pd.DataFrame(pattern_results)
            
            fig_patterns = px.bar(
                pattern_df,
                x=['Without', 'With'],
                y='Pattern',
                title='Impact of Title Patterns on Popularity',
                labels={'value': 'Average Popularity', 'variable': 'Pattern Type'},
                barmode='group',
                template='plotly_dark',
                color_discrete_sequence=['#B3B3B3', '#1DB954']
            )
            fig_patterns.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
            st.plotly_chart(fig_patterns, use_container_width=True)

    # --------------------------------------------------------
    # VIZ 17: COLLABORATION PATTERNS (Corrected: No arguments)
    # --------------------------------------------------------
    def render_collab(df_filtered: pd.DataFrame):
        st.subheader("17. Collaboration Economy: The Power of Features")
        st.info("**🔍 Strategic Questions:** Do collaborations boost popularity? Which artists collaborate most? What's the optimal number of artists per track?")

        # --- FIX: Check for empty data first ---
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
            return

        # --- This .copy() is perfect! ---
        df_collab = df_filtered.copy()
        
        # Check for NaNs in 'artists' column
        if df_collab['artists'].isna().any():
            df_collab = df_collab.dropna(subset=['artists'])
            
        df_collab['artist_count'] = df_collab['artists'].str.count(',') + 1

        collab_view = st.radio(
            "View:",
            ["Collaboration Impact", "Artist Networks", "Optimal Team Size"],
            horizontal=True,
            key="collab_view"
        )

        if collab_view == "Collaboration Impact":
            df_collab['is_collab'] = df_collab['artist_count'] > 1
            
            col_collab1, col_collab2 = st.columns(2)
            
            with col_collab1:
                collab_stats = df_collab.groupby('is_collab', observed=True)['popularity'].agg(['mean', 'std', 'count']).reset_index()
                collab_stats['is_collab'] = collab_stats['is_collab'].map({False: 'Solo', True: 'Collaboration'})
                
                if collab_stats.empty:
                    st.warning("No data for Solo vs. Collab comparison.")
                else:
                    fig_collab_impact = px.bar(
                        collab_stats,
                        x='is_collab', y='mean', error_y='std',
                        title='Solo vs Collaboration Performance',
                        labels={'is_collab': 'Type', 'mean': 'Average Popularity'},
                        color='mean', color_continuous_scale='Viridis',
                        template='plotly_dark'
                    )
                    fig_collab_impact.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                    st.plotly_chart(fig_collab_impact, use_container_width=True)
            
            with col_collab2:
                collab_trend = df_collab[df_collab['year'] >= 1980].groupby('year')['is_collab'].mean() * 100
                
                if collab_trend.empty:
                    st.warning("No data for Collaboration Trend timeline.")
                else:
                    fig_collab_trend = px.line(
                        x=collab_trend.index, y=collab_trend.values,
                        title='Collaboration Trend Over Time',
                        labels={'x': 'Year', 'y': 'Collaboration %'},
                        markers=True, template='plotly_dark'
                    )
                    fig_collab_trend.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
                    st.plotly_chart(fig_collab_trend, use_container_width=True)

        elif collab_view == "Artist Networks":
            collab_only = df_collab[df_collab['artist_count'] > 1].copy()
            
            if len(collab_only) > 0:
                artist_collabs = []
                # Sample for performance
                for artists_str in collab_only['artists'].head(1000): 
                    artists_list = [a.strip() for a in artists_str.replace('[','').replace(']','').replace("'",'').split(',')]
                    for artist in artists_list:
                        artist_collabs.append(artist)
                
                if not artist_collabs:
                    st.info("No collaboration artists found in the sample.")
                    return

                collab_counts = Counter(artist_collabs)
                top_collabs = pd.DataFrame(collab_counts.most_common(15), columns=['artist', 'collaborations'])
                
                fig_network = px.bar(
                    top_collabs,
                    x='collaborations', y='artist', orientation='h',
                    title='Most Collaborative Artists',
                    labels={'collaborations': 'Number of Collaborations', 'artist': 'Artist'},
                    color='collaborations', color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
                fig_network.update_layout(
                    plot_bgcolor='#181818', paper_bgcolor='#181818',
                    font_color='#B3B3B3', height=500,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("No collaboration tracks found in the current filter.")

        else:  # Optimal Team Size
            team_size = df_collab.groupby('artist_count')['popularity'].agg(['mean', 'count']).reset_index()
            team_size = team_size[team_size['artist_count'] <= 5]  # Focus on reasonable team sizes
            
            if team_size.empty:
                st.warning("No data available for team size analysis.")
                return

            fig_team = px.scatter(
                team_size,
                x='artist_count', y='mean', size='count',
                title='Optimal Team Size for Hit Songs',
                labels={'artist_count': 'Number of Artists', 'mean': 'Average Popularity', 'count': 'Sample Size'},
                template='plotly_dark', color='mean',
                color_continuous_scale='RdYlGn'
            )
            fig_team.update_layout(plot_bgcolor='#181818', paper_bgcolor='#181818', font_color='#B3B3B3')
            st.plotly_chart(fig_team, use_container_width=True)
            
            # --- FIX: Check if team_size has data before finding idxmax ---
            if not team_size.empty:
                optimal = team_size.loc[team_size['mean'].idxmax()]
                st.success(f"**Optimal Team Size:** {int(optimal['artist_count'])} artist(s) with average popularity of {optimal['mean']:.1f}")

if music_data is not None:
    render_active_dataset_header()

    st.sidebar.markdown("---")
    
    df_to_show = df_filtered if 'df_filtered' in locals() else None
    
    # Pass the *whole* music_data dict
    display_sidebar_stats(
        music_data=music_data,
        df_filtered=df_to_show
    )
    
    st.sidebar.markdown("---")
        

    # --------------------------------------------------------
    # TAB 1: DASHBOARD
    # --------------------------------------------------------
    if st.session_state.current_tab == "Dashboard":
        
        # --- 1. Define the viz_map with LAMBDA functions ---
        # The lambda waits to pass the required df_filtered until the function is called.
        viz_map = {
            "🏠 Welcome & Table of Contents": render_welcome_page,
            "📈 Evolution of Features": lambda: render_evolution(df_filtered),
            "📊 Popularity vs Features": lambda: render_correlation(df_filtered),
            "🎸 Genre DNA": lambda: render_genre_dna(df_filtered),
            "🔞 Explicit Strategy": lambda: render_explicit(df_filtered),
            "📈 Explicit Over Time": lambda: render_explicit_time(df_filtered),
            "🔗 Feature Relationships": lambda: render_feature_relationships(df_filtered),
            "🔗 Temporal Trends": lambda: render_temporal(df_filtered),
            "👤 Artist Success Patterns": lambda: render_artists(df_filtered),
            "🔍 Feature Explorer": lambda: render_explorer(df_filtered),
            "🎵 Key & Mode": lambda: render_keys(df_filtered),
            "📅 Decade Evolution": lambda: render_decades(df_filtered),
            "💰 Genre Economics": lambda: render_genre_econ(df_filtered),
            "⏱️ Tempo Zones": lambda: render_tempo(df_filtered),
            "🌟 Popularity Lifecycle": lambda: render_pop_lifecycle(df_filtered),
            "🚀 Artist Evolution": lambda: render_artist_evo(df_filtered),
            "💬 Title Analytics": lambda: render_titles(df_filtered),
            "🤝 Collaboration Patterns": lambda: render_collab(df_filtered),
        }

        # --- 2. Show the selectbox ---
        selected_viz_name = st.selectbox(
            "Choose a visualization to load:", 
            list(viz_map.keys()), 
            key="viz_picker"
        )
        
        # --- 3. Show the Dashboard Summary ---
        render_dashboard_summary() 

        # --- 4. Get and call the selected render function ---
        render_function = viz_map.get(selected_viz_name)
        
        if render_function:
            # The function is called here! The lambda passes df_filtered automatically.
            render_function() 
        else:
            st.error(f"Error: Could not find function for '{selected_viz_name}'.")
        
        st.divider()
        
        # ---------------------------------------------------
        # PERFORMANCE MODE (Now much cleaner)
        # ---------------------------------------------------
        
        performance_mode = st.checkbox("⚡ Show all visualizations (Performance Mode)", value=False, key="perf_mode", 
                                        help="Load all visualizations at once in tabs. May be slow.")
        
        if performance_mode:            
            st.markdown("""
            <style>
                /* Visual indicator for scrollable tabs */
                .stTabs [data-baseweb="tab-list"] {
                    background: linear-gradient(90deg, 
                        #181818 0%, 
                        transparent 5%, 
                        transparent 95%, 
                        #181818 100%);
                    padding: 0 20px;
                }
                
                /* Pulsing indicator */
                .scroll-hint {
                    display: inline-block;
                    margin-left: 10px;
                    animation: blink 2s infinite;
                }
                
                @keyframes blink {
                    0%, 100% { opacity: 0.3; }
                    50% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True)
            
            viz_tabs = st.tabs([
                "📈 Evolution", "📊 Correlations", "🎸 Genres", "🔞 Explicit",
                "📈 Explicit OT", "🔗 Temporal", "👤 Artists", "🔍 Explorer",
                "🎵 Keys", "📅 Decades", "💰 Economics", "⏱️ Tempo",
                "🌟 Popularity", "🚀 Artist Evo", "💬 Titles", "🤝 Collabs"
                # Note: You have 17 functions, but 1 is the Welcome Page.
                # You'll have 16 tabs for graphs. I've shortened some names.
            ])

            with viz_tabs[0]:
                render_evolution(df_filtered)
            with viz_tabs[1]:
                render_correlation(df_filtered)
            with viz_tabs[2]:
                render_genre_dna(df_filtered)
            with viz_tabs[3]:
                render_explicit(df_filtered)
            with viz_tabs[4]:
                render_explicit_time(df_filtered)
            with viz_tabs[5]:
                render_feature_relationships(df_filtered)
            with viz_tabs[6]:
                render_temporal(df_filtered)
            with viz_tabs[7]:
                render_artists(df_filtered)
            with viz_tabs[8]:
                render_explorer(df_filtered)
            with viz_tabs[9]:
                render_keys(df_filtered)
            with viz_tabs[10]:
                render_decades(df_filtered)
            with viz_tabs[11]:
                render_genre_econ(df_filtered)
            with viz_tabs[12]:
                render_tempo(df_filtered)
            with viz_tabs[13]:
                render_pop_lifecycle(df_filtered)
            with viz_tabs[14]:
                render_artist_evo(df_filtered)
            with viz_tabs[15]:
                render_titles(df_filtered)
            with viz_tabs[16]:
                render_collab(df_filtered)


    # --------------------------------------------------------
    # TAB 2: INSIGHTS
    # --------------------------------------------------------
    elif st.session_state.current_tab == "Insights":
        st.header("💡 Key Insights & Deep Analysis")
        st.markdown("Strategic insights and decade-by-decade analysis of music trends.")
        
        # ---------------------------------------------------
        # KEY INSIGHTS SUMMARY
        # ---------------------------------------------------
        st.subheader("📊 Key Insights Summary")
        st.markdown("Metrics below are dynamically calculated based on your filter selections.")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            # Metric 1: Most Popular Decade (Dynamic)
            st.metric("Most Popular Decade", 
                        f"{df_filtered.groupby('decade')['popularity'].mean().idxmax()}s",
                        f"Avg Pop: {df_filtered.groupby('decade')['popularity'].mean().max():.1f}")
        
        with col_summary2:
            # Metric 2: Top Track (Dynamic)
            if not df_filtered.empty:
                top_track = df_filtered.loc[df_filtered['popularity'].idxmax()]
                cleaned_artists = top_track['artists'].replace('[','').replace(']','').replace("'",'').replace('"', '')
                st.metric("Top Track in Selection",
                            top_track['name'][:20],
                            f"{cleaned_artists.split(',')[0].strip()} - Pop: {top_track['popularity']:.1f}")
            else:
                st.metric("Top Track in Selection", "N/A", "No data")

        with col_summary3:
            # Metric 3: Explicit Content % (Dynamic)
            if not df_filtered.empty:
                explicit_pct = (df_filtered['explicit'].sum() / len(df_filtered)) * 100
                st.metric("Explicit Content",
                            f"{explicit_pct:.1f}%",
                            f"{df_filtered['explicit'].sum()} explicit tracks")
            else:
                st.metric("Explicit Content", "N/A", "No data")
        
        col_summary4, col_summary5, col_summary6 = st.columns(3)
        
        with col_summary4:
            # Metric 4: Highest Rated Genre (Static - from df_genres)
            top_genre = df_genres.nlargest(1, 'popularity')['genres'].values[0] if len(df_genres) > 0 else "N/A"
            st.metric("Highest Rated Genre (All Time)", 
                        top_genre[:20],
                        f"Avg Pop: {df_genres.nlargest(1, 'popularity')['popularity'].values[0]:.1f}")
        
        with col_summary5:
            # Metric 5: Top Popularity Driver (Dynamic)
            correlation_with_pop = df_filtered[['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']].corrwith(df_filtered['popularity'])
            best_feature = correlation_with_pop.abs().idxmax()
            st.metric("Top Popularity Driver", 
                        best_feature.capitalize(),
                        f"Corr: {correlation_with_pop[best_feature]:.3f}")
        
        with col_summary6:
            # Metric 6: Average Mood (Dynamic)
            if not df_filtered.empty:
                avg_valence = df_filtered['valence'].mean()
                mood = "😄 Happy" if avg_valence > 0.6 else "😐 Neutral" if avg_valence > 0.4 else "😢 Sad"
                st.metric("Average Mood",
                            mood,
                            f"Valence: {avg_valence:.2f}")
            else:
                st.metric("Average Mood", "N/A", "No data")
        
        # ---------------------------------------------------
        # DECADE DEEP DIVE INSIGHTS
        # ---------------------------------------------------
        st.divider()
        st.subheader("🔎 Decade Deep Dive Insights")
        st.info("**🔍 Strategic Goal:** Compare audio trends against the previous decade and highlight the most successful artists and tracks of the era.")

        # 1. Preparação dos Dados (df_year)

        
        if 'decade' not in df_year.columns:
            df_year['decade'] = (df_year['year'] // 10) * 10
            
        # Calcular o perfil médio de áudio por década
        decade_audio_profile = df_year.groupby('decade')[
            ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'speechiness']
        ].mean()

        # 2. Criar o seletor de década
        decade_options = sorted(df['decade'].unique())
        
        selected_decade = st.selectbox(
            "Select a Decade to Analyze:",
            options=decade_options,
            index=len(decade_options)-1,
            key="deep_dive_decade_select"
        )

        # 3. Obter os perfis (atual e anterior)
        profile = decade_audio_profile.loc[selected_decade] if selected_decade in decade_audio_profile.index else None
        previous_decade = selected_decade - 10
        previous_profile = decade_audio_profile.loc[previous_decade] if previous_decade in decade_audio_profile.index else None

        # 4. Exibir Comparação de Perfis de Áudio com VARIAÇÃO %
        st.markdown("#### Audio Feature Evolution: Comparison vs. Previous Decade")

        if profile is not None:
            col_comp1, col_comp2, col_comp3, col_comp4, col_comp5, col_comp6 = st.columns(6)
            cols = [col_comp1, col_comp2, col_comp3, col_comp4, col_comp5, col_comp6]
            features = ['energy', 'danceability', 'valence', 'acousticness', 'loudness', 'speechiness']
            
            for i, feat in enumerate(features):
                current_val = profile[feat]
                
                if previous_profile is not None:
                    previous_val = previous_profile[feat]
                    variation = ((current_val - previous_val) / previous_val) * 100 if previous_val != 0 else 0
                    delta_text = f"{variation:.1f}% vs. {previous_decade}s"
                else:
                    delta_text = f"No {previous_decade}s data"

                with cols[i]:
                    st.metric(
                        feat.capitalize(), 
                        f"{current_val:.3f}", 
                        delta=delta_text
                    )
        else:
            st.info("No audio profile data available for the selected decade.")
        
        # --- Top 20 Tracks & Artists ---
        st.divider()
        st.markdown("#### Top Commercial Hits & Artists")

        decade_df = df[df['decade'] == selected_decade].copy()
        
        col_tracks, col_artists = st.columns(2)
        
        with col_tracks:
            st.markdown("**Top 20 Tracks by Popularity**")
            
            top_tracks = decade_df.sort_values('popularity', ascending=False)\
                                    .drop_duplicates(subset=['name', 'artists'])\
                                    .head(20)
            
            top_tracks['Artists'] = top_tracks['artists'].str.replace(r"[\[\]\'\"]", "", regex=True)
            
            st.dataframe(
                top_tracks[['name', 'Artists', 'popularity', 'energy', 'danceability']],
                hide_index=True,
                use_container_width=True
            )
            
        with col_artists:
            st.markdown("**Top 20 Artists by Average Popularity**")
            
            artist_pop = decade_df.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(20).reset_index()
            
            artist_pop['Artists'] = artist_pop['artists'].str.replace(r"[\[\]\'\"]", "", regex=True)
            
            st.dataframe(
                artist_pop[['Artists', 'popularity']],
                hide_index=True,
                use_container_width=True
            )
        
        st.warning("⚠️ **Genre Unavailable**: The genre for artists and tracks cannot be displayed because the data (Tracks/Year) does not have a linking key to the genre file.")

    # --------------------------------------------------------
    # --- TAB 3: AI Data Consultant ---
    # --------------------------------------------------------
    elif st.session_state.current_tab == "AI Consultant":
        st.header("🧠 Music Data Consultant 💬")
        st.markdown("Ask complex questions about music trends, correlations, and patterns. The AI executes Python code to analyze the data.")

        if not IA_DISPONIVEL:
            st.warning("LangChain libraries are not properly installed. The AI tab is disabled.")
            st.info("Please run the structured reinstallation in the terminal.")

        else:
            try:
                # Check if API key exists
                import os
                api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
                
                if api_key is None:
                    st.warning("Google API key not found.")
                    st.write("Please add the `GOOGLE_API_KEY` environment variable.")
                    st.stop()
                
                @st.cache_resource
                def get_ai_agent(api_key: str):
                    model = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key,
                        temperature=0
                    )
                    return create_agent(model=model, tools=tools, system_prompt=system_prompt)
                
                agent = get_ai_agent(api_key)

                # Initialize chat history
                if "chat_messages_executor" not in st.session_state:
                    st.session_state.chat_messages_executor = []

                # Initialize button prompt
                if 'button_prompt' not in st.session_state:
                    st.session_state.button_prompt = None
                
                    # Function to handle button prompts
                def set_button_prompt(prompt):
                    st.session_state.button_prompt = prompt

                # Display messages from history
                for message in st.session_state.chat_messages_executor:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Get user input from chat box
                chat_input = st.chat_input("Ex: What's the correlation between energy and popularity for explicit tracks?")

                st.divider()

                # Pre-defined question buttons
                st.subheader("**💡 Suggested Questions:**")

                # ROW 1: POPULARITY ANALYSIS
                st.markdown("**1. POPULARITY DRIVERS**")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    if st.button("📊 Feature Correlations", key='btn_1a', use_container_width=True):
                        set_button_prompt("Calculate the correlation between all audio features (energy, danceability, valence, acousticness) and popularity.")

                with col2:
                    if st.button("🎯 Popularity Formula", key='btn_1b', use_container_width=True):
                        set_button_prompt("What combination of audio features best predicts popularity? Show the top 3 correlated features.")

                with col3:
                    if st.button("📈 High-Energy Hits", key='btn_1c', use_container_width=True):
                        set_button_prompt("What's the average popularity for songs with energy > 0.8 vs energy < 0.3?")

                with col4:
                    if st.button("💃 Dance vs Popularity", key='btn_1d', use_container_width=True):
                        set_button_prompt("Show the correlation between danceability and popularity for each decade since 1980.")

                with col5:
                    if st.button("🎭 Mood Impact", key='btn_1e', use_container_width=True):
                        set_button_prompt("Compare average popularity for high valence (>0.7) vs low valence (<0.3) songs.")

                # ROW 2: GENRE AND ARTIST INSIGHTS
                st.markdown("**2. GENRE & ARTIST INSIGHTS**")
                col6, col7, col8, col9, col10 = st.columns(5)

                with col6:
                    if st.button("🎸 Top Genres", key='btn_2a', use_container_width=True):
                        set_button_prompt("List the top 10 genres by average popularity using df_genres.")

                with col7:
                    if st.button("🌟 Artist Rankings", key='btn_2b', use_container_width=True):
                        set_button_prompt("Who are the top 10 artists by average popularity with at least 5 tracks?")

                with col8:
                    if st.button("🎵 Genre Features", key='btn_2c', use_container_width=True):
                        set_button_prompt("Compare average energy and danceability for rock, pop, and hip-hop genres.")

                with col9:
                    if st.button("🏆 Consistent Artists", key='btn_2d', use_container_width=True):
                        set_button_prompt("Which artists have the lowest standard deviation in popularity (most consistent)?")

                with col10:
                    if st.button("🎼 Genre Evolution", key='btn_2e', use_container_width=True):
                        set_button_prompt("How has the average energy of rock music changed from the 1970s to 2010s?")

                # ROW 3: TEMPORAL TRENDS
                st.markdown("**3. TEMPORAL TRENDS**")
                col11, col12, col13, col14, col15 = st.columns(5)

                with col11:
                    if st.button("📅 Feature Evolution", key='btn_3a', use_container_width=True):
                        set_button_prompt("How have danceability and energy evolved from 1960 to 2020? Show decade averages.")

                with col12:
                    if st.button("⏱️ Song Duration", key='btn_3b', use_container_width=True):
                        set_button_prompt("What's the trend in average song duration from 1960 to 2020?")

                with col13:
                    if st.button("🎚️ Loudness Wars", key='btn_3c', use_container_width=True):
                        set_button_prompt("Show how average loudness has changed over the decades. Is music getting louder?")

                with col14:
                    if st.button("🎹 Acoustic Trends", key='btn_3d', use_container_width=True):
                        set_button_prompt("Has music become more or less acoustic over time? Show the trend from 1950 to 2020.")

                with col15:
                    if st.button("🔥 Modern vs Classic", key='btn_3e', use_container_width=True):
                        set_button_prompt("Compare average audio features between songs from before 1990 and after 2010.")

                # ROW 4: DEEP ANALYSIS
                st.markdown("**4. DEEP STATISTICAL ANALYSIS**")
                col16, col17, col18, col19, col20 = st.columns(5)

                with col16:
                    if st.button("🔗 Feature Clusters", key='btn_4a', use_container_width=True):
                        set_button_prompt("Which audio features tend to occur together? Show the correlation matrix for all features.")

                with col17:
                    if st.button("🎯 Explicit Impact", key='btn_4b', use_container_width=True):
                        set_button_prompt("Do explicit songs have higher popularity on average? Compare explicit vs clean songs.")

                with col18:
                    if st.button("🎵 Key Analysis", key='btn_4c', use_container_width=True):
                        set_button_prompt("Which musical key (0-11) is most popular? Show average popularity by key.")

                with col19:
                    if st.button("🎭 Major vs Minor", key='btn_4d', use_container_width=True):
                        set_button_prompt("Compare average valence and popularity between major (mode=1) and minor (mode=0) keys.")

                with col20:
                    if st.button("📊 Outlier Songs", key='btn_4e', use_container_width=True):
                        set_button_prompt("Find songs with unusual combinations: high energy but low danceability (energy>0.8, danceability<0.3).")

                st.divider()

                # Process input
                if user_input := st.session_state.button_prompt or chat_input:
                    st.chat_message("user").markdown(user_input)
                    st.session_state.chat_messages_executor.append({"role": "user", "content": user_input})
                    
                    # Clear button prompt after use
                    if st.session_state.button_prompt:
                        st.session_state.button_prompt = None

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()

                        try:
                            with st.spinner("Analyzing music data..."):
                                response = agent.invoke({"messages": st.session_state.chat_messages_executor})

                            # Check for malformed call
                            if response["messages"][-1].response_metadata.get('finish_reason') == 'MALFORMED_FUNCTION_CALL':
                                message_placeholder.empty()
                                st.error("The model had difficulty processing your request. Please try rephrasing.")
                                st.stop()    

                            # Extract AI response
                            ai_content = response["messages"][-1].content

                            if isinstance(ai_content, list) and len(ai_content) > 0:
                                text_content = ai_content[0].get('text', '')
                            else:
                                text_content = ai_content

                            # DEBUG
                            with st.expander("🔍 Debug: View executed code"):
                                st.code(str(response), language="python")

                            # Display text only
                            message_placeholder.markdown(text_content)
                            st.session_state.chat_messages_executor.append({
                                "role": "assistant",
                                "content": text_content
                            })                            

                        except Exception as e:
                            message_placeholder.empty()
                            st.error(f"Error during processing: {str(e)}")
                            st.write("Error details:", e)

            except Exception as e:
                st.error(f"Error initializing Agent: {e}")

# --------------------------------------------------------
# TAB 4: DATA EXPLORER
# --------------------------------------------------------
    elif st.session_state.current_tab == "Data Explorer":
        st.header("📁 Data Explorer")
        st.info("Showing the raw, filtered data. Use the sidebar filters to explore.")
        
        # Define the features we want to format
        features_to_format = [
        'danceability', 'energy', 'valence', 'acousticness', 
        'instrumentalness', 'speechiness', 'liveness'
        ]
        
        # Create a dictionary of formatters
        formatter = {feat: "{:.1%}" for feat in features_to_format}

        # Apply the formatting to the dataframe's style
        st.dataframe(df_filtered.style.format(formatter))

        # Select dataset to view
        dataset_choice = st.selectbox(
            "Select Dataset to View:",
            ["Main Data", "By Artist", "By Genre", "By Year", "With Genres"]
        )
        
        # Dataset info cards
        col1, col2, col3 = st.columns(3)
        
        if dataset_choice == "Main Data":
            with col1:
                st.metric("Total Rows", f"{len(music_data['main']):,}")
            with col2:
                st.metric("Total Columns", f"{len(music_data['main'].columns)}")
            with col3:
                st.metric("Memory Usage", f"{music_data['main'].memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.subheader("📊 Dataset Preview")
            st.dataframe(
                music_data['main'].head(100),
                use_container_width=True,
                height=400
            )
            
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column': music_data['main'].columns,
                'Type': music_data['main'].dtypes,
                'Non-Null Count': music_data['main'].count(),
                'Null %': (music_data['main'].isnull().sum() / len(music_data['main']) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        # [Repeat similar structure for other datasets]

    else:
        st.info("Waiting for music data files to start the application.")


