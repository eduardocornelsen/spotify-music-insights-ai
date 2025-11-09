import streamlit as st
import pandas as pd
import plotly.express as px
import io
import sys
import re
from pathlib import Path
from scipy import stats
from langchain.tools import tool

# --- Importações do LangChain (Tool Calling Agent) ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    
    IA_DISPONIVEL = True
except ImportError:
    IA_DISPONIVEL = False
except Exception:
    IA_DISPONIVEL = False

# Read System Prompt from file
system_prompt = Path("./prompts/system.txt").read_text()

# --- Configuração da Página ---
st.set_page_config(
    page_title="MusicInsights AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Spotify Theme ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #121212;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #181818;
        border-radius: 8px;
        padding: 4px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #B3B3B3;
        border-radius: 4px;
        font-weight: 600;
        padding: 8px 16px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1DB954;
        color: #000000;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        border: none;
        border-radius: 500px;
        font-weight: 700;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1ED760;
        transform: scale(1.02);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #181818;
        border: 1px solid #282828;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #181818;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 8px;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background-color: #282828;
    }
    
    /* Radio button group */
    .stRadio > div {
        background-color: #181818;
        padding: 8px;
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #181818 !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #181818 !important;
    }
    
    /* Navigation buttons in sidebar */
    .nav-button {
        width: 100%;
        text-align: left;
        padding: 12px 16px;
        margin: 4px 0;
        background-color: #181818;
        border: none;
        border-radius: 8px;
        color: #B3B3B3;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .nav-button:hover {
        background-color: #282828;
        color: #FFFFFF;
    }
    
    .nav-button.active {
        background-color: #1DB954;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# --- Carregar e Limpar os Dados (com cache) ---
@st.cache_data
def load_data():
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

music_data = load_data()

# --- LOGO E DADOS DO DATASET ---
spotify_logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png"
dataset_url = "https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks"

# --- Initialize Session State for Navigation ---
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dashboard"


# --- Sidebar Navigation ---
st.sidebar.image(spotify_logo_url, width=120)
st.sidebar.markdown("""
<h2 style='color: #1DB954; margin-bottom: 20px;'>MusicInsights AI</h2>
""", unsafe_allow_html=True)

# Navigation Section
st.sidebar.markdown("### 🎵 Navigation")

# Navigation buttons
tabs = ["Dashboard", "Insights", "AI Consultant", "Data Explorer"]
for tab in tabs:
    if st.sidebar.button(
        f"{'📊' if tab == 'Dashboard' else '💡' if tab == 'Insights' else '🤖' if tab == 'AI Consultant' else '📁'} {tab}", 
        key=f"nav_{tab}",
        use_container_width=True,
        type="primary" if st.session_state.current_tab == tab else "secondary"
    ):
        st.session_state.current_tab = tab


st.sidebar.markdown("---")

# About Section
st.sidebar.markdown("### 📱 About")
st.sidebar.markdown(
    f"""
    <div style='background-color: #181818; padding: 12px; border-radius: 8px; border: 1px solid #282828;'>
    <p style='color: #B3B3B3; font-size: 14px; margin: 0;'>
    Interactive music analytics platform powered by <b style='color: #1DB954;'>AI</b> and advanced data visualization.
    </p>
    
    <p style='color: #B3B3B3; font-size: 12px; margin-top: 8px;'>
    <b>Data:</b> <a href='{dataset_url}' style='color: #1DB954;'>Spotify Dataset (160k+ tracks)</a><br>
    <b>Tech:</b> Streamlit + LangChain + Gemini<br>
    <b>Author:</b> Eduardo Cornelsen
    </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Quick Stats
if music_data is not None:
    st.sidebar.markdown("### 📈 Quick Stats")
    total_tracks = len(music_data['main'])
    total_artists = music_data['by_artist'].shape[0]
    total_genres = music_data['by_genres'].shape[0]
    
    st.sidebar.markdown(f"""
    <div style='background-color: #181818; padding: 12px; border-radius: 8px; border: 1px solid #282828;'>
    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
        <span style='color: #B3B3B3;'>Tracks:</span>
        <span style='color: #1DB954; font-weight: bold;'>{total_tracks:,}</span>
    </div>
    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
        <span style='color: #B3B3B3;'>Artists:</span>
        <span style='color: #1DB954; font-weight: bold;'>{total_artists:,}</span>
    </div>
    <div style='display: flex; justify-content: space-between;'>
        <span style='color: #B3B3B3;'>Genres:</span>
        <span style='color: #1DB954; font-weight: bold;'>{total_genres:,}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Title ---
st.markdown("""
<h1 style='text-align: center; color: #1DB954; font-size: 48px; margin-bottom: 8px;'>
    🎵 MusicInsights AI
</h1>
<p style='text-align: center; color: #B3B3B3; font-size: 18px; margin-bottom: 32px;'>
    Interactive Music Analytics with Advanced AI & Data Visualization
</p>
""", unsafe_allow_html=True)


# --------------------------------------------------------
# CRIAR A FERRAMENTA CUSTOMIZADA COM IA
# --------------------------------------------------------

@tool
def PythonCodeExecutor(code: str) -> str:
    """
    Execute Python code for data analysis on music DataFrames.
    Available dataframes: 
    - df (main data with all tracks)
    - df_artist (aggregated by artist)
    - df_genres (aggregated by genres)
    - df_year (aggregated by year)
    - df_with_genres (tracks with genre info)
    
    CRITICAL: You MUST use the actual DataFrames - do NOT create fake data.
    Always verify results with actual data from the dataframes.
    Example: print(df['popularity'].mean())
    """
    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        
        global music_data
        df = music_data['main'].copy()
        df_artist = music_data['by_artist'].copy()
        df_genres = music_data['by_genres'].copy()
        df_year = music_data['by_year'].copy()
        df_with_genres = music_data['with_genres'].copy()
        
        # Validate code doesn't create fake data
        if 'pd.DataFrame' in code and '{' in code:
            return "ERROR: Do NOT create fake DataFrames. Use the existing dataframe variables only."

        exec(code, {
            'df': df,
            'df_artist': df_artist,
            'df_genres': df_genres,
            'df_year': df_year,
            'df_with_genres': df_with_genres,
            'pd': pd
        }, {})
        
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        # Check if output is empty
        if not output.strip():
            return "ERROR: No output generated. Make sure to use print() to display results."
                
        return output
    
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {e}"

tools = [PythonCodeExecutor]

# --- Renderização do App ---
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
    key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                   6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    key_options = list(key_mapping.values())

    # ---------------------------------------------------
    # GLOBAL FILTERS SECTION (Available for all tabs)
    # ---------------------------------------------------
    with st.expander("🎛️ **Global Filters** (Apply to all visualizations)", expanded=True):
        
        col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 2])
        
        with col_filter1:
            # 1. Get the list of decades
            decade_options = sorted(df[df['decade'] >= 1950]['decade'].unique())
            
            # 2. Create the select_slider
            decade_range = st.select_slider(
                "Filter by Decade Range:",
                options=decade_options,
                value=(decade_options[0], decade_options[-1]),
                key="global_decade_slider"
            )
            
            # 3. Add the popularity slider underneath
            popularity_range = st.slider(
                "Popularity Range:",
                min_value=0,
                max_value=100,
                value=(0, 100),
                key="global_popularity_filter"
            )
        
        with col_filter2:
            explicit_filter = st.radio(
                "Content Type:",
                options=["All", "Clean Only", "Explicit Only"],
                key="global_explicit_filter"
            )
        
        with col_filter3:
            key_filter_names = st.multiselect(
                "Filter by Key:",
                options=key_options,
                default=key_options,
                key="global_key_filter"
            )
    
    selected_key_numbers = [k for k, v in key_mapping.items() if v in key_filter_names]

    @st.cache_data 
    def get_filtered_data(df, decade_range, pop_range, explicit, keys):
        """Filters the main DataFrame based on global selections."""
        df_filtered = df.copy()
        
        if decade_range:
            df_filtered = df_filtered[
                (df_filtered['decade'] >= decade_range[0]) & 
                (df_filtered['decade'] <= decade_range[1])
            ]
        
        df_filtered = df_filtered[(df_filtered['popularity'] >= pop_range[0]) & 
                                  (df_filtered['popularity'] <= pop_range[1])]
        
        if explicit == "Clean Only":
            df_filtered = df_filtered[df_filtered['explicit'] == 0]
        elif explicit == "Explicit Only":
            df_filtered = df_filtered[df_filtered['explicit'] == 1]
        
        if keys:
            df_filtered = df_filtered[df_filtered['key'].isin(keys)]
            
        return df_filtered

    df_filtered = get_filtered_data(
        df, 
        decade_range,
        popularity_range, 
        explicit_filter, 
        selected_key_numbers
    )

    # --------------------------------------------------------
    # TAB 1: DASHBOARD
    # --------------------------------------------------------
    if st.session_state.current_tab == "Dashboard":
        st.header("📊 Music Analytics Dashboard")
        st.markdown("Explore trends in audio features and understand what makes songs popular.")

    # Performance Mode Toggle
    col_perf1, col_perf2 = st.columns([1, 4])
    with col_perf1:
        performance_mode = st.checkbox("⚡ Performance Mode", value=False, key="perf_mode", 
                                       help="Load visualizations on-demand for better performance")
    with col_perf2:
        if performance_mode:
            st.info("Performance Mode enabled: Click on tabs to load visualizations")
    
    if performance_mode:
        # ===== PERFORMANCE MODE: TABBED INTERFACE =====
        viz_tabs = st.tabs([
            "📈 Evolution", "📊 Correlations", "🎸 Genres", "🔞 Explicit",
            "🔗 Features", "⏱️ Temporal", "👤 Artists", "🔍 Explorer",
            "🎵 Keys", "📅 Decades", "💰 Economics", "🎚️ Tempo",
            "📈 Explicit Trends", "🌟 Popularity", "🚀 Artist Evolution"
        ])

        # ---------------------------------------------------
        # EXECUTIVE SUMMARY - QUICK INSIGHTS
        # ---------------------------------------------------
        st.subheader("📈 Dashboard Summary")

        # Create quick insight cards
        col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)

        with col_quick1:
            trending_feature = df_year.iloc[-1][['energy', 'danceability', 'valence']].idxmax()
            st.info(f"**🔥 Trending Feature**\n\n{trending_feature.capitalize()} is dominating modern music")

        with col_quick2:
            growth_rate = ((df_year.iloc[-1]['popularity'] - df_year.iloc[-10]['popularity']) / df_year.iloc[-10]['popularity']) * 100
            st.success(f"**📊 10-Year Growth**\n\n{growth_rate:.1f}% popularity increase")

        with col_quick3:
            dominant_mode = "Major" if df_filtered['mode'].mean() > 0.5 else "Minor"
            st.warning(f"**🎵 Dominant Mode**\n\n{dominant_mode} keys rule the charts")

        with col_quick4:
            avg_duration = df_filtered['duration_ms'].mean() / 60000
            st.error(f"**⏱️ Avg Duration**\n\n{avg_duration:.1f} minutes per song")

        # ---------------------------------------------------
        # VIZ 1: Audio Features Over Time - ALL FEATURES
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[0]:
            st.subheader("1. Evolution of ALL Audio Features Over Decades")
            st.info("**🔍 Key Question:** How have ALL musical characteristics evolved? Which features show the most dramatic changes, indicating major shifts in music production and consumer preferences?")
            
            # Select which features to display
            all_audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                                'instrumentalness', 'speechiness', 'liveness', 'loudness']
            
            selected_features_viz1 = st.multiselect(
                "Select Audio Features to Display:",
                options=all_audio_features,
                default=all_audio_features,
                key="features_selector_viz1"
            )
            
            # Toggle for normalized view
            normalize_features = st.checkbox("Normalize features for better comparison", value=False, key="normalize_viz1")
            
            # Prepare data for time series
            features_cols = ['year'] + selected_features_viz1
            features_over_time = df_year[features_cols].copy()
            
            if normalize_features:
                for feat in selected_features_viz1:
                    features_over_time[feat] = (features_over_time[feat] - features_over_time[feat].min()) / \
                                            (features_over_time[feat].max() - features_over_time[feat].min())
            
            features_melted = features_over_time.melt(
                id_vars=['year'],
                var_name='Feature',
                value_name='Value'
            )
            
            fig_evolution = px.line(
                features_melted,
                x='year',
                y='Value',
                color='Feature',
                title='Evolution of Audio Features (1920-2020)',
                labels={'Value': 'Feature Value' + (' (Normalized)' if normalize_features else ' (0-1)'), 
                    'year': 'Year'},
                template='plotly_dark'
            )
            
            # Update layout for Spotify theme
            fig_evolution.update_layout(
                plot_bgcolor='#181818',
                paper_bgcolor='#181818',
                font_color='#B3B3B3'
            )
            
            # Add annotations for major music eras
            music_eras = [
                (1955, "Rock'n'Roll Era"),
                (1975, "Disco Era"),
                (1985, "MTV Era"),
                (1995, "Hip-Hop Rise"),
                (2005, "Digital/Streaming Era")
            ]
            
            for year, era in music_eras:
                fig_evolution.add_vline(x=year, line_dash="dash", line_color="gray", opacity=0.3)
                fig_evolution.add_annotation(x=year, y=0.9, text=era, showarrow=False, 
                                            textangle=-90, font=dict(size=10, color="gray"))
            
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        # ---------------------------------------------------
        # VIZ 2: Popularity vs Audio Features - ENHANCED
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[1]:
            st.subheader("2. Correlation Analysis: Popularity vs Audio Features")
            st.info("**🔍 Key Questions:** Which audio feature best predicts commercial success? Are there threshold values that guarantee popularity? How do feature combinations create hit songs?")
            
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
            
            # Sample for better performance
            df_sample = df_filtered.sample(min(5000, len(df_filtered)))
            
            if viz2_type == "Scatter with Trend":
                fig_correlation = px.scatter(
                    df_sample,
                    x=selected_feature,
                    y='popularity',
                    title=f'Popularity vs {selected_feature.capitalize()}',
                    opacity=0.6,
                    trendline='ols',
                    labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'}
                )
                
                if add_percentiles:
                    # Add percentile lines
                    p25 = df_sample[selected_feature].quantile(0.25)
                    p75 = df_sample[selected_feature].quantile(0.75)
                    fig_correlation.add_vline(x=p25, line_dash="dash", line_color="red", opacity=0.5)
                    fig_correlation.add_vline(x=p75, line_dash="dash", line_color="green", opacity=0.5)
                    
            elif viz2_type == "Hexbin Density":
                fig_correlation = px.density_heatmap(
                    df_sample,
                    x=selected_feature,
                    y='popularity',
                    title=f'Density: Popularity vs {selected_feature.capitalize()}',
                    labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'},
                    nbinsx=30,
                    nbinsy=20
                )
                
            else:  # Box Plot by Bins
                # Create bins for the feature
                df_sample['feature_bin'] = pd.cut(df_sample[selected_feature], bins=5, 
                                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                fig_correlation = px.box(
                    df_sample,
                    x='feature_bin',
                    y='popularity',
                    title=f'Popularity Distribution by {selected_feature.capitalize()} Levels',
                    labels={'feature_bin': f'{selected_feature.capitalize()} Level', 'popularity': 'Popularity Score'}
                )
            
            st.plotly_chart(fig_correlation, use_container_width=True)
            
            # Show correlation coefficient
            correlation = df_filtered[selected_feature].corr(df_filtered['popularity'])
            st.metric(f"Correlation Coefficient", f"{correlation:.3f}", 
                    delta=f"{'Positive' if correlation > 0 else 'Negative'} correlation")

        # ---------------------------------------------------
        # VIZ 3: Genre Analysis - ENHANCED
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[2]:
            st.subheader("3. Genre DNA: Audio Feature Signatures")
            st.info("**🔍 Strategic Questions:** What is the unique 'DNA' of each genre? Which genres are converging in style? How can artists differentiate within saturated genres?")
            
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
                    ["Box Plot", "Radar Chart"], # <-- ALTERAÇÃO: Removido "Violin Plot"
                    horizontal=True,
                    key="genre_viz_type"
                )
            
            # Get top genres
            top_genres = df_genres.nlargest(15, 'popularity')
            
            if genre_viz_type == "Box Plot":
                fig_genre = px.box(
                    top_genres,
                    x='genres',
                    y=genre_feature,
                    color='genres',
                    title=f'{genre_feature.capitalize()} Distribution Across Top 15 Genres',
                    labels={'genres': 'Genre', genre_feature: f'{genre_feature.capitalize()}'}
                )
                fig_genre.update_layout(showlegend=False, xaxis_tickangle=-45)
                
            # <-- ALTERAÇÃO: O bloco "elif" para Violin Plot foi completamente removido
                
            else:  # Radar Chart (Este "else" agora é ativado quando "Radar Chart" é selecionado)
                # Select top 8 genres for radar chart
                import plotly.graph_objects as go
                
                top_8_genres = df_genres.nlargest(8, 'popularity')
                features_for_radar = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
                
                fig_genre = go.Figure()
                
                for _, genre_row in top_8_genres.iterrows():
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
                    title="Genre DNA: Audio Feature Signatures"
                )
            
            st.plotly_chart(fig_genre, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 4: Explicit Content Strategy Analysis
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[3]:
            st.subheader("4. Explicit Content: Commercial Strategy Analysis")
            st.info("**🔍 Strategic Questions:** Is explicit content a successful commercial strategy? Which genres benefit most from explicit content? Should artists release both clean and explicit versions?")
            
            # Create explicit label
            df_filtered['explicit_label'] = df_filtered['explicit'].map({0: 'Clean', 1: 'Explicit'})
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Popularity comparison
                fig_explicit = px.violin(
                    df_filtered[df_filtered['popularity'] > 0],
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
                    df_filtered,
                    x='explicit_label',
                    y=explicit_feature,
                    color='explicit_label',
                    title=f'{explicit_feature.capitalize()} by Content Type',
                    labels={'explicit_label': 'Content Type', explicit_feature: explicit_feature.capitalize()}
                )
                st.plotly_chart(fig_explicit_feat, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 5: Feature Correlation & Clustering
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[4]:
            st.subheader("5. Feature Relationships: The Music Formula")
            st.info("**🔍 Strategic Questions:** Which features must be balanced together? Can we identify 'formula' combinations for different popularity levels? What trade-offs do producers make?")
            
            # Toggle between correlation matrix and cluster analysis
            analysis_type = st.radio(
                "Analysis Type:",
                ["Correlation Matrix", "Feature Pairs Analysis", "Success Formula"],
                horizontal=True,
                key="analysis_type_viz5"
            )
            
            audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'instrumentalness', 'liveness', 'loudness', 'speechiness', 'popularity']
            
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
                # Show strongest correlations
                correlation_matrix = df_filtered[audio_features].corr()
                
                # Get correlations with popularity
                pop_corr = correlation_matrix['popularity'].drop('popularity').sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Positive Correlations with Popularity:**")
                    for feat, corr in pop_corr.head(3).items():
                        st.write(f"• {feat}: {corr:.3f}")
                
                with col2:
                    st.markdown("**Top Negative Correlations with Popularity:**")
                    for feat, corr in pop_corr.tail(3).items():
                        st.write(f"• {feat}: {corr:.3f}")
                        
            else:  # Success Formula
                # Segment songs by popularity
                df_success = df_filtered.copy()
                df_success['success_level'] = pd.cut(df_success['popularity'], 
                                                    bins=[0, 30, 60, 100],
                                                    labels=['Low', 'Medium', 'High'])
                
                # Calculate mean features by success level
                success_formula = df_success.groupby('success_level')[audio_features[:-1]].mean()
                
                fig_formula = px.bar(
                    success_formula.T,
                    title="The Success Formula: Average Features by Popularity Level",
                    labels={'index': 'Audio Feature', 'value': 'Average Value'},
                    barmode='group'
                )
                st.plotly_chart(fig_formula, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 6: Temporal Strategy Analysis
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[5]:
            st.subheader("6. Temporal Trends: Predicting the Future of Music")
            st.info("**🔍 Strategic Questions:** Can we predict the next trend? Are songs converging to a standard formula? What features are becoming obsolete?")
            
            col_tempo1, col_tempo2 = st.columns(2)
            
            # Convert duration to minutes BEFORE using it
            df_year['duration_min'] = df_year['duration_ms'] / 60000
            
            with col_tempo1:
                # Add trend prediction toggle
                show_prediction = st.checkbox("Show trend projection", value=False, key="prediction_duration")
                
                fig_duration = px.line(
                    df_year[df_year['year'] >= 1960],
                    x='year',
                    y='duration_min',
                    title='Song Duration: The Attention Span Crisis',
                    labels={'duration_min': 'Duration (minutes)', 'year': 'Year'}
                )
                
                if show_prediction:
                    # Simple linear projection
                    try:
                        from scipy import stats
                        recent_years = df_year[df_year['year'] >= 2000].copy()
                        
                        # Make sure duration_min is calculated for recent_years
                        if 'duration_min' not in recent_years.columns:
                            recent_years['duration_min'] = recent_years['duration_ms'] / 60000
                        
                        # Check if we have valid data
                        if len(recent_years) > 0 and not recent_years['duration_min'].isna().all():
                            slope, intercept, _, _, _ = stats.linregress(recent_years['year'], recent_years['duration_min'])
                            future_years = list(range(2020, 2031))
                            future_duration = [slope * year + intercept for year in future_years]
                            fig_duration.add_scatter(x=future_years, y=future_duration, mode='lines', 
                                                    name='Projection', line=dict(dash='dash', color='red'))
                    except:
                        pass  # If scipy is not available or calculation fails, just skip projection
                
                st.plotly_chart(fig_duration, use_container_width=True)
            
            with col_tempo2:
                            
                fig_tempo = px.line(
                    df_year[df_year['year'] >= 1960],
                    x='year',
                    y='tempo',
                    title='Tempo Evolution: The BPM Arms Race',
                    labels={'tempo': 'Tempo (BPM)', 'year': 'Year'}
                )
                st.plotly_chart(fig_tempo, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 7: Artist Success Patterns
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[6]:
            st.subheader("7. Artist Success Patterns: One-Hit Wonder vs Consistency")
            st.info("**🔍 Strategic Questions:** What separates consistent hitmakers from one-hit wonders? Is it better to release many average songs or few excellent ones? How do successful artists maintain their sound signature?")
            
            # Enhanced artist analysis
            artist_strategy = st.radio(
                "Analysis Focus:",
                ["Top 50 Artists Overview", "Consistency Analysis", "Feature Signature (Top 5)"],
                horizontal=True,
                key="artist_strategy"
            )
            
            top_artists = df_artist.nlargest(50, 'popularity')
            
            if artist_strategy == "Top 50 Artists Overview":
                fig_artists = px.scatter(
                    top_artists,
                    x='count',
                    y='popularity',
                    size='energy',
                    color='valence',
                    hover_data=['artists'],
                    title='Artist Strategy: Volume vs Quality',
                    labels={'count': 'Number of Tracks', 'popularity': 'Average Popularity', 'valence': 'Valence'}
                )
                
                # Add quadrant lines
                median_count = top_artists['count'].median()
                median_pop = top_artists['popularity'].median()
                fig_artists.add_hline(y=median_pop, line_dash="dash", line_color="gray", opacity=0.5)
                fig_artists.add_vline(x=median_count, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Add quadrant labels
                fig_artists.add_annotation(x=median_count*0.3, y=median_pop*1.2, text="Quality over Quantity", 
                                        showarrow=False, font=dict(color="green"))
                fig_artists.add_annotation(x=median_count*1.7, y=median_pop*1.2, text="Consistent Hitmakers", 
                                        showarrow=False, font=dict(color="#89ccff"))
                
            elif artist_strategy == "Consistency Analysis":
                # Calculate coefficient of variation (std/mean) for each artist
                artist_consistency = []
                
                for artist in top_artists['artists'].head(50):
                    artist_songs = df[df['artists'].str.contains(artist, na=False, case=True)]['popularity']
                    # --- FIM DA CORREÇÃO ---
                    
                    if len(artist_songs) > 1:
                        cv = artist_songs.std() / artist_songs.mean() if artist_songs.mean() > 0 else 0
                        artist_consistency.append({'artist': artist[:20], 'consistency': 1 - cv, 
                                                'avg_popularity': artist_songs.mean()})
                
                consistency_df = pd.DataFrame(artist_consistency)
                
                # A verificação 'if empty' ainda é útil, caso alguns artistas 
                # ainda não sejam encontrados.
                if consistency_df.empty:
                    st.info("Não foi possível calcular a consistência dos artistas. (Nenhuma correspondência de artista com >1 música encontrada nos dados brutos).")
                    import plotly.graph_objects as go
                    fig_artists = go.Figure()
                    fig_artists.update_layout(title='Artist Consistency Score (No data to display)')
                else:
                    # O DataFrame é válido, então podemos criar o gráfico
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
                            
            else:  # Feature Signature
                # Compare top 5 artists' average features
                import plotly.graph_objects as go
                
                top_5_artists = top_artists.head(5)
                features_for_signature = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
                
                fig_artists = go.Figure()
                
                for _, artist_row in top_5_artists.iterrows():
                    values = [artist_row[feat] for feat in features_for_signature]
                    fig_artists.add_trace(go.Scatterpolar(
                        r=values,
                        theta=features_for_signature,
                        fill='toself',
                        name=artist_row['artists'][:20]
                    ))
                
                fig_artists.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Artist Sound Signatures: What Makes Them Unique"
                )
            
            st.plotly_chart(fig_artists, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 8: Interactive Feature Explorer - ENHANCED
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[7]:
            st.subheader("8. Feature Interaction Explorer: Finding the Sweet Spot")
            st.info("**🔍 Strategic Questions:** What feature combinations create viral hits? Are there 'dead zones' to avoid? How do different eras prefer different feature combinations?")
            
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
                    index=0,
                    key="density_y_feature"
                )
            
            with col_exp3:
                viz_style = st.radio(
                    "Style:",
                    ["Density Heatmap", "Scatter with Size", "Contour Plot"],
                    key="viz_style_8"
                )
            
            # Add decade filter for this specific viz
            decade_filter_8 = st.select_slider(
                "Filter by Decade Range:",
                options=sorted(df[df['decade'] >= 1950]['decade'].unique()),
                value=(1990, 2020),
                key="decade_filter_8"
            )
            
            # Filter data
            df_density = df_filtered[(df_filtered['decade'] >= decade_filter_8[0]) & 
                                    (df_filtered['decade'] <= decade_filter_8[1])]
            df_density_sample = df_density.sample(min(10000, len(df_density)))
            
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
            
            st.plotly_chart(fig_density, use_container_width=True)
            
            # Show insights
            high_pop_threshold = df_density['popularity'].quantile(0.75)
            sweet_spot_df = df_density[(df_density['popularity'] > high_pop_threshold)]
            
            if len(sweet_spot_df) > 0:
                col_insight1, col_insight2 = st.columns(2)
                with col_insight1:
                    st.metric(f"Sweet Spot {feature_x.capitalize()}", 
                            f"{sweet_spot_df[feature_x].mean():.2f}",
                            f"±{sweet_spot_df[feature_x].std():.2f}")
                with col_insight2:
                    st.metric(f"Sweet Spot {feature_y.capitalize()}", 
                            f"{sweet_spot_df[feature_y].mean():.2f}",
                            f"±{sweet_spot_df[feature_y].std():.2f}")

        # ---------------------------------------------------
        # VIZ 9: Musical Key Strategy
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[8]:
            st.subheader("9. Musical Key & Mode: The Emotional Blueprint")
            st.info("**🔍 Strategic Questions:** Do certain keys resonate better with audiences? Should artists favor major (happy) or minor (sad) keys? Is there a 'golden key' for hits?")
            
            # Map keys to musical notation
            key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
            
            df_filtered['key_name'] = df_filtered['key'].map(key_mapping)
            df_filtered['mode_name'] = df_filtered['mode'].map({0: 'Minor', 1: 'Major'})
            
            key_analysis = st.radio(
                "Analysis Type:",
                ["Distribution", "Popularity by Key", "Emotional Impact"],
                horizontal=True,
                key="key_analysis"
            )
            
            if key_analysis == "Distribution":
                # Count combinations
                key_mode_counts = df_filtered.groupby(['key_name', 'mode_name']).size().reset_index(name='count')
                
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
                # Average popularity by key
                key_popularity = df_filtered.groupby(['key_name', 'mode_name'])['popularity'].mean().reset_index()
                
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
                # Compare valence (happiness) between major and minor
                fig_keys = px.box(
                    df_filtered,
                    x='mode_name',
                    y='valence',
                    color='mode_name',
                    title='Emotional Impact: Valence in Major vs Minor Keys',
                    labels={'mode_name': 'Mode', 'valence': 'Valence (Happiness)'}
                )
            
            st.plotly_chart(fig_keys, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 10: Decade Evolution Deep Dive
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[9]:
            st.subheader("10. Decade Evolution: The Sound of Generations")
            st.info("**🔍 Strategic Questions:** How homogeneous is modern music? Which decades had the most experimental music? Can we identify the 'signature sound' of each generation?")
            
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
            
            # Filter for relevant decades
            df_decades = df_filtered[df_filtered['decade'] >= 1950].copy()
            
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
                variance_by_decade['cv'] = variance_by_decade['std'] / variance_by_decade['mean']
                
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
            
            st.plotly_chart(fig_decade_box, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 11: Genre Popularity & Market Share
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[10]:
            st.subheader("11. Genre Economics: Market Share & Commercial Viability")
            st.info("**🔍 Strategic Questions:** Which genres dominate the market? Are niche genres more loyal? Where should new artists position themselves for maximum impact?")
            
            genre_view = st.radio(
                "View:",
                ["Top Genres by Popularity", "Genre Market Share", "Genre Loyalty Index"],
                horizontal=True,
                key="genre_view"
            )
            
            if genre_view == "Top Genres by Popularity":
                # Get top 20 genres by popularity
                top_genres_pop = df_genres.nlargest(20, 'popularity')[['genres', 'popularity']]
                
                fig_top_genres = px.bar(
                    top_genres_pop,
                    x='popularity',
                    y='genres',
                    orientation='h',
                    color='popularity',
                    color_continuous_scale='Viridis',
                    title='Genre Power Rankings: Commercial Appeal',
                    labels={'genres': 'Genre', 'popularity': 'Average Popularity'}
                )
                fig_top_genres.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                
            elif genre_view == "Genre Market Share":
                # Calculate market share (number of tracks)
                genre_counts = music_data['with_genres']['genres'].value_counts().head(15)
                
                fig_top_genres = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title='Genre Market Share by Track Volume'
                )
                
            else:  # Genre Loyalty Index
                # Calculate consistency (inverse of std deviation)
                genre_loyalty = []
                for genre in df_genres.nlargest(15, 'popularity')['genres']:
                    genre_data = music_data['with_genres'][music_data['with_genres']['genres'] == genre]
                    if len(genre_data) > 10:
                        loyalty = 1 / (genre_data['popularity'].std() + 1)  # +1 to avoid division by zero
                        genre_loyalty.append({'genre': genre[:20], 'loyalty_index': loyalty * 100,
                                            'avg_popularity': genre_data['popularity'].mean()})
                
                loyalty_df = pd.DataFrame(genre_loyalty).sort_values('loyalty_index', ascending=True)
                
                fig_top_genres = px.scatter(
                    loyalty_df,
                    x='avg_popularity',
                    y='loyalty_index',
                    text='genre',
                    title='Genre Loyalty vs Popularity: Finding Your Niche',
                    labels={'loyalty_index': 'Loyalty Index', 'avg_popularity': 'Average Popularity'}
                )
                fig_top_genres.update_traces(textposition='top center')
            
            st.plotly_chart(fig_top_genres, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 12: Tempo-Popularity Sweet Spots
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[11]:
            st.subheader("12. The Tempo Formula: BPM Success Zones")
            st.info("**🔍 Strategic Questions:** Is there an optimal BPM for chart success? Do different decades prefer different tempos? Should producers target specific BPM ranges?")
            
            # Add BPM zones
            df_tempo = df_filtered.copy()
            df_tempo['bpm_zone'] = pd.cut(df_tempo['tempo'], 
                                        bins=[0, 80, 100, 120, 140, 200],
                                        labels=['Slow (<80)', 'Moderate (80-100)', 
                                                'Dance (100-120)', 'Fast (120-140)', 'Very Fast (>140)'])
            
            tempo_analysis = st.radio(
                "Analysis:",
                ["Density Map", "Success Zones", "Evolution"],
                horizontal=True,
                key="tempo_analysis"
            )
            
            if tempo_analysis == "Density Map":
                # Sample data for better performance
                df_tempo_sample = df_tempo.sample(min(10000, len(df_tempo)))
                
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
                # Average popularity by BPM zone
                bpm_success = df_tempo.groupby('bpm_zone')['popularity'].agg(['mean', 'std', 'count']).reset_index()
                
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
                # BPM zones over decades
                tempo_evolution = df_tempo.groupby(['decade', 'bpm_zone']).size().reset_index(name='count')
                tempo_evolution = tempo_evolution[tempo_evolution['decade'] >= 1960]
                
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

        # ---------------------------------------------------
        # VIZ 13: Explicit Content Strategy Over Time
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[12]:
            st.subheader("13. Explicit Content Evolution: Cultural Shifts & Commercial Impact")
            st.info("**🔍 Strategic Questions:** When did explicit content become mainstream? Which genres pioneered explicit content? Is the trend reversing or accelerating?")
            
            explicit_view = st.radio(
                "View:",
                ["Timeline", "By Genre", "Commercial Impact"],
                horizontal=True,
                key="explicit_view"
            )
            
            if explicit_view == "Timeline":
                # Group by year and explicit
                explicit_years = df[df['year'] >= 1960].groupby(['year', 'explicit']).size().reset_index(name='count')
                explicit_years['explicit_label'] = explicit_years['explicit'].map({0: 'Clean', 1: 'Explicit'})
                
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
                # Explicit percentage by genre
                genre_explicit = music_data['with_genres'].copy()
                genre_explicit['explicit'] = genre_explicit.get('explicit', 0)
                
                top_genres_list = df_genres.nlargest(10, 'popularity')['genres'].tolist()
                
                explicit_by_genre = []
                for genre in top_genres_list:
                    genre_data = genre_explicit[genre_explicit['genres'] == genre]
                    if len(genre_data) > 0:
                        explicit_pct = (genre_data['explicit'].sum() / len(genre_data)) * 100
                        explicit_by_genre.append({'genre': genre[:20], 'explicit_percentage': explicit_pct})
                
                explicit_genre_df = pd.DataFrame(explicit_by_genre).sort_values('explicit_percentage')
                
                fig_explicit_years = px.bar(
                    explicit_genre_df,
                    x='explicit_percentage',
                    y='genre',
                    orientation='h',
                    title='Explicit Content by Genre (%)',
                    labels={'genre': 'Genre', 'explicit_percentage': 'Explicit Content (%)'},
                    color='explicit_percentage',
                    color_continuous_scale='Reds'
                )
                
            else:  # Commercial Impact
                # Compare popularity over time
                explicit_impact = df[df['year'] >= 1980].groupby(['year', 'explicit'])['popularity'].mean().reset_index()
                explicit_impact['explicit_label'] = explicit_impact['explicit'].map({0: 'Clean', 1: 'Explicit'})
                
                fig_explicit_years = px.line(
                    explicit_impact,
                    x='year',
                    y='popularity',
                    color='explicit_label',
                    title='Commercial Performance: Clean vs Explicit Over Time',
                    labels={'year': 'Year', 'popularity': 'Average Popularity', 'explicit_label': 'Content Type'},
                    markers=True
                )
            
            st.plotly_chart(fig_explicit_years, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 14: Popularity Lifecycle & Trends
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[13]:
            st.subheader("14. The Popularity Lifecycle: Understanding Music Relevance")
            st.info("**🔍 Strategic Questions:** How long do songs stay relevant? Are older songs experiencing a streaming renaissance? What makes a song 'timeless'?")
            
            popularity_view = st.radio(
                "Analysis:",
                ["Overall Trend", "By Era", "Timeless Features"],
                horizontal=True,
                key="popularity_view"
            )
            
            # Filter for recent years with enough data
            popularity_trend = df_year[df_year['year'] >= 1920][['year', 'popularity']]
            
            if popularity_view == "Overall Trend":
                fig_pop_trend = px.line(
                    popularity_trend,
                    x='year',
                    y='popularity',
                    title='The Streaming Effect: How Digital Platforms Changed Music Popularity',
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
                # Define music eras
                df_eras = df.copy()
                df_eras['era'] = pd.cut(df_eras['year'], 
                                        bins=[0, 1960, 1980, 1990, 2000, 2010, 2025],
                                        labels=['Pre-1960', '1960s-70s', '1980s', '1990s', '2000s', '2010s+'])
                
                era_popularity = df_eras.groupby('era')['popularity'].agg(['mean', 'std']).reset_index()
                
                fig_pop_trend = px.bar(
                    era_popularity,
                    x='era',
                    y='mean',
                    error_y='std',
                    title='Popularity by Musical Era',
                    labels={'era': 'Era', 'mean': 'Average Popularity'},
                    color='mean',
                    color_continuous_scale='Blues'
                )
                
            else:  # Timeless Features
                # Compare features of consistently popular songs vs others
                timeless_threshold = df['popularity'].quantile(0.7)
                df_timeless = df.copy()
                df_timeless['is_timeless'] = df_timeless['popularity'] > timeless_threshold
                
                features_compare = ['energy', 'danceability', 'valence', 'acousticness']
                timeless_features = df_timeless.groupby('is_timeless')[features_compare].mean().T
                timeless_features.columns = ['Regular', 'Timeless']
                
                fig_pop_trend = px.bar(
                    timeless_features,
                    title='The DNA of Timeless Songs',
                    labels={'index': 'Feature', 'value': 'Average Value'},
                    barmode='group'
                )
            
            st.plotly_chart(fig_pop_trend, use_container_width=True)
        

        # ---------------------------------------------------
        # VIZ 15: ARTIST EVOLUTION OVER TIME
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[14]:
            st.subheader("15. Artist Evolution: The Rise and Fall of Music Icons")
            st.info("**🔍 Strategic Questions:** Which artists dominated different eras? How has artist longevity changed? Can we identify 'comeback' artists or one-era wonders?")

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
                        min_value=5,
                        max_value=30,
                        value=15,
                        step=5,
                        key="top_n_timeline"
                    )
                
                # Filter data for reasonable time period
                df_artist_time = df_filtered[df_filtered['year'] >= 1960].copy()
                
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
                
                # Clean artist names (remove brackets and quotes)
                df_artist_time['artist_clean'] = df_artist_time['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                
                # Split artists if multiple (take first artist only for simplicity)
                df_artist_time['artist_clean'] = df_artist_time['artist_clean'].str.split(',').str[0].str.strip()
                
                # Calculate metrics based on choice
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
                
                # Get top artists overall (across all time periods)
                top_artists_overall = artist_metrics.groupby('artist_clean')[metric_col].mean().nlargest(top_n_artists).index.tolist()
                
                # Filter for top artists
                artist_metrics_filtered = artist_metrics[artist_metrics['artist_clean'].isin(top_artists_overall)]
                
                # Create line chart
                fig_artist_timeline = px.line(
                    artist_metrics_filtered,
                    x='time_period',
                    y=metric_col,
                    color='artist_clean',
                    title=f'Top {top_n_artists} Artists: {metric_label} Over Time',
                    labels={'time_period': time_granularity.replace('By ', ''), 
                            metric_col: metric_label, 
                            'artist_clean': 'Artist'},
                    markers=True,
                    template='plotly_dark'
                )
                
                # Update layout for better visibility
                fig_artist_timeline.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3',
                    hovermode='x unified',
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.02
                    ),
                    height=600
                )
                
                # Add range slider for better navigation
                fig_artist_timeline.update_xaxes(rangeslider_visible=True)
                
                st.plotly_chart(fig_artist_timeline, use_container_width=True)
                
                # Show summary statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    most_consistent = artist_metrics_filtered.groupby('artist_clean').size().idxmax()
                    appearances = artist_metrics_filtered.groupby('artist_clean').size().max()
                    st.metric("Most Consistent Artist", most_consistent[:25], f"{appearances} periods")
                with col_stat2:
                    peak_artist = artist_metrics_filtered.loc[artist_metrics_filtered[metric_col].idxmax(), 'artist_clean']
                    peak_value = artist_metrics_filtered[metric_col].max()
                    st.metric(f"Peak {metric_label}", peak_artist[:25], f"{peak_value:.1f}")
                with col_stat3:
                    recent_period = artist_metrics_filtered['time_period'].max()
                    recent_top = artist_metrics_filtered[artist_metrics_filtered['time_period'] == recent_period].nlargest(1, metric_col)['artist_clean'].values[0] if len(artist_metrics_filtered[artist_metrics_filtered['time_period'] == recent_period]) > 0 else "N/A"
                    st.metric("Current Leader", recent_top[:25], f"In {recent_period}")

            elif artist_time_analysis == "Artist Dominance by Decade":
                # Analyze which artists dominated each decade
                decade_selection = st.select_slider(
                    "Select Decade Range:",
                    options=sorted(df[df['decade'] >= 1950]['decade'].unique()),
                    value=(1970, 2020),
                    key="dominance_decade_range"
                )
                
                # Filter data
                df_dominance = df_filtered[(df_filtered['decade'] >= decade_selection[0]) & 
                                        (df_filtered['decade'] <= decade_selection[1])].copy()
                
                # Clean artist names
                df_dominance['artist_clean'] = df_dominance['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                df_dominance['artist_clean'] = df_dominance['artist_clean'].str.split(',').str[0].str.strip()
                
                # Calculate dominance score (combination of track count and average popularity)
                dominance_by_decade = []
                
                for decade in range(decade_selection[0], decade_selection[1] + 10, 10):
                    decade_data = df_dominance[df_dominance['decade'] == decade]
                    if len(decade_data) > 0:
                        # Get top 10 artists for this decade
                        artist_stats = decade_data.groupby('artist_clean').agg({
                            'popularity': ['mean', 'max', 'count']
                        }).round(2)
                        artist_stats.columns = ['avg_popularity', 'max_popularity', 'track_count']
                        
                        # Calculate dominance score
                        artist_stats['dominance_score'] = (
                            artist_stats['avg_popularity'] * 0.5 + 
                            artist_stats['max_popularity'] * 0.3 + 
                            artist_stats['track_count'] * 0.2
                        )
                        
                        # Get top 10 for this decade
                        top_decade_artists = artist_stats.nlargest(10, 'dominance_score')
                        
                        for artist, row in top_decade_artists.iterrows():
                            dominance_by_decade.append({
                                'decade': decade,
                                'artist': artist[:30],
                                'dominance_score': row['dominance_score'],
                                'avg_popularity': row['avg_popularity'],
                                'track_count': row['track_count']
                            })
                
                dominance_df = pd.DataFrame(dominance_by_decade)
                
                # Create heatmap
                if not dominance_df.empty:
                    # Pivot for heatmap
                    heatmap_data = dominance_df.pivot_table(
                        index='artist',
                        columns='decade',
                        values='dominance_score',
                        fill_value=0
                    )
                    
                    # Sort by total dominance
                    heatmap_data['total'] = heatmap_data.sum(axis=1)
                    heatmap_data = heatmap_data.sort_values('total', ascending=False).drop('total', axis=1).head(20)
                    
                    fig_dominance = px.imshow(
                        heatmap_data,
                        title='Artist Dominance Heatmap by Decade',
                        labels=dict(x="Decade", y="Artist", color="Dominance Score"),
                        aspect="auto",
                        color_continuous_scale="Viridis",
                        template='plotly_dark'
                    )
                    
                    fig_dominance.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3',
                        height=700
                    )
                    
                    st.plotly_chart(fig_dominance, use_container_width=True)
                    
                    # Show decade leaders
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
                # Analyze artist career spans and consistency
                st.markdown("### 🏆 Artist Career Longevity & Consistency")
                
                # Minimum tracks threshold
                min_tracks = st.slider(
                    "Minimum tracks to be included:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key="longevity_min_tracks"
                )
                
                # Clean artist names
                df_longevity = df_filtered.copy()
                df_longevity['artist_clean'] = df_longevity['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                df_longevity['artist_clean'] = df_longevity['artist_clean'].str.split(',').str[0].str.strip()
                
                # Calculate longevity metrics
                longevity_stats = df_longevity.groupby('artist_clean').agg({
                    'year': ['min', 'max', 'nunique'],
                    'popularity': ['mean', 'std', 'max'],
                    'name': 'count'  # track count
                }).round(2)
                
                longevity_stats.columns = ['first_year', 'last_year', 'active_years', 
                                        'avg_popularity', 'std_popularity', 'max_popularity', 'track_count']
                
                # Filter by minimum tracks
                longevity_stats = longevity_stats[longevity_stats['track_count'] >= min_tracks]
                
                # Calculate career span and consistency score
                longevity_stats['career_span'] = longevity_stats['last_year'] - longevity_stats['first_year']
                longevity_stats['consistency_score'] = (longevity_stats['avg_popularity'] / 
                                                    (longevity_stats['std_popularity'] + 1)) * (longevity_stats['active_years'] / longevity_stats['career_span'].clip(lower=1))
                
                # Reset index to get artist names as column
                longevity_stats = longevity_stats.reset_index()
                
                # Get top artists by different metrics
                col_long1, col_long2 = st.columns(2)
                
                with col_long1:
                    # Longest careers
                    fig_career_span = px.scatter(
                        longevity_stats.nlargest(30, 'career_span'),
                        x='career_span',
                        y='avg_popularity',
                        size='track_count',
                        color='consistency_score',
                        hover_data=['artist_clean', 'first_year', 'last_year'],
                        title='Longest Career Spans (Top 30)',
                        labels={'career_span': 'Career Span (Years)', 
                            'avg_popularity': 'Average Popularity',
                            'consistency_score': 'Consistency'},
                        template='plotly_dark',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig_career_span.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    
                    st.plotly_chart(fig_career_span, use_container_width=True)
                
                with col_long2:
                    # Most consistent artists
                    fig_consistency = px.bar(
                        longevity_stats.nlargest(15, 'consistency_score'),
                        x='consistency_score',
                        y='artist_clean',
                        orientation='h',
                        color='avg_popularity',
                        title='Most Consistent Artists (Top 15)',
                        labels={'consistency_score': 'Consistency Score', 
                            'artist_clean': 'Artist',
                            'avg_popularity': 'Avg Popularity'},
                        template='plotly_dark',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig_consistency.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3',
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig_consistency, use_container_width=True)
                
                # Career type classification
                st.markdown("### 🎭 Artist Career Types")
                
                # Classify artists
                longevity_stats['career_type'] = 'Standard'
                longevity_stats.loc[(longevity_stats['career_span'] > 20) & 
                                (longevity_stats['avg_popularity'] > 50), 'career_type'] = 'Legend'
                longevity_stats.loc[(longevity_stats['career_span'] < 5) & 
                                (longevity_stats['max_popularity'] > 70), 'career_type'] = 'One-Hit Wonder'
                longevity_stats.loc[(longevity_stats['career_span'] > 10) & 
                                (longevity_stats['consistency_score'] > longevity_stats['consistency_score'].quantile(0.75)), 
                                'career_type'] = 'Steady Performer'
                
                # Show distribution
                career_distribution = longevity_stats['career_type'].value_counts()
                
                col_type1, col_type2, col_type3, col_type4 = st.columns(4)
                
                for idx, (career_type, count) in enumerate(career_distribution.items()):
                    col = [col_type1, col_type2, col_type3, col_type4][idx % 4]
                    with col:
                        emoji = {'Legend': '👑', 'One-Hit Wonder': '🌟', 
                                'Steady Performer': '📊', 'Standard': '🎵'}.get(career_type, '🎵')
                        st.metric(f"{emoji} {career_type}", f"{count} artists", 
                                f"{(count/len(longevity_stats)*100):.1f}%")

            else:  # Rising Stars
                # Identify artists with rapid growth
                st.markdown("### 🚀 Rising Stars & Trending Artists")
                
                # Time window for analysis
                window_years = st.slider(
                    "Analysis window (recent years):",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=5,
                    key="rising_window"
                )
                
                current_year = df_filtered['year'].max()
                cutoff_year = current_year - window_years
                
                # Split data into periods
                df_recent = df_filtered[df_filtered['year'] > cutoff_year].copy()
                df_previous = df_filtered[(df_filtered['year'] <= cutoff_year) & 
                                        (df_filtered['year'] > cutoff_year - window_years)].copy()
                
                # Clean artist names
                for data in [df_recent, df_previous]:
                    data['artist_clean'] = data['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                    data['artist_clean'] = data['artist_clean'].str.split(',').str[0].str.strip()
                
                # Calculate metrics for both periods
                recent_stats = df_recent.groupby('artist_clean').agg({
                    'popularity': 'mean',
                    'name': 'count'
                }).rename(columns={'popularity': 'recent_pop', 'name': 'recent_tracks'})
                
                previous_stats = df_previous.groupby('artist_clean').agg({
                    'popularity': 'mean',
                    'name': 'count'
                }).rename(columns={'popularity': 'previous_pop', 'name': 'previous_tracks'})
                
                # Merge and calculate growth
                growth_analysis = pd.merge(recent_stats, previous_stats, 
                                        left_index=True, right_index=True, how='left')
                growth_analysis = growth_analysis.fillna(0)
                
                # Calculate growth metrics
                growth_analysis['pop_growth'] = ((growth_analysis['recent_pop'] - growth_analysis['previous_pop']) / 
                                                (growth_analysis['previous_pop'] + 1)) * 100
                growth_analysis['track_growth'] = ((growth_analysis['recent_tracks'] - growth_analysis['previous_tracks']) / 
                                                (growth_analysis['previous_tracks'] + 1)) * 100
                growth_analysis['momentum_score'] = (growth_analysis['pop_growth'] * 0.6 + 
                                                    growth_analysis['track_growth'] * 0.4)
                
                # Reset index to get artist names
                growth_analysis = growth_analysis.reset_index()
                
                # Filter for meaningful results (at least 3 recent tracks)
                growth_analysis = growth_analysis[growth_analysis['recent_tracks'] >= 3]
                
                # Categories
                col_rise1, col_rise2 = st.columns(2)
                
                with col_rise1:
                    # Fastest rising
                    rising_stars = growth_analysis.nlargest(15, 'momentum_score')
                    
                    fig_rising = px.scatter(
                        rising_stars,
                        x='previous_pop',
                        y='recent_pop',
                        size='recent_tracks',
                        color='momentum_score',
                        hover_data=['artist_clean'],
                        title='Rising Stars: Previous vs Current Popularity',
                        labels={'previous_pop': f'Popularity ({cutoff_year-window_years}-{cutoff_year})',
                            'recent_pop': f'Popularity ({cutoff_year}-{current_year})',
                            'momentum_score': 'Momentum'},
                        template='plotly_dark',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Add diagonal line (no change)
                    max_val = max(rising_stars['previous_pop'].max(), rising_stars['recent_pop'].max())
                    fig_rising.add_shape(
                        type='line',
                        x0=0, y0=0, x1=max_val, y1=max_val,
                        line=dict(color='gray', dash='dash', width=1)
                    )
                    
                    fig_rising.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    
                    st.plotly_chart(fig_rising, use_container_width=True)
                
                with col_rise2:
                    # New entrants (artists with no previous tracks)
                    new_artists = growth_analysis[growth_analysis['previous_tracks'] == 0].nlargest(10, 'recent_pop')
                    
                    if not new_artists.empty:
                        fig_new = px.bar(
                            new_artists,
                            x='recent_pop',
                            y='artist_clean',
                            orientation='h',
                            color='recent_tracks',
                            title='Breakthrough Artists (New Entrants)',
                            labels={'recent_pop': 'Current Popularity',
                                'artist_clean': 'Artist',
                                'recent_tracks': 'Track Count'},
                            template='plotly_dark',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_new.update_layout(
                            plot_bgcolor='#181818',
                            paper_bgcolor='#181818',
                            font_color='#B3B3B3',
                            yaxis={'categoryorder':'total ascending'}
                        )
                        
                        st.plotly_chart(fig_new, use_container_width=True)
                    else:
                        st.info("No breakthrough artists found in the selected time window.")
                
                    # Top movers summary
                    st.markdown("### 📊 Movement Summary")
                    
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    
                    with col_sum1:
                        biggest_gainer = growth_analysis.nlargest(1, 'pop_growth')
                        if not biggest_gainer.empty:
                            st.metric(
                                "🎯 Biggest Popularity Gain",
                                biggest_gainer['artist_clean'].values[0][:25],
                                f"+{biggest_gainer['pop_growth'].values[0]:.1f}%"
                            )
                    
                    with col_sum2:
                        most_productive = growth_analysis.nlargest(1, 'recent_tracks')
                        if not most_productive.empty:
                            st.metric(
                                "🎵 Most Productive",
                                most_productive['artist_clean'].values[0][:25],
                                f"{most_productive['recent_tracks'].values[0]:.0f} tracks"
                            )
                    
                    with col_sum3:
                        highest_momentum = growth_analysis.nlargest(1, 'momentum_score')
                        if not highest_momentum.empty:
                            st.metric(
                                "🚀 Highest Momentum",
                                highest_momentum['artist_clean'].values[0][:25],
                                f"Score: {highest_momentum['momentum_score'].values[0]:.1f}"
                        )
                               

        # ---------------------------------------------------
        # VIZ 16: SONG TITLE ANALYTICS
        # ---------------------------------------------------
        st.divider()

        with viz_tabs[15]:
            st.subheader("16. The Power of Words: Song Title Analysis")
            st.info("**🔍 Strategic Questions:** Do shorter titles perform better? What words appear most in hit songs? Can title sentiment predict success?")

            import re
            from collections import Counter

            title_analysis_type = st.radio(
                "Analysis Type:",
                ["Title Length vs Popularity", "Most Common Words", "Title Patterns"],
                horizontal=True,
                key="title_analysis"
            )

            # Add title length column
            df_titles = df_filtered.copy()
            df_titles['title_length'] = df_titles['name'].str.len()
            df_titles['title_word_count'] = df_titles['name'].str.split().str.len()

            if title_analysis_type == "Title Length vs Popularity":
                col_title1, col_title2 = st.columns(2)
                
                with col_title1:
                    # Character length analysis
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
                    fig_title_len.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    st.plotly_chart(fig_title_len, use_container_width=True)
                
                with col_title2:
                    # Word count analysis
                    word_bins = pd.cut(df_titles['title_word_count'], 
                                    bins=[0, 1, 2, 3, 4, 10], 
                                    labels=['1 word', '2 words', '3 words', '4 words', '5+ words'])
                    word_popularity = df_titles.groupby(word_bins)['popularity'].mean().reset_index()
                    
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
                    fig_word_count.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    st.plotly_chart(fig_word_count, use_container_width=True)

            elif title_analysis_type == "Most Common Words":
                # Extract and count words from titles
                all_words = ' '.join(df_titles[df_titles['popularity'] > 50]['name'].str.lower()).split()
                # Remove common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'been'}
                filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
                
                word_freq = Counter(filtered_words)
                top_words = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
                
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
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3',
                    height=600,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_words, use_container_width=True)

            else:  # Title Patterns
                # Check for patterns
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
                    if len(data) == 2:
                        pattern_results.append({
                            'Pattern': pattern,
                            'Without': data[False],
                            'With': data[True],
                            'Impact': data[True] - data[False]
                        })
                
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
                fig_patterns.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_patterns, use_container_width=True)
            
        # ---------------------------------------------------
        # VIZ 17: COLLABORATION PATTERNS
        # ---------------------------------------------------
        st.divider()
        
        with viz_tabs[16]:
            st.subheader("17. Collaboration Economy: The Power of Features")
            st.info("**🔍 Strategic Questions:** Do collaborations boost popularity? Which artists collaborate most? What's the optimal number of artists per track?")

            # Analyze collaboration patterns
            df_collab = df_filtered.copy()
            df_collab['artist_count'] = df_collab['artists'].str.count(',') + 1

            collab_view = st.radio(
                "View:",
                ["Collaboration Impact", "Artist Networks", "Optimal Team Size"],
                horizontal=True,
                key="collab_view"
            )

            if collab_view == "Collaboration Impact":
                # Solo vs Collaborations
                df_collab['is_collab'] = df_collab['artist_count'] > 1
                
                col_collab1, col_collab2 = st.columns(2)
                
                with col_collab1:
                    collab_stats = df_collab.groupby('is_collab')['popularity'].agg(['mean', 'std', 'count']).reset_index()
                    collab_stats['is_collab'] = collab_stats['is_collab'].map({False: 'Solo', True: 'Collaboration'})
                    
                    fig_collab_impact = px.bar(
                        collab_stats,
                        x='is_collab',
                        y='mean',
                        error_y='std',
                        title='Solo vs Collaboration Performance',
                        labels={'is_collab': 'Type', 'mean': 'Average Popularity'},
                        color='mean',
                        color_continuous_scale='Viridis',
                        template='plotly_dark'
                    )
                    fig_collab_impact.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    st.plotly_chart(fig_collab_impact, use_container_width=True)
                
                with col_collab2:
                    # Trend over time
                    collab_trend = df_collab[df_collab['year'] >= 1980].groupby('year')['is_collab'].mean() * 100
                    
                    fig_collab_trend = px.line(
                        x=collab_trend.index,
                        y=collab_trend.values,
                        title='Collaboration Trend Over Time',
                        labels={'x': 'Year', 'y': 'Collaboration %'},
                        markers=True,
                        template='plotly_dark'
                    )
                    fig_collab_trend.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3'
                    )
                    st.plotly_chart(fig_collab_trend, use_container_width=True)

            elif collab_view == "Artist Networks":
                # Most collaborative artists
                collab_only = df_collab[df_collab['artist_count'] > 1].copy()
                
                if len(collab_only) > 0:
                    # Count collaborations per artist (simplified)
                    artist_collabs = []
                    for artists_str in collab_only['artists'].head(1000):  # Sample for performance
                        artists_list = [a.strip() for a in artists_str.replace('[','').replace(']','').replace("'",'').split(',')]
                        for artist in artists_list:
                            artist_collabs.append(artist)
                    
                    collab_counts = Counter(artist_collabs)
                    top_collabs = pd.DataFrame(collab_counts.most_common(15), columns=['artist', 'collaborations'])
                    
                    fig_network = px.bar(
                        top_collabs,
                        x='collaborations',
                        y='artist',
                        orientation='h',
                        title='Most Collaborative Artists',
                        labels={'collaborations': 'Number of Collaborations', 'artist': 'Artist'},
                        color='collaborations',
                        color_continuous_scale='Viridis',
                        template='plotly_dark'
                    )
                    fig_network.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3',
                        height=500,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_network, use_container_width=True)

            else:  # Optimal Team Size
                # Popularity by number of artists
                team_size = df_collab.groupby('artist_count')['popularity'].agg(['mean', 'count']).reset_index()
                team_size = team_size[team_size['artist_count'] <= 5]  # Focus on reasonable team sizes
                
                fig_team = px.scatter(
                    team_size,
                    x='artist_count',
                    y='mean',
                    size='count',
                    title='Optimal Team Size for Hit Songs',
                    labels={'artist_count': 'Number of Artists', 'mean': 'Average Popularity', 'count': 'Sample Size'},
                    template='plotly_dark',
                    color='mean',
                    color_continuous_scale='RdYlGn'
                )
                fig_team.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_team, use_container_width=True)
                
                # Show optimal
                optimal = team_size.loc[team_size['mean'].idxmax()]
                st.success(f"**Optimal Team Size:** {int(optimal['artist_count'])} artist(s) with average popularity of {optimal['mean']:.1f}")
           
            st.divider()

    else:       
        # Normal mode - load all visualizations sequentially
        # ---------------------------------------------------
        # VIZ 1: Audio Features Over Time - ALL FEATURES
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("1. Evolution of ALL Audio Features Over Decades")
        st.info("**🔍 Key Question:** How have ALL musical characteristics evolved? Which features show the most dramatic changes, indicating major shifts in music production and consumer preferences?")
        
        # Select which features to display
        all_audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'instrumentalness', 'speechiness', 'liveness', 'loudness']
        
        selected_features_viz1 = st.multiselect(
            "Select Audio Features to Display:",
            options=all_audio_features,
            default=all_audio_features,
            key="features_selector_viz1"
        )
        
        # Toggle for normalized view
        normalize_features = st.checkbox("Normalize features for better comparison", value=False, key="normalize_viz1")
        
        # Prepare data for time series
        features_cols = ['year'] + selected_features_viz1
        features_over_time = df_year[features_cols].copy()
        
        if normalize_features:
            for feat in selected_features_viz1:
                features_over_time[feat] = (features_over_time[feat] - features_over_time[feat].min()) / \
                                        (features_over_time[feat].max() - features_over_time[feat].min())
        
        features_melted = features_over_time.melt(
            id_vars=['year'],
            var_name='Feature',
            value_name='Value'
        )
        
        fig_evolution = px.line(
            features_melted,
            x='year',
            y='Value',
            color='Feature',
            title='Evolution of Audio Features (1920-2020)',
            labels={'Value': 'Feature Value' + (' (Normalized)' if normalize_features else ' (0-1)'), 
                'year': 'Year'},
            template='plotly_dark'
        )
        
        # Update layout for Spotify theme
        fig_evolution.update_layout(
            plot_bgcolor='#181818',
            paper_bgcolor='#181818',
            font_color='#B3B3B3'
        )
        
        # Add annotations for major music eras
        music_eras = [
            (1955, "Rock'n'Roll Era"),
            (1975, "Disco Era"),
            (1985, "MTV Era"),
            (1995, "Hip-Hop Rise"),
            (2005, "Digital/Streaming Era")
        ]
        
        for year, era in music_eras:
            fig_evolution.add_vline(x=year, line_dash="dash", line_color="gray", opacity=0.3)
            fig_evolution.add_annotation(x=year, y=0.9, text=era, showarrow=False, 
                                        textangle=-90, font=dict(size=10, color="gray"))
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # ---------------------------------------------------
        # VIZ 2: Popularity vs Audio Features - ENHANCED
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("2. Correlation Analysis: Popularity vs Audio Features")
        st.info("**🔍 Key Questions:** Which audio feature best predicts commercial success? Are there threshold values that guarantee popularity? How do feature combinations create hit songs?")
        
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
        
        # Sample for better performance
        df_sample = df_filtered.sample(min(5000, len(df_filtered)))
        
        if viz2_type == "Scatter with Trend":
            fig_correlation = px.scatter(
                df_sample,
                x=selected_feature,
                y='popularity',
                title=f'Popularity vs {selected_feature.capitalize()}',
                opacity=0.6,
                trendline='ols',
                labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'}
            )
            
            if add_percentiles:
                # Add percentile lines
                p25 = df_sample[selected_feature].quantile(0.25)
                p75 = df_sample[selected_feature].quantile(0.75)
                fig_correlation.add_vline(x=p25, line_dash="dash", line_color="red", opacity=0.5)
                fig_correlation.add_vline(x=p75, line_dash="dash", line_color="green", opacity=0.5)
                
        elif viz2_type == "Hexbin Density":
            fig_correlation = px.density_heatmap(
                df_sample,
                x=selected_feature,
                y='popularity',
                title=f'Density: Popularity vs {selected_feature.capitalize()}',
                labels={selected_feature: f'{selected_feature.capitalize()} (0-1)', 'popularity': 'Popularity Score'},
                nbinsx=30,
                nbinsy=20
            )
            
        else:  # Box Plot by Bins
            # Create bins for the feature
            df_sample['feature_bin'] = pd.cut(df_sample[selected_feature], bins=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            fig_correlation = px.box(
                df_sample,
                x='feature_bin',
                y='popularity',
                title=f'Popularity Distribution by {selected_feature.capitalize()} Levels',
                labels={'feature_bin': f'{selected_feature.capitalize()} Level', 'popularity': 'Popularity Score'}
            )
        
        st.plotly_chart(fig_correlation, use_container_width=True)
        
        # Show correlation coefficient
        correlation = df_filtered[selected_feature].corr(df_filtered['popularity'])
        st.metric(f"Correlation Coefficient", f"{correlation:.3f}", 
                delta=f"{'Positive' if correlation > 0 else 'Negative'} correlation")

        # ---------------------------------------------------
        # VIZ 3: Genre Analysis - ENHANCED
        # ---------------------------------------------------
        st.divider()

        st.subheader("3. Genre DNA: Audio Feature Signatures")
        st.info("**🔍 Strategic Questions:** What is the unique 'DNA' of each genre? Which genres are converging in style? How can artists differentiate within saturated genres?")
        
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
                ["Box Plot", "Radar Chart"], # <-- ALTERAÇÃO: Removido "Violin Plot"
                horizontal=True,
                key="genre_viz_type"
            )
        
        # Get top genres
        top_genres = df_genres.nlargest(15, 'popularity')
        
        if genre_viz_type == "Box Plot":
            fig_genre = px.box(
                top_genres,
                x='genres',
                y=genre_feature,
                color='genres',
                title=f'{genre_feature.capitalize()} Distribution Across Top 15 Genres',
                labels={'genres': 'Genre', genre_feature: f'{genre_feature.capitalize()}'}
            )
            fig_genre.update_layout(showlegend=False, xaxis_tickangle=-45)
            
        # <-- ALTERAÇÃO: O bloco "elif" para Violin Plot foi completamente removido
            
        else:  # Radar Chart (Este "else" agora é ativado quando "Radar Chart" é selecionado)
            # Select top 8 genres for radar chart
            import plotly.graph_objects as go
            
            top_8_genres = df_genres.nlargest(8, 'popularity')
            features_for_radar = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
            
            fig_genre = go.Figure()
            
            for _, genre_row in top_8_genres.iterrows():
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
                title="Genre DNA: Audio Feature Signatures"
            )
        
        st.plotly_chart(fig_genre, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 4: Explicit Content Strategy Analysis
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("4. Explicit Content: Commercial Strategy Analysis")
        st.info("**🔍 Strategic Questions:** Is explicit content a successful commercial strategy? Which genres benefit most from explicit content? Should artists release both clean and explicit versions?")
        
        # Create explicit label
        df_filtered['explicit_label'] = df_filtered['explicit'].map({0: 'Clean', 1: 'Explicit'})
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Popularity comparison
            fig_explicit = px.violin(
                df_filtered[df_filtered['popularity'] > 0],
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
                df_filtered,
                x='explicit_label',
                y=explicit_feature,
                color='explicit_label',
                title=f'{explicit_feature.capitalize()} by Content Type',
                labels={'explicit_label': 'Content Type', explicit_feature: explicit_feature.capitalize()}
            )
            st.plotly_chart(fig_explicit_feat, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 5: Feature Correlation & Clustering
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("5. Feature Relationships: The Music Formula")
        st.info("**🔍 Strategic Questions:** Which features must be balanced together? Can we identify 'formula' combinations for different popularity levels? What trade-offs do producers make?")
        
        # Toggle between correlation matrix and cluster analysis
        analysis_type = st.radio(
            "Analysis Type:",
            ["Correlation Matrix", "Feature Pairs Analysis", "Success Formula"],
            horizontal=True,
            key="analysis_type_viz5"
        )
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'popularity']
        
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
            # Show strongest correlations
            correlation_matrix = df_filtered[audio_features].corr()
            
            # Get correlations with popularity
            pop_corr = correlation_matrix['popularity'].drop('popularity').sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Correlations with Popularity:**")
                for feat, corr in pop_corr.head(3).items():
                    st.write(f"• {feat}: {corr:.3f}")
            
            with col2:
                st.markdown("**Top Negative Correlations with Popularity:**")
                for feat, corr in pop_corr.tail(3).items():
                    st.write(f"• {feat}: {corr:.3f}")
                    
        else:  # Success Formula
            # Segment songs by popularity
            df_success = df_filtered.copy()
            df_success['success_level'] = pd.cut(df_success['popularity'], 
                                                bins=[0, 30, 60, 100],
                                                labels=['Low', 'Medium', 'High'])
            
            # Calculate mean features by success level
            success_formula = df_success.groupby('success_level')[audio_features[:-1]].mean()
            
            fig_formula = px.bar(
                success_formula.T,
                title="The Success Formula: Average Features by Popularity Level",
                labels={'index': 'Audio Feature', 'value': 'Average Value'},
                barmode='group'
            )
            st.plotly_chart(fig_formula, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 6: Temporal Strategy Analysis
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("6. Temporal Trends: Predicting the Future of Music")
        st.info("**🔍 Strategic Questions:** Can we predict the next trend? Are songs converging to a standard formula? What features are becoming obsolete?")
        
        col_tempo1, col_tempo2 = st.columns(2)
        
        # Convert duration to minutes BEFORE using it
        df_year['duration_min'] = df_year['duration_ms'] / 60000
        
        with col_tempo1:
            # Add trend prediction toggle
            show_prediction = st.checkbox("Show trend projection", value=False, key="prediction_duration")
            
            fig_duration = px.line(
                df_year[df_year['year'] >= 1960],
                x='year',
                y='duration_min',
                title='Song Duration: The Attention Span Crisis',
                labels={'duration_min': 'Duration (minutes)', 'year': 'Year'}
            )
            
            if show_prediction:
                # Simple linear projection
                try:
                    from scipy import stats
                    recent_years = df_year[df_year['year'] >= 2000].copy()
                    
                    # Make sure duration_min is calculated for recent_years
                    if 'duration_min' not in recent_years.columns:
                        recent_years['duration_min'] = recent_years['duration_ms'] / 60000
                    
                    # Check if we have valid data
                    if len(recent_years) > 0 and not recent_years['duration_min'].isna().all():
                        slope, intercept, _, _, _ = stats.linregress(recent_years['year'], recent_years['duration_min'])
                        future_years = list(range(2020, 2031))
                        future_duration = [slope * year + intercept for year in future_years]
                        fig_duration.add_scatter(x=future_years, y=future_duration, mode='lines', 
                                                name='Projection', line=dict(dash='dash', color='red'))
                except:
                    pass  # If scipy is not available or calculation fails, just skip projection
            
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col_tempo2:
                        
            fig_tempo = px.line(
                df_year[df_year['year'] >= 1960],
                x='year',
                y='tempo',
                title='Tempo Evolution: The BPM Arms Race',
                labels={'tempo': 'Tempo (BPM)', 'year': 'Year'}
            )
            st.plotly_chart(fig_tempo, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 7: Artist Success Patterns
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("7. Artist Success Patterns: One-Hit Wonder vs Consistency")
        st.info("**🔍 Strategic Questions:** What separates consistent hitmakers from one-hit wonders? Is it better to release many average songs or few excellent ones? How do successful artists maintain their sound signature?")
        
        # Enhanced artist analysis
        artist_strategy = st.radio(
            "Analysis Focus:",
            ["Top 50 Artists Overview", "Consistency Analysis", "Feature Signature (Top 5)"],
            horizontal=True,
            key="artist_strategy"
        )
        
        top_artists = df_artist.nlargest(50, 'popularity')
        
        if artist_strategy == "Top 50 Artists Overview":
            fig_artists = px.scatter(
                top_artists,
                x='count',
                y='popularity',
                size='energy',
                color='valence',
                hover_data=['artists'],
                title='Artist Strategy: Volume vs Quality',
                labels={'count': 'Number of Tracks', 'popularity': 'Average Popularity', 'valence': 'Valence'}
            )
            
            # Add quadrant lines
            median_count = top_artists['count'].median()
            median_pop = top_artists['popularity'].median()
            fig_artists.add_hline(y=median_pop, line_dash="dash", line_color="gray", opacity=0.5)
            fig_artists.add_vline(x=median_count, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant labels
            fig_artists.add_annotation(x=median_count*0.3, y=median_pop*1.2, text="Quality over Quantity", 
                                    showarrow=False, font=dict(color="green"))
            fig_artists.add_annotation(x=median_count*1.7, y=median_pop*1.2, text="Consistent Hitmakers", 
                                    showarrow=False, font=dict(color="#89ccff"))
            
        elif artist_strategy == "Consistency Analysis":
            # Calculate coefficient of variation (std/mean) for each artist
            artist_consistency = []
            
            for artist in top_artists['artists'].head(50):
                artist_songs = df[df['artists'].str.contains(artist, na=False, case=True)]['popularity']
                # --- FIM DA CORREÇÃO ---
                
                if len(artist_songs) > 1:
                    cv = artist_songs.std() / artist_songs.mean() if artist_songs.mean() > 0 else 0
                    artist_consistency.append({'artist': artist[:20], 'consistency': 1 - cv, 
                                            'avg_popularity': artist_songs.mean()})
            
            consistency_df = pd.DataFrame(artist_consistency)
            
            # A verificação 'if empty' ainda é útil, caso alguns artistas 
            # ainda não sejam encontrados.
            if consistency_df.empty:
                st.info("Não foi possível calcular a consistência dos artistas. (Nenhuma correspondência de artista com >1 música encontrada nos dados brutos).")
                import plotly.graph_objects as go
                fig_artists = go.Figure()
                fig_artists.update_layout(title='Artist Consistency Score (No data to display)')
            else:
                # O DataFrame é válido, então podemos criar o gráfico
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
                        
        else:  # Feature Signature
            # Compare top 5 artists' average features
            import plotly.graph_objects as go
            
            top_5_artists = top_artists.head(5)
            features_for_signature = ['energy', 'danceability', 'valence', 'acousticness', 'speechiness']
            
            fig_artists = go.Figure()
            
            for _, artist_row in top_5_artists.iterrows():
                values = [artist_row[feat] for feat in features_for_signature]
                fig_artists.add_trace(go.Scatterpolar(
                    r=values,
                    theta=features_for_signature,
                    fill='toself',
                    name=artist_row['artists'][:20]
                ))
            
            fig_artists.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Artist Sound Signatures: What Makes Them Unique"
            )
        
        st.plotly_chart(fig_artists, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 8: Interactive Feature Explorer - ENHANCED
        # ---------------------------------------------------
        st.divider()

        st.subheader("8. Feature Interaction Explorer: Finding the Sweet Spot")
        st.info("**🔍 Strategic Questions:** What feature combinations create viral hits? Are there 'dead zones' to avoid? How do different eras prefer different feature combinations?")
        
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
                index=0,
                key="density_y_feature"
            )
        
        with col_exp3:
            viz_style = st.radio(
                "Style:",
                ["Density Heatmap", "Scatter with Size", "Contour Plot"],
                key="viz_style_8"
            )
        
        # Add decade filter for this specific viz
        decade_filter_8 = st.select_slider(
            "Filter by Decade Range:",
            options=sorted(df[df['decade'] >= 1950]['decade'].unique()),
            value=(1990, 2020),
            key="decade_filter_8"
        )
        
        # Filter data
        df_density = df_filtered[(df_filtered['decade'] >= decade_filter_8[0]) & 
                                (df_filtered['decade'] <= decade_filter_8[1])]
        df_density_sample = df_density.sample(min(10000, len(df_density)))
        
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
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Show insights
        high_pop_threshold = df_density['popularity'].quantile(0.75)
        sweet_spot_df = df_density[(df_density['popularity'] > high_pop_threshold)]
        
        if len(sweet_spot_df) > 0:
            col_insight1, col_insight2 = st.columns(2)
            with col_insight1:
                st.metric(f"Sweet Spot {feature_x.capitalize()}", 
                        f"{sweet_spot_df[feature_x].mean():.2f}",
                        f"±{sweet_spot_df[feature_x].std():.2f}")
            with col_insight2:
                st.metric(f"Sweet Spot {feature_y.capitalize()}", 
                        f"{sweet_spot_df[feature_y].mean():.2f}",
                        f"±{sweet_spot_df[feature_y].std():.2f}")

        # ---------------------------------------------------
        # VIZ 9: Musical Key Strategy
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("9. Musical Key & Mode: The Emotional Blueprint")
        st.info("**🔍 Strategic Questions:** Do certain keys resonate better with audiences? Should artists favor major (happy) or minor (sad) keys? Is there a 'golden key' for hits?")
        
        # Map keys to musical notation
        key_mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
        
        df_filtered['key_name'] = df_filtered['key'].map(key_mapping)
        df_filtered['mode_name'] = df_filtered['mode'].map({0: 'Minor', 1: 'Major'})
        
        key_analysis = st.radio(
            "Analysis Type:",
            ["Distribution", "Popularity by Key", "Emotional Impact"],
            horizontal=True,
            key="key_analysis"
        )
        
        if key_analysis == "Distribution":
            # Count combinations
            key_mode_counts = df_filtered.groupby(['key_name', 'mode_name']).size().reset_index(name='count')
            
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
            # Average popularity by key
            key_popularity = df_filtered.groupby(['key_name', 'mode_name'])['popularity'].mean().reset_index()
            
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
            # Compare valence (happiness) between major and minor
            fig_keys = px.box(
                df_filtered,
                x='mode_name',
                y='valence',
                color='mode_name',
                title='Emotional Impact: Valence in Major vs Minor Keys',
                labels={'mode_name': 'Mode', 'valence': 'Valence (Happiness)'}
            )
        
        st.plotly_chart(fig_keys, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 10: Decade Evolution Deep Dive
        # ---------------------------------------------------
        st.divider()

        st.subheader("10. Decade Evolution: The Sound of Generations")
        st.info("**🔍 Strategic Questions:** How homogeneous is modern music? Which decades had the most experimental music? Can we identify the 'signature sound' of each generation?")
        
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
        
        # Filter for relevant decades
        df_decades = df_filtered[df_filtered['decade'] >= 1950].copy()
        
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
            variance_by_decade['cv'] = variance_by_decade['std'] / variance_by_decade['mean']
            
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
        
        st.plotly_chart(fig_decade_box, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 11: Genre Popularity & Market Share
        # ---------------------------------------------------
        st.divider()

        st.subheader("11. Genre Economics: Market Share & Commercial Viability")
        st.info("**🔍 Strategic Questions:** Which genres dominate the market? Are niche genres more loyal? Where should new artists position themselves for maximum impact?")
        
        genre_view = st.radio(
            "View:",
            ["Top Genres by Popularity", "Genre Market Share", "Genre Loyalty Index"],
            horizontal=True,
            key="genre_view"
        )
        
        if genre_view == "Top Genres by Popularity":
            # Get top 20 genres by popularity
            top_genres_pop = df_genres.nlargest(20, 'popularity')[['genres', 'popularity']]
            
            fig_top_genres = px.bar(
                top_genres_pop,
                x='popularity',
                y='genres',
                orientation='h',
                color='popularity',
                color_continuous_scale='Viridis',
                title='Genre Power Rankings: Commercial Appeal',
                labels={'genres': 'Genre', 'popularity': 'Average Popularity'}
            )
            fig_top_genres.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            
        elif genre_view == "Genre Market Share":
            # Calculate market share (number of tracks)
            genre_counts = music_data['with_genres']['genres'].value_counts().head(15)
            
            fig_top_genres = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title='Genre Market Share by Track Volume'
            )
            
        else:  # Genre Loyalty Index
            # Calculate consistency (inverse of std deviation)
            genre_loyalty = []
            for genre in df_genres.nlargest(15, 'popularity')['genres']:
                genre_data = music_data['with_genres'][music_data['with_genres']['genres'] == genre]
                if len(genre_data) > 10:
                    loyalty = 1 / (genre_data['popularity'].std() + 1)  # +1 to avoid division by zero
                    genre_loyalty.append({'genre': genre[:20], 'loyalty_index': loyalty * 100,
                                        'avg_popularity': genre_data['popularity'].mean()})
            
            loyalty_df = pd.DataFrame(genre_loyalty).sort_values('loyalty_index', ascending=True)
            
            fig_top_genres = px.scatter(
                loyalty_df,
                x='avg_popularity',
                y='loyalty_index',
                text='genre',
                title='Genre Loyalty vs Popularity: Finding Your Niche',
                labels={'loyalty_index': 'Loyalty Index', 'avg_popularity': 'Average Popularity'}
            )
            fig_top_genres.update_traces(textposition='top center')
        
        st.plotly_chart(fig_top_genres, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 12: Tempo-Popularity Sweet Spots
        # ---------------------------------------------------
        st.divider()

        st.subheader("12. The Tempo Formula: BPM Success Zones")
        st.info("**🔍 Strategic Questions:** Is there an optimal BPM for chart success? Do different decades prefer different tempos? Should producers target specific BPM ranges?")
        
        # Add BPM zones
        df_tempo = df_filtered.copy()
        df_tempo['bpm_zone'] = pd.cut(df_tempo['tempo'], 
                                    bins=[0, 80, 100, 120, 140, 200],
                                    labels=['Slow (<80)', 'Moderate (80-100)', 
                                            'Dance (100-120)', 'Fast (120-140)', 'Very Fast (>140)'])
        
        tempo_analysis = st.radio(
            "Analysis:",
            ["Density Map", "Success Zones", "Evolution"],
            horizontal=True,
            key="tempo_analysis"
        )
        
        if tempo_analysis == "Density Map":
            # Sample data for better performance
            df_tempo_sample = df_tempo.sample(min(10000, len(df_tempo)))
            
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
            # Average popularity by BPM zone
            bpm_success = df_tempo.groupby('bpm_zone')['popularity'].agg(['mean', 'std', 'count']).reset_index()
            
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
            # BPM zones over decades
            tempo_evolution = df_tempo.groupby(['decade', 'bpm_zone']).size().reset_index(name='count')
            tempo_evolution = tempo_evolution[tempo_evolution['decade'] >= 1960]
            
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

        # ---------------------------------------------------
        # VIZ 13: Explicit Content Strategy Over Time
        # ---------------------------------------------------
        st.divider()

        st.subheader("13. Explicit Content Evolution: Cultural Shifts & Commercial Impact")
        st.info("**🔍 Strategic Questions:** When did explicit content become mainstream? Which genres pioneered explicit content? Is the trend reversing or accelerating?")
        
        explicit_view = st.radio(
            "View:",
            ["Timeline", "By Genre", "Commercial Impact"],
            horizontal=True,
            key="explicit_view"
        )
        
        if explicit_view == "Timeline":
            # Group by year and explicit
            explicit_years = df[df['year'] >= 1960].groupby(['year', 'explicit']).size().reset_index(name='count')
            explicit_years['explicit_label'] = explicit_years['explicit'].map({0: 'Clean', 1: 'Explicit'})
            
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
            # Explicit percentage by genre
            genre_explicit = music_data['with_genres'].copy()
            genre_explicit['explicit'] = genre_explicit.get('explicit', 0)
            
            top_genres_list = df_genres.nlargest(10, 'popularity')['genres'].tolist()
            
            explicit_by_genre = []
            for genre in top_genres_list:
                genre_data = genre_explicit[genre_explicit['genres'] == genre]
                if len(genre_data) > 0:
                    explicit_pct = (genre_data['explicit'].sum() / len(genre_data)) * 100
                    explicit_by_genre.append({'genre': genre[:20], 'explicit_percentage': explicit_pct})
            
            explicit_genre_df = pd.DataFrame(explicit_by_genre).sort_values('explicit_percentage')
            
            fig_explicit_years = px.bar(
                explicit_genre_df,
                x='explicit_percentage',
                y='genre',
                orientation='h',
                title='Explicit Content by Genre (%)',
                labels={'genre': 'Genre', 'explicit_percentage': 'Explicit Content (%)'},
                color='explicit_percentage',
                color_continuous_scale='Reds'
            )
            
        else:  # Commercial Impact
            # Compare popularity over time
            explicit_impact = df[df['year'] >= 1980].groupby(['year', 'explicit'])['popularity'].mean().reset_index()
            explicit_impact['explicit_label'] = explicit_impact['explicit'].map({0: 'Clean', 1: 'Explicit'})
            
            fig_explicit_years = px.line(
                explicit_impact,
                x='year',
                y='popularity',
                color='explicit_label',
                title='Commercial Performance: Clean vs Explicit Over Time',
                labels={'year': 'Year', 'popularity': 'Average Popularity', 'explicit_label': 'Content Type'},
                markers=True
            )
        
        st.plotly_chart(fig_explicit_years, use_container_width=True)

        # ---------------------------------------------------
        # VIZ 14: Popularity Lifecycle & Trends
        # ---------------------------------------------------
        st.divider()

        st.subheader("14. The Popularity Lifecycle: Understanding Music Relevance")
        st.info("**🔍 Strategic Questions:** How long do songs stay relevant? Are older songs experiencing a streaming renaissance? What makes a song 'timeless'?")
        
        popularity_view = st.radio(
            "Analysis:",
            ["Overall Trend", "By Era", "Timeless Features"],
            horizontal=True,
            key="popularity_view"
        )
        
        # Filter for recent years with enough data
        popularity_trend = df_year[df_year['year'] >= 1920][['year', 'popularity']]
        
        if popularity_view == "Overall Trend":
            fig_pop_trend = px.line(
                popularity_trend,
                x='year',
                y='popularity',
                title='The Streaming Effect: How Digital Platforms Changed Music Popularity',
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
            # Define music eras
            df_eras = df.copy()
            df_eras['era'] = pd.cut(df_eras['year'], 
                                    bins=[0, 1960, 1980, 1990, 2000, 2010, 2025],
                                    labels=['Pre-1960', '1960s-70s', '1980s', '1990s', '2000s', '2010s+'])
            
            era_popularity = df_eras.groupby('era')['popularity'].agg(['mean', 'std']).reset_index()
            
            fig_pop_trend = px.bar(
                era_popularity,
                x='era',
                y='mean',
                error_y='std',
                title='Popularity by Musical Era',
                labels={'era': 'Era', 'mean': 'Average Popularity'},
                color='mean',
                color_continuous_scale='Blues'
            )
            
        else:  # Timeless Features
            # Compare features of consistently popular songs vs others
            timeless_threshold = df['popularity'].quantile(0.7)
            df_timeless = df.copy()
            df_timeless['is_timeless'] = df_timeless['popularity'] > timeless_threshold
            
            features_compare = ['energy', 'danceability', 'valence', 'acousticness']
            timeless_features = df_timeless.groupby('is_timeless')[features_compare].mean().T
            timeless_features.columns = ['Regular', 'Timeless']
            
            fig_pop_trend = px.bar(
                timeless_features,
                title='The DNA of Timeless Songs',
                labels={'index': 'Feature', 'value': 'Average Value'},
                barmode='group'
            )
        
        st.plotly_chart(fig_pop_trend, use_container_width=True)
        

        # ---------------------------------------------------
        # VIZ 15: ARTIST EVOLUTION OVER TIME
        # ---------------------------------------------------
        st.divider()

        st.subheader("15. Artist Evolution: The Rise and Fall of Music Icons")
        st.info("**🔍 Strategic Questions:** Which artists dominated different eras? How has artist longevity changed? Can we identify 'comeback' artists or one-era wonders?")

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
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=5,
                    key="top_n_timeline"
                )
            
            # Filter data for reasonable time period
            df_artist_time = df_filtered[df_filtered['year'] >= 1960].copy()
            
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
            
            # Clean artist names (remove brackets and quotes)
            df_artist_time['artist_clean'] = df_artist_time['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            
            # Split artists if multiple (take first artist only for simplicity)
            df_artist_time['artist_clean'] = df_artist_time['artist_clean'].str.split(',').str[0].str.strip()
            
            # Calculate metrics based on choice
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
            
            # Get top artists overall (across all time periods)
            top_artists_overall = artist_metrics.groupby('artist_clean')[metric_col].mean().nlargest(top_n_artists).index.tolist()
            
            # Filter for top artists
            artist_metrics_filtered = artist_metrics[artist_metrics['artist_clean'].isin(top_artists_overall)]
            
            # Create line chart
            fig_artist_timeline = px.line(
                artist_metrics_filtered,
                x='time_period',
                y=metric_col,
                color='artist_clean',
                title=f'Top {top_n_artists} Artists: {metric_label} Over Time',
                labels={'time_period': time_granularity.replace('By ', ''), 
                        metric_col: metric_label, 
                        'artist_clean': 'Artist'},
                markers=True,
                template='plotly_dark'
            )
            
            # Update layout for better visibility
            fig_artist_timeline.update_layout(
                plot_bgcolor='#181818',
                paper_bgcolor='#181818',
                font_color='#B3B3B3',
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02
                ),
                height=600
            )
            
            # Add range slider for better navigation
            fig_artist_timeline.update_xaxes(rangeslider_visible=True)
            
            st.plotly_chart(fig_artist_timeline, use_container_width=True)
            
            # Show summary statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                most_consistent = artist_metrics_filtered.groupby('artist_clean').size().idxmax()
                appearances = artist_metrics_filtered.groupby('artist_clean').size().max()
                st.metric("Most Consistent Artist", most_consistent[:25], f"{appearances} periods")
            with col_stat2:
                peak_artist = artist_metrics_filtered.loc[artist_metrics_filtered[metric_col].idxmax(), 'artist_clean']
                peak_value = artist_metrics_filtered[metric_col].max()
                st.metric(f"Peak {metric_label}", peak_artist[:25], f"{peak_value:.1f}")
            with col_stat3:
                recent_period = artist_metrics_filtered['time_period'].max()
                recent_top = artist_metrics_filtered[artist_metrics_filtered['time_period'] == recent_period].nlargest(1, metric_col)['artist_clean'].values[0] if len(artist_metrics_filtered[artist_metrics_filtered['time_period'] == recent_period]) > 0 else "N/A"
                st.metric("Current Leader", recent_top[:25], f"In {recent_period}")

        elif artist_time_analysis == "Artist Dominance by Decade":
            # Analyze which artists dominated each decade
            decade_selection = st.select_slider(
                "Select Decade Range:",
                options=sorted(df[df['decade'] >= 1950]['decade'].unique()),
                value=(1970, 2020),
                key="dominance_decade_range"
            )
            
            # Filter data
            df_dominance = df_filtered[(df_filtered['decade'] >= decade_selection[0]) & 
                                    (df_filtered['decade'] <= decade_selection[1])].copy()
            
            # Clean artist names
            df_dominance['artist_clean'] = df_dominance['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            df_dominance['artist_clean'] = df_dominance['artist_clean'].str.split(',').str[0].str.strip()
            
            # Calculate dominance score (combination of track count and average popularity)
            dominance_by_decade = []
            
            for decade in range(decade_selection[0], decade_selection[1] + 10, 10):
                decade_data = df_dominance[df_dominance['decade'] == decade]
                if len(decade_data) > 0:
                    # Get top 10 artists for this decade
                    artist_stats = decade_data.groupby('artist_clean').agg({
                        'popularity': ['mean', 'max', 'count']
                    }).round(2)
                    artist_stats.columns = ['avg_popularity', 'max_popularity', 'track_count']
                    
                    # Calculate dominance score
                    artist_stats['dominance_score'] = (
                        artist_stats['avg_popularity'] * 0.5 + 
                        artist_stats['max_popularity'] * 0.3 + 
                        artist_stats['track_count'] * 0.2
                    )
                    
                    # Get top 10 for this decade
                    top_decade_artists = artist_stats.nlargest(10, 'dominance_score')
                    
                    for artist, row in top_decade_artists.iterrows():
                        dominance_by_decade.append({
                            'decade': decade,
                            'artist': artist[:30],
                            'dominance_score': row['dominance_score'],
                            'avg_popularity': row['avg_popularity'],
                            'track_count': row['track_count']
                        })
            
            dominance_df = pd.DataFrame(dominance_by_decade)
            
            # Create heatmap
            if not dominance_df.empty:
                # Pivot for heatmap
                heatmap_data = dominance_df.pivot_table(
                    index='artist',
                    columns='decade',
                    values='dominance_score',
                    fill_value=0
                )
                
                # Sort by total dominance
                heatmap_data['total'] = heatmap_data.sum(axis=1)
                heatmap_data = heatmap_data.sort_values('total', ascending=False).drop('total', axis=1).head(20)
                
                fig_dominance = px.imshow(
                    heatmap_data,
                    title='Artist Dominance Heatmap by Decade',
                    labels=dict(x="Decade", y="Artist", color="Dominance Score"),
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    template='plotly_dark'
                )
                
                fig_dominance.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3',
                    height=700
                )
                
                st.plotly_chart(fig_dominance, use_container_width=True)
                
                # Show decade leaders
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
            # Analyze artist career spans and consistency
            st.markdown("### 🏆 Artist Career Longevity & Consistency")
            
            # Minimum tracks threshold
            min_tracks = st.slider(
                "Minimum tracks to be included:",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="longevity_min_tracks"
            )
            
            # Clean artist names
            df_longevity = df_filtered.copy()
            df_longevity['artist_clean'] = df_longevity['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
            df_longevity['artist_clean'] = df_longevity['artist_clean'].str.split(',').str[0].str.strip()
            
            # Calculate longevity metrics
            longevity_stats = df_longevity.groupby('artist_clean').agg({
                'year': ['min', 'max', 'nunique'],
                'popularity': ['mean', 'std', 'max'],
                'name': 'count'  # track count
            }).round(2)
            
            longevity_stats.columns = ['first_year', 'last_year', 'active_years', 
                                    'avg_popularity', 'std_popularity', 'max_popularity', 'track_count']
            
            # Filter by minimum tracks
            longevity_stats = longevity_stats[longevity_stats['track_count'] >= min_tracks]
            
            # Calculate career span and consistency score
            longevity_stats['career_span'] = longevity_stats['last_year'] - longevity_stats['first_year']
            longevity_stats['consistency_score'] = (longevity_stats['avg_popularity'] / 
                                                (longevity_stats['std_popularity'] + 1)) * (longevity_stats['active_years'] / longevity_stats['career_span'].clip(lower=1))
            
            # Reset index to get artist names as column
            longevity_stats = longevity_stats.reset_index()
            
            # Get top artists by different metrics
            col_long1, col_long2 = st.columns(2)
            
            with col_long1:
                # Longest careers
                fig_career_span = px.scatter(
                    longevity_stats.nlargest(30, 'career_span'),
                    x='career_span',
                    y='avg_popularity',
                    size='track_count',
                    color='consistency_score',
                    hover_data=['artist_clean', 'first_year', 'last_year'],
                    title='Longest Career Spans (Top 30)',
                    labels={'career_span': 'Career Span (Years)', 
                        'avg_popularity': 'Average Popularity',
                        'consistency_score': 'Consistency'},
                    template='plotly_dark',
                    color_continuous_scale='Viridis'
                )
                
                fig_career_span.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                
                st.plotly_chart(fig_career_span, use_container_width=True)
            
            with col_long2:
                # Most consistent artists
                fig_consistency = px.bar(
                    longevity_stats.nlargest(15, 'consistency_score'),
                    x='consistency_score',
                    y='artist_clean',
                    orientation='h',
                    color='avg_popularity',
                    title='Most Consistent Artists (Top 15)',
                    labels={'consistency_score': 'Consistency Score', 
                        'artist_clean': 'Artist',
                        'avg_popularity': 'Avg Popularity'},
                    template='plotly_dark',
                    color_continuous_scale='Viridis'
                )
                
                fig_consistency.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3',
                    yaxis={'categoryorder':'total ascending'}
                )
                
                st.plotly_chart(fig_consistency, use_container_width=True)
            
            # Career type classification
            st.markdown("### 🎭 Artist Career Types")
            
            # Classify artists
            longevity_stats['career_type'] = 'Standard'
            longevity_stats.loc[(longevity_stats['career_span'] > 20) & 
                            (longevity_stats['avg_popularity'] > 50), 'career_type'] = 'Legend'
            longevity_stats.loc[(longevity_stats['career_span'] < 5) & 
                            (longevity_stats['max_popularity'] > 70), 'career_type'] = 'One-Hit Wonder'
            longevity_stats.loc[(longevity_stats['career_span'] > 10) & 
                            (longevity_stats['consistency_score'] > longevity_stats['consistency_score'].quantile(0.75)), 
                            'career_type'] = 'Steady Performer'
            
            # Show distribution
            career_distribution = longevity_stats['career_type'].value_counts()
            
            col_type1, col_type2, col_type3, col_type4 = st.columns(4)
            
            for idx, (career_type, count) in enumerate(career_distribution.items()):
                col = [col_type1, col_type2, col_type3, col_type4][idx % 4]
                with col:
                    emoji = {'Legend': '👑', 'One-Hit Wonder': '🌟', 
                            'Steady Performer': '📊', 'Standard': '🎵'}.get(career_type, '🎵')
                    st.metric(f"{emoji} {career_type}", f"{count} artists", 
                            f"{(count/len(longevity_stats)*100):.1f}%")

        else:  # Rising Stars
            # Identify artists with rapid growth
            st.markdown("### 🚀 Rising Stars & Trending Artists")
            
            # Time window for analysis
            window_years = st.slider(
                "Analysis window (recent years):",
                min_value=5,
                max_value=20,
                value=10,
                step=5,
                key="rising_window"
            )
            
            current_year = df_filtered['year'].max()
            cutoff_year = current_year - window_years
            
            # Split data into periods
            df_recent = df_filtered[df_filtered['year'] > cutoff_year].copy()
            df_previous = df_filtered[(df_filtered['year'] <= cutoff_year) & 
                                    (df_filtered['year'] > cutoff_year - window_years)].copy()
            
            # Clean artist names
            for data in [df_recent, df_previous]:
                data['artist_clean'] = data['artists'].str.replace(r"[\[\]'\"]", "", regex=True)
                data['artist_clean'] = data['artist_clean'].str.split(',').str[0].str.strip()
            
            # Calculate metrics for both periods
            recent_stats = df_recent.groupby('artist_clean').agg({
                'popularity': 'mean',
                'name': 'count'
            }).rename(columns={'popularity': 'recent_pop', 'name': 'recent_tracks'})
            
            previous_stats = df_previous.groupby('artist_clean').agg({
                'popularity': 'mean',
                'name': 'count'
            }).rename(columns={'popularity': 'previous_pop', 'name': 'previous_tracks'})
            
            # Merge and calculate growth
            growth_analysis = pd.merge(recent_stats, previous_stats, 
                                    left_index=True, right_index=True, how='left')
            growth_analysis = growth_analysis.fillna(0)
            
            # Calculate growth metrics
            growth_analysis['pop_growth'] = ((growth_analysis['recent_pop'] - growth_analysis['previous_pop']) / 
                                            (growth_analysis['previous_pop'] + 1)) * 100
            growth_analysis['track_growth'] = ((growth_analysis['recent_tracks'] - growth_analysis['previous_tracks']) / 
                                            (growth_analysis['previous_tracks'] + 1)) * 100
            growth_analysis['momentum_score'] = (growth_analysis['pop_growth'] * 0.6 + 
                                                growth_analysis['track_growth'] * 0.4)
            
            # Reset index to get artist names
            growth_analysis = growth_analysis.reset_index()
            
            # Filter for meaningful results (at least 3 recent tracks)
            growth_analysis = growth_analysis[growth_analysis['recent_tracks'] >= 3]
            
            # Categories
            col_rise1, col_rise2 = st.columns(2)
            
            with col_rise1:
                # Fastest rising
                rising_stars = growth_analysis.nlargest(15, 'momentum_score')
                
                fig_rising = px.scatter(
                    rising_stars,
                    x='previous_pop',
                    y='recent_pop',
                    size='recent_tracks',
                    color='momentum_score',
                    hover_data=['artist_clean'],
                    title='Rising Stars: Previous vs Current Popularity',
                    labels={'previous_pop': f'Popularity ({cutoff_year-window_years}-{cutoff_year})',
                        'recent_pop': f'Popularity ({cutoff_year}-{current_year})',
                        'momentum_score': 'Momentum'},
                    template='plotly_dark',
                    color_continuous_scale='RdYlGn'
                )
                
                # Add diagonal line (no change)
                max_val = max(rising_stars['previous_pop'].max(), rising_stars['recent_pop'].max())
                fig_rising.add_shape(
                    type='line',
                    x0=0, y0=0, x1=max_val, y1=max_val,
                    line=dict(color='gray', dash='dash', width=1)
                )
                
                fig_rising.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                
                st.plotly_chart(fig_rising, use_container_width=True)
            
            with col_rise2:
                # New entrants (artists with no previous tracks)
                new_artists = growth_analysis[growth_analysis['previous_tracks'] == 0].nlargest(10, 'recent_pop')
                
                if not new_artists.empty:
                    fig_new = px.bar(
                        new_artists,
                        x='recent_pop',
                        y='artist_clean',
                        orientation='h',
                        color='recent_tracks',
                        title='Breakthrough Artists (New Entrants)',
                        labels={'recent_pop': 'Current Popularity',
                            'artist_clean': 'Artist',
                            'recent_tracks': 'Track Count'},
                        template='plotly_dark',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig_new.update_layout(
                        plot_bgcolor='#181818',
                        paper_bgcolor='#181818',
                        font_color='#B3B3B3',
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig_new, use_container_width=True)
                else:
                    st.info("No breakthrough artists found in the selected time window.")
            
                # Top movers summary
                st.markdown("### 📊 Movement Summary")
                
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    biggest_gainer = growth_analysis.nlargest(1, 'pop_growth')
                    if not biggest_gainer.empty:
                        st.metric(
                            "🎯 Biggest Popularity Gain",
                            biggest_gainer['artist_clean'].values[0][:25],
                            f"+{biggest_gainer['pop_growth'].values[0]:.1f}%"
                        )
                
                with col_sum2:
                    most_productive = growth_analysis.nlargest(1, 'recent_tracks')
                    if not most_productive.empty:
                        st.metric(
                            "🎵 Most Productive",
                            most_productive['artist_clean'].values[0][:25],
                            f"{most_productive['recent_tracks'].values[0]:.0f} tracks"
                        )
                
                with col_sum3:
                    highest_momentum = growth_analysis.nlargest(1, 'momentum_score')
                    if not highest_momentum.empty:
                        st.metric(
                            "🚀 Highest Momentum",
                            highest_momentum['artist_clean'].values[0][:25],
                            f"Score: {highest_momentum['momentum_score'].values[0]:.1f}"
                        )
                               

        # ---------------------------------------------------
        # VIZ 16: SONG TITLE ANALYTICS
        # ---------------------------------------------------
        st.divider()

        st.subheader("16. The Power of Words: Song Title Analysis")
        st.info("**🔍 Strategic Questions:** Do shorter titles perform better? What words appear most in hit songs? Can title sentiment predict success?")

        import re
        from collections import Counter

        title_analysis_type = st.radio(
            "Analysis Type:",
            ["Title Length vs Popularity", "Most Common Words", "Title Patterns"],
            horizontal=True,
            key="title_analysis"
        )

        # Add title length column
        df_titles = df_filtered.copy()
        df_titles['title_length'] = df_titles['name'].str.len()
        df_titles['title_word_count'] = df_titles['name'].str.split().str.len()

        if title_analysis_type == "Title Length vs Popularity":
            col_title1, col_title2 = st.columns(2)
            
            with col_title1:
                # Character length analysis
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
                fig_title_len.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_title_len, use_container_width=True)
            
            with col_title2:
                # Word count analysis
                word_bins = pd.cut(df_titles['title_word_count'], 
                                bins=[0, 1, 2, 3, 4, 10], 
                                labels=['1 word', '2 words', '3 words', '4 words', '5+ words'])
                word_popularity = df_titles.groupby(word_bins)['popularity'].mean().reset_index()
                
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
                fig_word_count.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_word_count, use_container_width=True)

        elif title_analysis_type == "Most Common Words":
            # Extract and count words from titles
            all_words = ' '.join(df_titles[df_titles['popularity'] > 50]['name'].str.lower()).split()
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'been'}
            filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
            
            word_freq = Counter(filtered_words)
            top_words = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
            
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
                plot_bgcolor='#181818',
                paper_bgcolor='#181818',
                font_color='#B3B3B3',
                height=600,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_words, use_container_width=True)

        else:  # Title Patterns
            # Check for patterns
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
                if len(data) == 2:
                    pattern_results.append({
                        'Pattern': pattern,
                        'Without': data[False],
                        'With': data[True],
                        'Impact': data[True] - data[False]
                    })
            
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
            fig_patterns.update_layout(
                plot_bgcolor='#181818',
                paper_bgcolor='#181818',
                font_color='#B3B3B3'
            )
            st.plotly_chart(fig_patterns, use_container_width=True)
            
        # ---------------------------------------------------
        # VIZ 17: COLLABORATION PATTERNS
        # ---------------------------------------------------
        st.divider()
        
        st.subheader("17. Collaboration Economy: The Power of Features")
        st.info("**🔍 Strategic Questions:** Do collaborations boost popularity? Which artists collaborate most? What's the optimal number of artists per track?")

        # Analyze collaboration patterns
        df_collab = df_filtered.copy()
        df_collab['artist_count'] = df_collab['artists'].str.count(',') + 1

        collab_view = st.radio(
            "View:",
            ["Collaboration Impact", "Artist Networks", "Optimal Team Size"],
            horizontal=True,
            key="collab_view"
        )

        if collab_view == "Collaboration Impact":
            # Solo vs Collaborations
            df_collab['is_collab'] = df_collab['artist_count'] > 1
            
            col_collab1, col_collab2 = st.columns(2)
            
            with col_collab1:
                collab_stats = df_collab.groupby('is_collab')['popularity'].agg(['mean', 'std', 'count']).reset_index()
                collab_stats['is_collab'] = collab_stats['is_collab'].map({False: 'Solo', True: 'Collaboration'})
                
                fig_collab_impact = px.bar(
                    collab_stats,
                    x='is_collab',
                    y='mean',
                    error_y='std',
                    title='Solo vs Collaboration Performance',
                    labels={'is_collab': 'Type', 'mean': 'Average Popularity'},
                    color='mean',
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
                fig_collab_impact.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_collab_impact, use_container_width=True)
            
            with col_collab2:
                # Trend over time
                collab_trend = df_collab[df_collab['year'] >= 1980].groupby('year')['is_collab'].mean() * 100
                
                fig_collab_trend = px.line(
                    x=collab_trend.index,
                    y=collab_trend.values,
                    title='Collaboration Trend Over Time',
                    labels={'x': 'Year', 'y': 'Collaboration %'},
                    markers=True,
                    template='plotly_dark'
                )
                fig_collab_trend.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3'
                )
                st.plotly_chart(fig_collab_trend, use_container_width=True)

        elif collab_view == "Artist Networks":
            # Most collaborative artists
            collab_only = df_collab[df_collab['artist_count'] > 1].copy()
            
            if len(collab_only) > 0:
                # Count collaborations per artist (simplified)
                artist_collabs = []
                for artists_str in collab_only['artists'].head(1000):  # Sample for performance
                    artists_list = [a.strip() for a in artists_str.replace('[','').replace(']','').replace("'",'').split(',')]
                    for artist in artists_list:
                        artist_collabs.append(artist)
                
                collab_counts = Counter(artist_collabs)
                top_collabs = pd.DataFrame(collab_counts.most_common(15), columns=['artist', 'collaborations'])
                
                fig_network = px.bar(
                    top_collabs,
                    x='collaborations',
                    y='artist',
                    orientation='h',
                    title='Most Collaborative Artists',
                    labels={'collaborations': 'Number of Collaborations', 'artist': 'Artist'},
                    color='collaborations',
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
                fig_network.update_layout(
                    plot_bgcolor='#181818',
                    paper_bgcolor='#181818',
                    font_color='#B3B3B3',
                    height=500,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_network, use_container_width=True)

        else:  # Optimal Team Size
            # Popularity by number of artists
            team_size = df_collab.groupby('artist_count')['popularity'].agg(['mean', 'count']).reset_index()
            team_size = team_size[team_size['artist_count'] <= 5]  # Focus on reasonable team sizes
            
            fig_team = px.scatter(
                team_size,
                x='artist_count',
                y='mean',
                size='count',
                title='Optimal Team Size for Hit Songs',
                labels={'artist_count': 'Number of Artists', 'mean': 'Average Popularity', 'count': 'Sample Size'},
                template='plotly_dark',
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            fig_team.update_layout(
                plot_bgcolor='#181818',
                paper_bgcolor='#181818',
                font_color='#B3B3B3'
            )
            st.plotly_chart(fig_team, use_container_width=True)
            
            # Show optimal
            optimal = team_size.loc[team_size['mean'].idxmax()]
            st.success(f"**Optimal Team Size:** {int(optimal['artist_count'])} artist(s) with average popularity of {optimal['mean']:.1f}")
        
        st.divider()



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
    decade_options = sorted(df[df['decade'] >= 1950]['decade'].unique())
    
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
    st.header("🤖 Music Data Consultant")
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
            
            # Create model
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=api_key,
                temperature=0
            )
            
            # Create Agent
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=system_prompt
            )
            
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
    st.header("📁 Raw Data Explorer")
    st.markdown("Explore the raw datasets and their structure.")
    
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