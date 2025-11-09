import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import sys
import os
from pathlib import Path

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

# Read System Prompt from file (fallback empty if not present)
try:
    system_prompt = Path("./prompts/system.txt").read_text(encoding="utf-8")
except Exception:
    system_prompt = "You are an AI data analyst. Use the provided DataFrames to answer questions with real computations."

# --- Configuração da Página ---
st.set_page_config(
    page_title="MusicInsights AI",
    page_icon="🎵",
    layout="wide"
)

# --- Carregar e Limpar os Dados (com cache) ---
@st.cache_data
def load_data():
    data_dir = Path("data")
    required_files = {
        "tracks": data_dir / "data.csv",
        "by_year": data_dir / "data_by_year.csv",
        "by_artist": data_dir / "data_by_artist.csv",
        "by_genres": data_dir / "data_by_genres.csv",
        "w_genres": data_dir / "data_w_genres.csv",
    }

    missing = [k for k, v in required_files.items() if not v.exists()]
    if missing:
        st.error(f"Erro: Arquivos ausentes em ./data: {', '.join(missing)}")
        return None

    try:
        df_tracks = pd.read_csv(required_files["tracks"])
        df_year = pd.read_csv(required_files["by_year"])
        df_artist = pd.read_csv(required_files["by_artist"])
        df_genres = pd.read_csv(required_files["by_genres"])
        df_w_genres = pd.read_csv(required_files["w_genres"])

        # Normalize tracks schema
        # Handle release_date column name variants
        if "release date" in df_tracks.columns and "release_date" not in df_tracks.columns:
            df_tracks.rename(columns={"release date": "release_date"}, inplace=True)

        # Ensure 'year' exists (if not, try parse from release_date)
        if "year" not in df_tracks.columns:
            if "release_date" in df_tracks.columns:
                # Attempt to parse year from release_date
                try:
                    df_tracks["release_date"] = pd.to_datetime(df_tracks["release_date"], errors="coerce")
                    df_tracks["year"] = df_tracks["release_date"].dt.year
                except Exception:
                    pass

        # Create decade column for tracks if year is available
        if "year" in df_tracks.columns:
            df_tracks["decade"] = (df_tracks["year"] // 10) * 10

        # Normalize 'explicit' to int/bool friendly
        if "explicit" in df_tracks.columns:
            # Keep original explicit, and also provide a boolean/helper label
            try:
                df_tracks["explicit_bool"] = df_tracks["explicit"].astype(int).astype(bool)
            except Exception:
                # If it is already boolean or non-numeric
                df_tracks["explicit_bool"] = df_tracks["explicit"].astype(str).str.lower().isin(["true", "1", "yes"])

        # Ensure df_year has 'year' int for plotting
        if "year" in df_year.columns:
            df_year["year"] = pd.to_numeric(df_year["year"], errors="coerce").astype("Int64")

        return {
            "df_tracks": df_tracks,
            "df_year": df_year,
            "df_artist": df_artist,
            "df_genres": df_genres,
            "df_w_genres": df_w_genres
        }

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

music_data = load_data()

# --- Sidebar ---
st.sidebar.title("About the Project 💡")
st.sidebar.markdown(
    """
    MusicInsights AI brings interactive EDA together with an **AI Data Consultant (Tool Calling)**.
    
    - Ask natural language questions like:
      - "What is the correlation between energy and popularity in explicit tracks from the 90s?"
      - "Average danceability for the top 10% most popular songs?"
    
    - The AI writes and executes Pandas code on the fly.

    Author: **Eduardo Cornelsen**  
    Stack: **Streamlit + LangChain + Gemini**
    """
)
st.sidebar.info("Use the ***Tab 2 (AI Consultant)*** to chat with the AI Agent.")

# --- Título Principal ---
st.title("🎵 MusicInsights AI")
st.write("Interactive Music Consultant with a Tool Calling Agent (LangChain + Gemini 2.5)")

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
        df_tracks = music_data["df_tracks"].copy()
        df_year = music_data["df_year"].copy()
        df_artist = music_data["df_artist"].copy()
        df_genres = music_data["df_genres"].copy()
        df_w_genres = music_data["df_w_genres"].copy()

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

# --- Renderização do App ---
if music_data is not None:

    df_tracks = music_data["df_tracks"]
    df_year = music_data["df_year"]
    df_artist = music_data["df_artist"]
    df_genres = music_data["df_genres"]
    df_w_genres = music_data["df_w_genres"]

    # --- Criar as Abas ---
    tab1, tab2, tab3 = st.tabs([
        "Exploratory Analysis (EDA)", 
        "AI Consultant (Agent Executor)",
        "View Raw Data"
    ])

    # --------------------------------------------------------
    # --- Tab 1: Exploratory Analysis (EDA) ---
    # --------------------------------------------------------
    with tab1:
        st.header("Exploratory Analysis of Audio Features and Popularity")
        st.markdown("Nine interactive visualizations to explore how audio features relate to popularity across years and genres.")

        st.divider()
        st.subheader("1) Data Viewer with Filters (Tracks Level)")
        # Filters
        decades_available = sorted(df_tracks["decade"].dropna().unique()) if "decade" in df_tracks.columns else []
        decade_sel = st.multiselect("Filter by Decade:", decades_available, default=decades_available)
        min_pop = st.slider("Minimum Popularity:", min_value=0, max_value=100, value=50, step=1)
        include_explicit = st.checkbox("Include Explicit Tracks", value=True)
        top_n_rows = st.number_input("Rows to Display (head):", min_value=10, max_value=500, value=50, step=10)

        df_display = df_tracks.copy()
        if decade_sel and "decade" in df_display.columns:
            df_display = df_display[df_display["decade"].isin(decade_sel)]
        df_display = df_display[df_display["popularity"] >= min_pop]
        if "explicit" in df_display.columns and not include_explicit:
            # if explicit==1 means explicit tracks
            df_display = df_display[df_display["explicit"] == 0]

        st.dataframe(df_display.head(int(top_n_rows)))
        st.markdown(f"Total Records After Filters: **{len(df_display):,}**")

        # 2) Popularity trend over time
        st.divider()
        st.subheader("2) Popularity Over Time (Yearly)")
        if "year" in df_year.columns and "popularity" in df_year.columns:
            fig_pop_trend = px.line(
                df_year.sort_values("year"),
                x="year", y="popularity",
                title="Average Popularity by Year"
            )
            st.plotly_chart(fig_pop_trend, use_container_width=True)
        else:
            st.info("data_by_year.csv is missing 'year' or 'popularity' columns.")

        # 3) Danceability / Energy / Valence trends over years
        st.divider()
        st.subheader("3) Feature Trends (Danceability, Energy, Valence)")
        feature_cols = [c for c in ["danceability", "energy", "valence"] if c in df_year.columns]
        if feature_cols and "year" in df_year.columns:
            df_year_long = df_year.melt(id_vars=["year"], value_vars=feature_cols,
                                        var_name="feature", value_name="value")
            fig_features_trend = px.line(
                df_year_long.sort_values("year"),
                x="year", y="value", color="feature",
                title="Audio Feature Trends Across Years"
            )
            st.plotly_chart(fig_features_trend, use_container_width=True)
        else:
            st.info("Feature columns not found in data_by_year.csv.")

        # 4) Box Plot: Danceability by Decade
        st.divider()
        st.subheader("4) Distribution of Danceability by Decade (Box Plot)")
        if "danceability" in df_tracks.columns and "decade" in df_tracks.columns:
            fig_box_dance = px.box(
                df_tracks.dropna(subset=["danceability", "decade"]),
                x="decade", y="danceability", color="decade",
                title="Danceability Distribution by Decade",
                category_orders={"decade": sorted(df_tracks["decade"].dropna().unique())}
            )
            fig_box_dance.update_layout(xaxis_title="Decade", yaxis_title="Danceability")
            st.plotly_chart(fig_box_dance, use_container_width=True)
        else:
            st.info("Missing 'danceability' or 'decade' in tracks dataset.")

        # 5) Scatter: Energy vs Popularity (Regression) by Explicit
        st.divider()
        st.subheader("5) Energy vs Popularity with Regression (Explicit vs Non-Explicit)")
        if all(c in df_tracks.columns for c in ["energy", "popularity"]):
            df_scatter = df_tracks[df_tracks["popularity"] <= df_tracks["popularity"].quantile(0.99)].copy()
            color_col = "explicit" if "explicit" in df_scatter.columns else None
            fig_energy_pop = px.scatter(
                df_scatter, x="energy", y="popularity", color=color_col,
                title="Energy vs Popularity",
                opacity=0.5, trendline="ols", height=550
            )
            st.plotly_chart(fig_energy_pop, use_container_width=True)
        else:
            st.info("Missing 'energy' or 'popularity' in tracks dataset.")

        # 6) Correlation Heatmap (selected numeric columns)
        st.divider()
        st.subheader("6) Correlation Heatmap (Features vs Popularity)")
        corr_cols = [c for c in ["popularity", "danceability", "energy", "valence", "loudness", "speechiness", "acousticness", "tempo"]
                     if c in df_tracks.columns]
        if len(corr_cols) >= 2:
            corr_matrix = df_tracks[corr_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                title="Correlation Matrix (Tracks Level)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns found for correlation heatmap.")

        # 7) Genres: Top 15 by Average Popularity
        st.divider()
        st.subheader("7) Top Genres by Average Popularity")
        if all(c in df_genres.columns for c in ["genres", "popularity"]):
            df_genres_pop = df_genres.dropna(subset=["genres", "popularity"]).copy()
            df_genres_pop["genres"] = df_genres_pop["genres"].astype(str)
            topN = st.slider("How many top genres?", min_value=5, max_value=30, value=15, step=5)
            df_top_genres = df_genres_pop.sort_values("popularity", ascending=False).head(topN)
            fig_genres = px.bar(
                df_top_genres,
                x="genres", y="popularity", color="genres",
                title=f"Top {topN} Genres by Average Popularity"
            )
            fig_genres.update_layout(xaxis_title="Genre", yaxis_title="Avg Popularity")
            st.plotly_chart(fig_genres, use_container_width=True)
        else:
            st.info("Missing 'genres' or 'popularity' in data_by_genres.csv.")

        # 8) Density Heatmap: Tempo vs Popularity
        st.divider()
        st.subheader("8) Density of Tracks by Tempo vs Popularity")
        if all(c in df_tracks.columns for c in ["tempo", "popularity"]):
            fig_temp_pop = px.density_heatmap(
                df_tracks, x="tempo", y="popularity",
                title="Density of Tracks by Tempo and Popularity", nbinsx=40, nbinsy=20, text_auto=True
            )
            st.plotly_chart(fig_temp_pop, use_container_width=True)
        else:
            st.info("Missing 'tempo' or 'popularity' in tracks dataset.")

        # 9) Stacked Bars: Explicit vs Non-Explicit by Decade
        st.divider()
        st.subheader("9) Explicit vs Non-Explicit by Decade")
        if "decade" in df_tracks.columns and "explicit" in df_tracks.columns:
            df_exp = df_tracks.dropna(subset=["decade", "explicit"]).copy()
            df_exp["explicit_label"] = df_exp["explicit"].apply(lambda x: "Explicit" if int(x) == 1 else "Non-Explicit")
            fig_explicit_decade = px.histogram(
                df_exp, x="decade", color="explicit_label",
                barmode="group", title="Explicit vs Non-Explicit by Decade"
            )
            fig_explicit_decade.update_layout(xaxis_title="Decade", yaxis_title="Count")
            st.plotly_chart(fig_explicit_decade, use_container_width=True)
        else:
            st.info("Missing 'decade' or 'explicit' in tracks dataset.")

        st.divider()

    # --------------------------------------------------------
    # --- Tab 2: AI Consultant (Agent Executor) ---
    # --------------------------------------------------------
    with tab2:
        st.header("AI Music Data Consultant 🧠")
        st.markdown("Ask complex questions in natural language. The AI will write and execute Pandas code on the real datasets.")

        if not IA_DISPONIVEL:
            st.warning("LangChain libraries not installed correctly. The AI tab is disabled.")
            st.info("Install dependencies and restart the app.")
        else:
            try:
                # Load API key (env or secrets)
                api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
                if api_key is None:
                    st.warning("Google API key not found.")
                    st.write("Please add the `GOOGLE_API_KEY` in environment or .streamlit/secrets.toml.")
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
                if "chat_messages_music" not in st.session_state:
                    st.session_state.chat_messages_music = []

                # Initialize button prompt
                if 'button_prompt_music' not in st.session_state:
                    st.session_state.button_prompt_music = None

                # Keep on this tab when active
                if 'force_tab2_music' not in st.session_state:
                    st.session_state.force_tab2_music = False

                def set_button_prompt(prompt):
                    st.session_state.button_prompt_music = prompt
                    st.session_state.force_tab2_music = True

                # Display history
                for message in st.session_state.chat_messages_music:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                chat_input = st.chat_input("Ex: What is the correlation between energy and popularity in explicit tracks from the 90s?")

                st.divider()
                st.subheader("💡 Suggested Questions")

                # ROW 1: Popularity & Features
                st.markdown("**1. Popularity & Features**")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("Avg Popularity by Decade", key='btn_m1a', use_container_width=True):
                        set_button_prompt("Compute the average popularity by decade using df_tracks (groupby 'decade'). Print a sorted table.")
                with col2:
                    if st.button("Energy-Popularity Corr (90s, Explicit)", key='btn_m1b', use_container_width=True):
                        set_button_prompt("Filter df_tracks to explicit==1 and year between 1990 and 1999, then print the correlation between 'energy' and 'popularity'.")
                with col3:
                    if st.button("Top 10% Popular: Valence & Danceability", key='btn_m1c', use_container_width=True):
                        set_button_prompt("Using df_tracks, find the 90th percentile of 'popularity'. For tracks above it, print the mean of 'valence' and 'danceability'.")
                with col4:
                    if st.button("Feature Corr Matrix (Yearly)", key='btn_m1d', use_container_width=True):
                        set_button_prompt("Using df_year, print the correlation matrix between 'popularity', 'danceability', 'energy', and 'valence'.")
                with col5:
                    if st.button("Tempo vs Popularity (Slope)", key='btn_m1e', use_container_width=True):
                        set_button_prompt("On df_tracks, compute a simple linear regression slope between 'tempo' and 'popularity' (use numpy polyfit) and print the slope.")

                # ROW 2: Genres
                st.markdown("**2. Genres**")
                col6, col7, col8, col9, col10 = st.columns(5)
                with col6:
                    if st.button("Top 10 Genres by Danceability", key='btn_m2a', use_container_width=True):
                        set_button_prompt("Using df_genres, print the top 10 genres by average 'danceability'.")
                with col7:
                    if st.button("Energy by Genre (Variance)", key='btn_m2b', use_container_width=True):
                        set_button_prompt("Using df_genres, compute and print the top 10 genres by 'energy' variance.")
                with col8:
                    if st.button("Hip Hop: Popularity vs Energy (2000s)", key='btn_m2c', use_container_width=True):
                        set_button_prompt("Using df_w_genres, filter rows whose 'genres' contains 'hip hop' (case-insensitive) and year between 2000 and 2009, then print the correlation between 'popularity' and 'energy'.")
                with col9:
                    if st.button("Genre Mix: Valence vs Popularity", key='btn_m2d', use_container_width=True):
                        set_button_prompt("Using df_w_genres, group by 'genres' and print the top 10 genres with highest average 'valence' among tracks with popularity >= 60.")
                with col10:
                    if st.button("Genres: Loudness Leaderboard", key='btn_m2e', use_container_width=True):
                        set_button_prompt("Using df_genres, print the top 10 genres by average 'loudness' (descending).")

                # ROW 3: Artists & Tracks
                st.markdown("**3. Artists & Tracks**")
                col11, col12, col13, col14, col15 = st.columns(5)
                with col11:
                    if st.button("Top Artists (2010s)", key='btn_m3a', use_container_width=True):
                        set_button_prompt("Using df_tracks, filter years 2010..2019 and print the 10 artists with highest average popularity (groupby 'artists').")
                with col12:
                    if st.button("Top Tracks in 2020", key='btn_m3b', use_container_width=True):
                        set_button_prompt("Using df_tracks, list the 10 most popular tracks in 2020. Print name, artists, popularity.")
                with col13:
                    if st.button("Consistency: High Median Popularity", key='btn_m3c', use_container_width=True):
                        set_button_prompt("Using df_tracks, group by 'artists' and compute median popularity. Print the top 10 artists by median popularity (min count >= 5).")
                with col14:
                    if st.button("Explicit Ratio by Decade", key='btn_m3d', use_container_width=True):
                        set_button_prompt("Using df_tracks, compute the ratio of explicit tracks by decade (explicit==1 / total). Print as a sorted table.")
                with col15:
                    if st.button("Loudness by Decade", key='btn_m3e', use_container_width=True):
                        set_button_prompt("Using df_tracks, print the average 'loudness' per decade.")

                st.divider()

                # Combine button prompt or chat input
                user_input = st.session_state.button_prompt_music or chat_input

                # Clear button prompt after use
                if st.session_state.button_prompt_music:
                    st.session_state.button_prompt_music = None

                # Process input
                if user_input:
                    st.chat_message("user").markdown(user_input)
                    st.session_state.chat_messages_music.append({"role": "user", "content": user_input})

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()

                        try:
                            with st.spinner("Thinking and executing..."):
                                response = agent.invoke({"messages": st.session_state.chat_messages_music})

                            # Check for malformed call
                            if response["messages"][-1].response_metadata.get('finish_reason') == 'MALFORMED_FUNCTION_CALL':
                                message_placeholder.empty()
                                st.error("The model had trouble processing your request. Please rephrase and try again.")
                                st.stop()

                            # Extract AI response
                            ai_content = response["messages"][-1].content

                            if isinstance(ai_content, list) and len(ai_content) > 0:
                                text_content = ai_content[0].get('text', '')
                            else:
                                text_content = ai_content

                            # DEBUG
                            with st.expander("🔍 Debug: Raw agent response"):
                                st.code(str(response), language="python")

                            # Display text
                            message_placeholder.markdown(text_content)
                            st.session_state.chat_messages_music.append({
                                "role": "assistant",
                                "content": text_content
                            })

                        except Exception as e:
                            message_placeholder.empty()
                            st.error(f"Error while processing: {str(e)}")
                            st.write("Details:", e)

            except Exception as e:
                st.error(f"Error initializing the Agent: {e}")

    # --------------------------------------------------------
    # --- Tab 3: View Raw Data ---
    # --------------------------------------------------------
    with tab3:
        st.header("Raw Datasets and Columns")

        dataset_choice = st.selectbox(
            "Choose a dataset to view:",
            ["df_tracks (data.csv)", "df_year (data_by_year.csv)", "df_artist (data_by_artist.csv)",
             "df_genres (data_by_genres.csv)", "df_w_genres (data_w_genres.csv)"]
        )

        if dataset_choice.startswith("df_tracks"):
            st.dataframe(df_tracks)
            st.subheader("Columns:")
            st.write(list(df_tracks.columns))
        elif dataset_choice.startswith("df_year"):
            st.dataframe(df_year)
            st.subheader("Columns:")
            st.write(list(df_year.columns))
        elif dataset_choice.startswith("df_artist"):
            st.dataframe(df_artist)
            st.subheader("Columns:")
            st.write(list(df_artist.columns))
        elif dataset_choice.startswith("df_genres"):
            st.dataframe(df_genres)
            st.subheader("Columns:")
            st.write(list(df_genres.columns))
        else:
            st.dataframe(df_w_genres)
            st.subheader("Columns:")
            st.write(list(df_w_genres.columns))

else:
    st.info("Awaiting datasets in './data' to start the application.")