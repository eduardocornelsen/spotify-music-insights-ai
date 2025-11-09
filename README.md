<div align='center'>
    
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-272D32?style=for-the-badge&logo=plotly&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-111111?style=for-the-badge&logo=langchain&logoColor=white) ![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white) ![Render](https://img.shields.io/badge/Render-009966?style=for-the-badge&logo=render&logoColor=white)

# 🎵 MusicInsights AI
## Interactive Music Consultant with an AI (Pandas) Agent

[ANIMATED GIF]

A data analysis dashboard that goes beyond static charts. This project uses a **Tool Calling Agent (LangChain)** to allow music executives and producers to ask complex questions in natural language and receive deep statistical analyses about *why* certain songs become popular, correlating `popularity` with audio features like `danceability`, `energy`, and `valence`.

<a href="https://music-insights-ai-demo.onrender.com/" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/Try%20The%20Live%20App-009966?style=for-the-badge&logo=rocket&logoColor=FFFFFF" 
    alt="Try the Live App" 
    style="border: none; height: 35px; margin-top:20px; margin-bottom: 35px;">
</a>
<br>

</div>

---



## Description / Descrição

<details>
 <summary>
 <b style="font-size: 1.4em;">1. 🇺🇸 English Version</b>
 </summary>
 
 > [![VERSÃO PT-BR](https://img.shields.io/badge/🇧🇷%20VERSÃO%20PT--BR-333?style=for-the-badge&logoColor=white)](#2-🇧🇷-versão-em-português---br)

### 📌 Table of Contents
1.  [Project Summary](#-project-summary)
2.  [Key Features](#-key-features)
3.  [Technical Approach](#-technical-approach-the-ai-agent)
4.  [Project Files](#-project-files)
5.  [Local Installation](#-local-installation)

<br>

## 📋 Project Summary

This project solves a core problem in the music industry: traditional dashboards show *what* songs are popular, but fail to explain *why*. **MusicInsights AI** bridges this gap.

The application is split into two main sections:
1.  **Exploratory Analysis (EDA):** An interactive dashboard (`Plotly Express`) that visualizes trends in audio features (danceability, energy, etc.) across the decades.
2.  **Data Consultant (AI):** The core feature. A chat interface that allows users to ask complex questions in natural language. The AI (Google Gemini) **writes and executes Pandas code** in real-time to provide deep statistical analysis not pre-calculated in the dashboard.

The goal is to shift data analysis from reactive to proactive, allowing an A&R executive to ask, "What is the correlation between `energy` and `popularity` in explicit tracks from the 90s?" and receive a factual, data-driven answer.

---

## 🚀 Key Features

| Section | Feature | Technologies | Impact (The Problem Solved) |
| :--- | :--- | :--- | :--- |
| **AI Consultant** | **Tool Calling Agent (Gemini)** | `LangChain`, `Gemini API`, `@tool` | **Solves Data Inaccessibility.** The AI can execute Pandas code to answer complex ad-hoc questions (`.corr()`, `.groupby()`, `.quantile()`) that are not in static charts. |
| **EDA Dashboard** | **Audio Feature Visualizations** | `Plotly Express`, `Streamlit` | Visualizes the evolution of music, showing how `danceability`, `energy`, and `valence` have changed over the decades (Box Plots, Heatmaps, Regression). |
| **Navigation** | **Interactive Sidebar** | `st.sidebar.radio` | Clean, professional navigation between the app's sections. |

---

## 🛠️ Technical Approach: The AI Agent

The heart of this project is the AI Agent, built with the modern "Tool Calling" standard (LangChain v1.x):

1.  **Model:** `ChatGoogleGenerativeAI` (using `gemini-2.5-flash`).
2.  **Tool (`@tool`):** A single custom tool (`PythonCodeExecutor`) is exposed to the model.
3.  **Execution Flow:**
    * User asks: "What's the average `valence` for songs with `energy` > 0.8?"
    * The Agent (created via `create_agent`) receives the prompt.
    * The AI (Gemini) decides it needs the tool and **writes** the code: `print(df[df['energy'] > 0.8]['valence'].mean())`.
    * The `PythonCodeExecutor` function executes this code safely on the `car_data` DataFrame (aliased as `df`) and captures the `print()` output.
    * The Agent returns the numerical result to the user in natural language.

---

## 📂 Project Files

```bash
.
├── app.py                     # Main Streamlit app code (Agent & Visualizations)
├── spotify_dataset.csv        # Music dataset (e.g., Kaggle 160k Tracks)
├── requirements.txt           # Python dependencies (LangChain, Streamlit, etc.)
├── runtime.txt                # Defines Python version for Render (python-3.11.8)
├── .gitignore                 # Ignores /venv, __pycache__, and secrets.toml
├── LICENSE                    # Project license (e.g., MIT)
├── .streamlit/                # Streamlit config folder
│   └── config.toml            # Render server configuration
└── prompts/                   # AI Agent instructions folder
    └── system.txt             # System Prompt for the AI Agent
````

-----

## 💻 Local Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/eduardocornelsen/music-insights-ai.git
cd music-insights-ai
```

### 2\. Create and Activate a Virtual Environment (Required)

```bash
# Python 3.11 is recommended for LangChain compatibility
conda create --name music-ai-env python=3.11
conda activate music-ai-env
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configure API Key

Create the `.streamlit/secrets.toml` file in the project root:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 
```

### 5\. Run the Streamlit App

```bash
streamlit run app.py
```

<div align='center'>

<a href="https://music-insights-ai-demo.onrender.com/" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/Try%20The%20Live%20App-009966?style=for-the-badge&logo=rocket&logoColor=FFFFFF" 
    alt="Try the Live App" 
    style="border: none; height: 35px; margin-top:20px; margin-bottom: 35px;">
</a>
<br>

</div>

</details>

-----

<details>
<summary>
<b style="font-size: 1.4em;">2. 🇧🇷 Versão em Português - BR</b>
</summary>

> [](https://www.google.com/search?q=%231-%F0%9F%87%BA%F0%9F%87%B8-english-version)

### 📌 Índice

1.  [Resumo do Projeto](https://www.google.com/search?q=%23-resumo-do-projeto)
2.  [Funcionalidades de Destaque](https://www.google.com/search?q=%23-funcionalidades-de-destaque)
3.  [Abordagem Técnica](https://www.google.com/search?q=%23-abordagem-t%C3%A9cnica-o-agente-de-ia)
4.  [Estrutura de Arquivos](https://www.google.com/search?q=%23-estrutura-de-arquivos)
5.  [Instalação Local](https://www.google.com/search?q=%23-instala%C3%A7%C3%A3o-local)

<br>

## 📋 Resumo do Projeto

Este projeto resolve um problema central na indústria da música: dashboards tradicionais mostram *quais* músicas são populares, mas falham em explicar o *porquê*. O **MusicInsights AI** preenche essa lacuna.

O aplicativo é dividido em duas seções principais:

1.  **Análise Exploratória (EDA):** Um dashboard interativo (`Plotly Express`) que visualiza tendências em características de áudio (dançabilidade, energia, etc.) ao longo das décadas.
2.  **Consultor de Dados (IA):** A funcionalidade principal. Um chat que permite ao usuário fazer perguntas complexas em linguagem natural. A IA (Google Gemini) **escreve e executa código Pandas** em tempo real para fornecer análises estatísticas profundas que não estão pré-calculadas no dashboard.

O objetivo é transformar a análise de dados de reativa para proativa, permitindo que um executivo de A\&R pergunte, por exemplo, "Qual é a correlação entre `energia` e `popularidade` nas músicas explícitas dos anos 90?" e receba uma resposta factual.

-----

## 🚀 Funcionalidades de Destaque

| Seção | Funcionalidade | Tecnologias | Impacto (O Problema Resolvido) |
| :--- | :--- | :--- | :--- |
| **Consultor de IA** | **Tool Calling Agent (Gemini)** | `LangChain`, `Gemini API`, `@tool` | **Resolve a Inacessibilidade de Dados.** A IA pode executar código Pandas para responder perguntas complexas (`.corr()`, `.groupby()`, `.quantile()`) que não estão em gráficos estáticos. |
| **EDA Avançada** | **Visualizações de Features de Áudio** | `Plotly Express`, `Streamlit` | Visualiza a evolução da música, mostrando como `danceability`, `energy`, e `valence` mudaram ao longo das décadas (Box Plots, Heatmaps, Regressão). |
| **Navegação** | **Sidebar Interativa** | `st.sidebar.radio` | Navegação limpa e profissional entre as seções do dashboard. |

-----

## 🛠️ Abordagem Técnica: O Agente de IA

O coração deste projeto é o Agente de IA, construído com o padrão moderno "Tool Calling" do LangChain v1.x:

1.  **Modelo:** `ChatGoogleGenerativeAI` (usando `gemini-2.5-flash`).
2.  **Ferramenta (`@tool`):** Uma única ferramenta customizada (`PythonCodeExecutor`) é exposta ao modelo.
3.  **Fluxo de Execução:**
      * O usuário pergunta: "Qual a média de `valence` para músicas com `energy` \> 0.8?"
      * O Agente (criado com `create_agent`) recebe o prompt.
      * A IA (Gemini) decide que precisa da ferramenta e **escreve** o código: `print(df[df['energy'] > 0.8]['valence'].mean())`.
      * A função `PythonCodeExecutor` executa esse código com segurança no DataFrame `car_data` (acessível como `df`) e captura a saída (`print()`).
      * O Agente retorna o resultado numérico ao usuário em linguagem natural.

-----

## 📂 Estrutura de Arquivos

```bash
.
├── app.py                     # Código principal do Streamlit (Agente e Visualizações)
├── spotify_dataset.csv        # Dataset de músicas (Ex: Kaggle 160k Tracks)
├── requirements.txt           # Dependências Python (LangChain, Streamlit, etc.)
├── runtime.txt                # Define a versão do Python no Render (python-3.11.8)
├── .gitignore                 # Ignora /venv, __pycache__, e secrets.toml
├── LICENSE                    # Licença do projeto (Ex: MIT)
├── .streamlit/                # Pasta de configuração do Streamlit
│   └── config.toml            # Configuração do servidor Render
└── prompts/                   # Pasta de instruções para a IA
    └── system.txt             # Instruções de alto nível (System Prompt)
```

-----

## 💻 Instalação Local

### 1\. Clonar o Repositório

```bash
git clone https://github.com/eduardocornelsen/music-insights-ai.git
cd music-insights-ai
```

### 2\. Criar e Ativar um Ambiente Virtual (Obrigatório)

```bash
# Recomendado Python 3.11 para compatibilidade do LangChain
conda create --name music-ai-env python=3.11
conda activate music-ai-env
```

### 3\. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4\. Configurar a Chave API

Crie o arquivo `.streamlit/secrets.toml` na raiz do projeto:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "SUA_CHAVE_API_DO_GEMINI_AQUI" 
```

### 5\. Executar o App Streamlit

```bash
streamlit run app.py
```

<div align='center'>

<a href="https://music-insights-ai-demo.onrender.com/" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/Teste%20o%20App%20Ao%20Vivo-009966?style=for-the-badge&logo=rocket&logoColor=FFFFFF" 
    alt="Teste o App Ao Vivo" 
    style="border: none; height: 35px; margin-top:20px; margin-bottom: 35px;">
</a>
<br>


</div>

</details>

-----

<p align = "center">
Copyright © 2025, Eduardo Cornelsen
</p>
