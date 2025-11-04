# app/streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys, os

# Ensure src/ modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import T20DataProcessor
from src.recommendation_engine import EnhancedBowlerRecommender


# -----------------------------
# ✅ Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="T20 Bowler Recommender", layout="wide")
st.title("🏏T20 Bowler Recommendation Engine")


# -----------------------------
# ⚙️ Load Data & Initialize Recommender (CACHED)
# -----------------------------
@st.cache_resource
def load_and_prepare_recommender():
    """
    Load dataset once, initialize data processor and recommender once.
    Cached between reruns for instant UI response.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / "t20_bbb.csv"
    if not data_path.exists():
        st.error(f"❌ Dataset not found at {data_path}")
        st.stop()

    df = pd.read_csv(data_path)
    dp = T20DataProcessor(df)
    recommender = EnhancedBowlerRecommender(dp)
    return df, dp, recommender


df, dp, recommender = load_and_prepare_recommender()


# -----------------------------
# 🧠 Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

batters = sorted(df['bat'].unique())
phases = ['Powerplay', 'Middle1', 'Middle2', 'Death']

batter = st.sidebar.selectbox("🎯 Select Batter", batters)
phase = st.sidebar.selectbox("⏰ Select Match Phase", phases)


# -----------------------------
# 🚀 Generate Recommendation
# -----------------------------
if batter and phase:
    with st.spinner(f"Analyzing {batter}'s performance in {phase}..."):
        result = recommender.recommend(batter, phase)

    if not result or result.get("recommended_type") in [None, "No Reliable Data"]:
        st.warning("Insufficient or unreliable data for this batter-phase combination.")
        st.stop()

    # Display core results
    st.subheader(f"Recommended Bowling Type for **{batter}** in **{phase}**")
    st.metric("🧩 Recommended Bowling Type", result["recommended_type"])
    st.metric("📊 Weakness Score", f"{result['weakness_score']:.1f}")

    # Visualization
    st.write("### 🔍 Weakness Analysis by Bowling Type")

    if result["all_predictions"]:
        data = pd.DataFrame({
            "Bowling Type": result["all_predictions"].keys(),
            "Weakness Score": result["all_predictions"].values()
        }).sort_values("Weakness Score", ascending=False)

        fig = px.bar(
            data,
            x="Bowling Type",
            y="Weakness Score",
            color="Bowling Type",
            text="Weakness Score",
            title=f"{batter}'s Weakness Profile Against Bowling Types"
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reliable bowling-type data available for visualization.")

    # Similar batters (if ML used)
    if result.get("similar_batters"):
        st.write(f"👥 **Similar Batters:** {', '.join(result['similar_batters'])}")
