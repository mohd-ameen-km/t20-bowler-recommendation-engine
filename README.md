# 🏏T20 Bowler Recommendation Engine

### 🎯 Overview

This project analyzes T20 cricket data to recommend **the most effective bowling type** against any batter in any match phase (Powerplay, Middle, or Death overs).
It combines **data-driven statistical analysis** and **machine learning models** to identify each batter’s weaknesses based on historical performance.

---

### ⚙️ Features

* 🧠 **AI Recommendations** – Predicts which bowling type is most effective against a given batter.
* 📊 **Phase-Aware Analysis** – Separate insights for Powerplay, Middle, and Death overs.
* 📈 **Interactive Dashboard** – Built with Streamlit for intuitive visualization.
* 🚀 **Caching for Speed** – Cached dataset and models ensure instant recommendations.
* ⚡ **Handles Sparse Data** – Automatically ignores bowling types with insufficient data.

---

### 🏗️ Project Structure

```
t20_bowler_recommender/
│
├── app/
│   └── streamlit_app.py               # Main Streamlit app
│
├── src/
│   ├── data_processing.py             # Data cleaning and phase classification
│   ├── feature_engineering.py         # Feature extraction and normalization
│   ├── ml_models.py                   # ML model training and prediction
│   ├── recommendation_engine.py       # Core logic for recommendations
│   └── utils/
│       └── constants.py               # Configurable constants
│
├── data/
│   └── t20_bbb.csv                    # Local dataset
│
├── models/                            # Saved ML models (optional)
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Data exploration and visualization
│
└── requirements.txt
```

---

### 🧩 Installation

#### 1. Clone the repository

```bash
git clone https://github.com/yourusername/t20-bowler-recommender.git
cd t20-bowler-recommender
```

#### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # (Windows: venv\Scripts\activate)
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

### 💡 Usage

1. The app automatically loads the local dataset (`data/t20_bbb.csv`).
2. Select a **batter** and **match phase** from the sidebar.
3. View:

   * The **recommended bowling type**.
   * **Weakness score** visualization by bowling type.
   * **Similar batters** (if ML model trained).

---

### 🧠 Tech Stack

* **Frontend**: Streamlit
* **Backend / ML**: Python, Pandas, NumPy, Scikit-learn
* **Visualization**: Plotly, Matplotlib, Seaborn
* **Caching**: Streamlit Resource Cache, `functools.lru_cache`

---

### 🧪 Optional (Training Models)

You can train ML models from within the app (once implemented) using a sidebar button.
Trained models are saved in the `/models` directory for reuse.

---

### 📚 Dataset

The project expects a T20 ball-by-ball dataset containing fields like:

```
bat, bowl, batruns, out, ball, over, bowl_style, bowl_kind, wide, noball, ground, ...
```

You can replace `data/t20_bbb.csv` with any compatible dataset.

---

### 🧾 License

This project is released under the **MIT License**.
Feel free to use, modify, and share it with attribution.

---

### 👨‍💻 Author

**Ameen K M**
*Data Science & AI Enthusiast*

