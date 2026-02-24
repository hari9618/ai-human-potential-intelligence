import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Human Potential Intelligence", layout="centered")

# -----------------------------
# 1️⃣ Generate Synthetic Dataset
# -----------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    tf.random.set_seed(42)

    samples = 800

    technical_skill = np.random.randint(1, 11, samples)
    learning_velocity = np.random.randint(1, 11, samples)
    problem_solving = np.random.randint(1, 11, samples)
    adaptability = np.random.randint(1, 11, samples)
    stress_tolerance = np.random.randint(1, 11, samples)
    communication = np.random.randint(1, 11, samples)
    consistency = np.random.randint(1, 11, samples)

    score = (
        0.2 * technical_skill +
        0.2 * learning_velocity +
        0.15 * problem_solving +
        0.15 * adaptability +
        0.1 * stress_tolerance +
        0.1 * communication +
        0.1 * consistency
    )

    high_potential = (score > 6.5).astype(int)

    data = pd.DataFrame({
        "technical_skill": technical_skill,
        "learning_velocity": learning_velocity,
        "problem_solving": problem_solving,
        "adaptability": adaptability,
        "stress_tolerance": stress_tolerance,
        "communication": communication,
        "consistency": consistency,
        "high_potential": high_potential
    })

    return data


data = generate_data()

X = data.drop("high_potential", axis=1)
y = data["high_potential"]

# -----------------------------
# 2️⃣ Preprocessing
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Train Logistic Regression
# -----------------------------
@st.cache_resource
def train_logistic():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

log_model = train_logistic()
log_preds = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_preds)

# -----------------------------
# 4️⃣ Train ANN
# -----------------------------
@st.cache_resource
def train_ann():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(7,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0)
    return model

ann_model = train_ann()

ann_preds = (ann_model.predict(X_test) > 0.5).astype(int)
ann_accuracy = accuracy_score(y_test, ann_preds)

cm = confusion_matrix(y_test, ann_preds)

# -----------------------------
# 5️⃣ UI
# -----------------------------
st.title("🚀 AI Human Potential Intelligence System")
st.subheader("Deep Neural Network for High-Performance Prediction")

# ---- Model Evaluation ----
st.markdown("## 📊 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric("Logistic Regression Accuracy", f"{log_accuracy*100:.2f}%")

with col2:
    st.metric("ANN Accuracy", f"{ann_accuracy*100:.2f}%")

# ---- Confusion Matrix ----
st.markdown("### Confusion Matrix (ANN)")

fig, ax = plt.subplots()
cax = ax.imshow(cm)
fig.colorbar(cax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0,1])
ax.set_yticks([0,1])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# -----------------------------
# 6️⃣ Prediction Section
# -----------------------------
st.markdown("## 🎯 Candidate Evaluation")

technical = st.slider("Technical Skill Strength", 1, 10, 5)
learning = st.slider("Learning Velocity", 1, 10, 5)
problem = st.slider("Problem Solving Depth", 1, 10, 5)
adapt = st.slider("Adaptability Score", 1, 10, 5)
stress = st.slider("Stress Tolerance", 1, 10, 5)
comm = st.slider("Communication Effectiveness", 1, 10, 5)
consist = st.slider("Consistency Index", 1, 10, 5)

if st.button("Predict High Performance Potential"):

    input_data = np.array([[technical, learning, problem, adapt, stress, comm, consist]])
    input_scaled = scaler.transform(input_data)

    prediction = ann_model.predict(input_scaled)[0][0]
    probability = prediction * 100

    st.subheader("📈 Prediction Result")
    st.write(f"High Performance Probability: {probability:.2f}%")

    if probability > 75:
        st.success("🚀 Category: High Performer")
    elif probability > 50:
        st.warning("📈 Category: Emerging Talent")
    else:
        st.error("⚠️ Category: Needs Development")

# -----------------------------
# 7️⃣ Future Scope
# -----------------------------
st.markdown("## 🔮 Future Enhancements")
st.write("""
- Integration with real HR datasets  
- LLM-based behavioral analysis  
- PostgreSQL candidate tracking  
- Deployment as enterprise SaaS platform  
""")