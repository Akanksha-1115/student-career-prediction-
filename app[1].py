import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Student Career Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CSS / UI --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef4ff 50%, #f9fbff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #172554 100%);
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }

.hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(120deg, #172554, #2563eb, #06b6d4);
    color: white;
    box-shadow: 0 18px 45px rgba(37,99,235,.22);
    margin-bottom: 24px;
    animation: fadeIn .65s ease-out;
}
.hero h1 { font-size: 38px; margin: 0 0 8px 0; font-weight: 800; }
.hero p { margin: 0; opacity: .9; font-size: 16px; }

.card {
    background: rgba(255,255,255,.92);
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
    transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(15,23,42,.10); }

.metric-title { color:#64748b; font-size:13px; font-weight:600; }
.metric-value { color:#0f172a; font-size:28px; font-weight:800; margin-top:4px; }

.prediction {
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 18px 0;
    animation: pop .5s ease-out;
}
.prediction.placed { background: linear-gradient(135deg,#059669,#10b981); }
.prediction.notplaced { background: linear-gradient(135deg,#dc2626,#f97316); }
.prediction h2 { font-size: 34px; margin: 4px; }
.prediction p { margin: 5px; opacity:.9; }

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin: 22px 0 12px;
}
.badge {
    display:inline-block; padding:5px 10px; border-radius:999px;
    background:#dbeafe; color:#1d4ed8; font-size:12px; font-weight:700;
}

@keyframes fadeIn { from {opacity:0; transform:translateY(10px)} to {opacity:1; transform:translateY(0)} }
@keyframes pop { 0% {transform:scale(.96); opacity:0} 100% {transform:scale(1); opacity:1} }
</style>
""", unsafe_allow_html=True)

# -------------------- DATA --------------------
DATA_PATHS = [
    "student_career_success_dataset.csv",
    "data/student_career_success_dataset.csv",
    "/kaggle/input/datasets/mobeenfatimah/student-career-success-prediction-dataset/student_career_success_dataset.csv"
]

@st.cache_data
def load_default_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df = load_default_data()

# Upload fallback
if df is None:
    uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🎓 CareerAI")
    st.caption("Student Career Prediction System")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Dashboard", "🔮 Predict", "📊 Analytics", "🤖 Model"])
    st.markdown("---")
    st.markdown("### Model")
    st.markdown("**Random Forest Classifier**")
    st.caption("200 trees • max depth 12")
    st.markdown("---")
    st.caption("Built with Python + Streamlit + Scikit-learn")

if df is None:
    st.markdown("""
    <div class="hero">
      <h1>🎓 Student Career Prediction</h1>
      <p>Upload the student career dataset from the sidebar to launch the ML dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

target = "Placement_Status"

# Train model using same core logic as notebook
@st.cache_resource
def train_model(data):
    X = data.drop(columns=[target]).copy()
    y_raw = data[target].astype(str)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return model, target_encoder, encoders, X.columns.tolist(), accuracy_score(y_test, pred), y_test, pred

model, target_encoder, encoders, feature_names, accuracy, y_test, y_pred = train_model(df)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <div class="badge">MACHINE LEARNING • CLASSIFICATION</div>
  <h1>🎓 Student Career Prediction</h1>
  <p>Predict placement outcomes using academic performance, skills, projects, internships and career-readiness indicators.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- DASHBOARD --------------------
if page == "🏠 Dashboard":
    placed = int((df[target] == "Placed").sum())
    not_placed = int((df[target] == "Not Placed").sum())

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("Students", f"{len(df):,}"),
        ("Placed", f"{placed:,}"),
        ("Placement Rate", f"{placed/len(df)*100:.1f}%"),
        ("Model Accuracy", f"{accuracy*100:.1f}%")
    ]
    for col,(title,value) in zip([c1,c2,c3,c4],metrics):
        col.markdown(f'<div class="card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Placement Overview</div>', unsafe_allow_html=True)
    left,right = st.columns([1.1,1])
    with left:
        fig,ax = plt.subplots(figsize=(7,4))
        counts = df[target].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Students")
        ax.set_title("Placement Status Distribution")
        ax.grid(axis="y", alpha=.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("""
        <div class="card">
        <h3>✨ What this app does</h3>
        <p>• Evaluates student career readiness</p>
        <p>• Predicts <b>Placed / Not Placed</b></p>
        <p>• Shows prediction confidence</p>
        <p>• Explores dataset patterns</p>
        <p>• Displays model performance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔎 Dataset Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

# -------------------- PREDICT --------------------
elif page == "🔮 Predict":
    st.markdown('<div class="section-title">🔮 Student Prediction</div>', unsafe_allow_html=True)
    st.caption("Enter the student's profile below and let the Random Forest model estimate placement status.")

    # Editable form for the useful student-facing fields
    form = {}
    cols = st.columns(3)

    numeric_defaults = {
        "Age": (18, 30, 21),
        "Attendance_Percentage": (50, 100, 80),
        "Study_Hours_Per_Week": (5, 45, 21),
        "CGPA": (2.0, 4.0, 3.1),
        "Programming_Skill": (1, 10, 7),
        "Projects_Completed": (0, 15, 8),
        "Certifications": (0, 8, 2),
        "Hackathons": (0, 10, 3),
        "Internships": (0, 5, 4),
        "Resume_Score": (44, 100, 85),
        "Communication_Skills": (3, 10, 8),
        "Teamwork": (2, 10, 7),
        "Problem_Solving": (1, 10, 7),
        "Interview_Score": (17, 100, 78),
        "Employability_Score": (82.3, 279.05, 214.0),
    }

    for i,col in enumerate(feature_names):
        if col == "Student_ID":
            form[col] = st.text_input("Student ID", "ST_DEMO_001", key=col)
        elif col in numeric_defaults:
            lo,hi,val = numeric_defaults[col]
            if col == "CGPA" or col == "Employability_Score":
                form[col] = st.number_input(col.replace("_"," "), min_value=float(lo), max_value=float(hi), value=float(val), key=col)
            else:
                form[col] = st.number_input(col.replace("_"," "), min_value=int(lo), max_value=int(hi), value=int(val), key=col)
        elif col in encoders:
            classes = list(encoders[col].classes_)
            default = classes[0]
            if col == "Gender" and "Female" in classes: default = "Female"
            if col == "University_Year" and "Junior" in classes: default = "Junior"
            if col == "Major" and "Computer Science" in classes: default = "Computer Science"
            form[col] = st.selectbox(col.replace("_"," "), classes, index=classes.index(default), key=col)
        else:
            # Existing notebook features such as Company_Tier, Career_Field and Placement_Mode
            # are retained to keep inference compatible with the trained model.
            classes = list(encoders[col].classes_)
            form[col] = st.selectbox(col.replace("_"," "), classes, key=col)

    if st.button("🚀 Predict Placement", type="primary", use_container_width=True):
        row = {}
        for col in feature_names:
            val = form[col]
            if col == "Student_ID":
                row[col] = int(encoders[col].transform([str(val)])[0])
            elif col in encoders:
                row[col] = int(encoders[col].transform([str(val)])[0])
            else:
                row[col] = val

        input_df = pd.DataFrame([row], columns=feature_names)
        probability = model.predict_proba(input_df)[0]
        pred_encoded = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        confidence = float(probability[pred_encoded] * 100)
        css_class = "placed" if pred_label == "Placed" else "notplaced"
        icon = "🎉" if pred_label == "Placed" else "📚"

        st.markdown(f"""
        <div class="prediction {css_class}">
          <div style="font-size:44px">{icon}</div>
          <h2>{pred_label}</h2>
          <p>Model confidence: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        a,b = st.columns(2)
        a.metric("Placed probability", f"{probability[list(target_encoder.classes_).index('Placed')]*100:.1f}%" if "Placed" in target_encoder.classes_ else "—")
        a.progress(float(probability[list(target_encoder.classes_).index('Placed')]) if "Placed" in target_encoder.classes_ else 0)
        b.metric("Predicted class", pred_label)
        b.metric("Confidence", f"{confidence:.1f}%")

        st.info("Note: the original notebook includes post-placement fields (such as company/career/placement mode/salary). For a production system, these should be removed before deployment to avoid target leakage.")

# -------------------- ANALYTICS --------------------
elif page == "📊 Analytics":
    st.markdown('<div class="section-title">📊 Student Analytics</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.hist(df["CGPA"], bins=20)
        ax.set_title("CGPA Distribution")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Students")
        ax.grid(axis="y", alpha=.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig,ax = plt.subplots(figsize=(7,4))
        grouped = df.groupby(target)["Interview_Score"].mean()
        ax.bar(grouped.index, grouped.values)
        ax.set_title("Average Interview Score by Placement")
        ax.set_ylabel("Average Score")
        ax.grid(axis="y", alpha=.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown('<div class="section-title">⭐ Top Model Features</div>', unsafe_allow_html=True)
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(12)

    fig,ax = plt.subplots(figsize=(9,5))
    ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", alpha=.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -------------------- MODEL --------------------
elif page == "🤖 Model":
    st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Trees", "200")
    c3.metric("Accuracy", f"{accuracy*100:.1f}%")

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_test, y_pred)
    fig,ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm)
    ax.set_xticks(range(len(target_encoder.classes_)))
    ax.set_yticks(range(len(target_encoder.classes_)))
    ax.set_xticklabels(target_encoder.classes_)
    ax.set_yticklabels(target_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="card">
    <h3>⚠️ Deployment note</h3>
    <p>The supplied notebook reports 100% test accuracy. That unusually high result should be interpreted carefully because the feature set contains fields that appear to describe placement outcomes themselves, including <b>Company_Tier</b>, <b>Career_Field</b>, <b>Placement_Mode</b>, and <b>Starting_Salary_USD</b>.</p>
    <p>A production-grade version should train only on information available <b>before</b> placement.</p>
    </div>
    """, unsafe_allow_html=True)
