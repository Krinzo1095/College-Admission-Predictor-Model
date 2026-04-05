import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="College Admission Predictor",
    page_icon="🎓",
    layout="wide"
)
if "page" not in st.session_state:
    st.session_state.page = "Admission Evaluator"
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

st.title("🎓 College Admission Predictor")
st.markdown("### Using ID3 Decision Tree Algorithm")
st.markdown("---")

df = pd.read_csv('dataset/Admission_Predict(1).csv')
df.columns = df.columns.str.strip()
df.drop(columns=['Serial No.'], inplace=True)
df['Admitted'] = (df['Chance of Admit'] >= 0.75).astype(int)
df.drop(columns=['Chance of Admit'], inplace=True)
X = df.drop('Admitted', axis=1)
y = df['Admitted']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sidebar
st.sidebar.title("Navigation")
with st.sidebar:
    if st.button("Admission Evaluator", use_container_width=True):
        st.session_state.page = "Admission Evaluator"
    if st.button("Visualizations", use_container_width=True):
        st.session_state.page = "Visualizations"
    if st.button("Project Info", use_container_width=True):
        st.session_state.page = "Project Info"

def styled_slider(label, min_val, max_val, default, step=1, icon=""):
    if label not in st.session_state:
        st.session_state[label] = default

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"{icon} **{label}**")

    with col2:
        st.markdown(f"### {st.session_state[label]}")

    value = st.slider(
        label="",
        min_value=min_val,
        max_value=max_val,
        step=step,
        key=label   
    )

    return value

if st.session_state.page == "Admission Evaluator":
    st.subheader(" Enter Student Details")
    st.markdown("Fill in the details below to predict admission outcome.")
    st.markdown("")
    st.markdown("""
    <style>
    div[data-baseweb="slider"] {
        margin-top: -10px;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gre = styled_slider("GRE Score", 260, 340, 310)
        toefl = styled_slider("TOEFL Score", 92, 120, 107)
        cgpa = styled_slider("CGPA (out of 10)", 6.0, 10.0, 8.5, step=0.1)

    with col2:
        st.markdown(" **University Rating**")
        uni_rating = st.selectbox("", [1, 2, 3, 4, 5])
        sop = styled_slider("SOP Strength", 1.0, 5.0, 3.5, step=0.5)
        lor = styled_slider("LOR Strength", 1.0, 5.0, 3.5, step=0.5)
        st.markdown(" **Research Experience**")
        research = st.selectbox("", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")
        
    if gre < 280:
        st.warning(" GRE score is quite low for most universities")

    if cgpa < 7.0:
        st.warning(" Low CGPA may reduce admission chances")

    if research == 0:
        st.info(" Having research experience can improve your chances")

    # Dataframe
    input_data = pd.DataFrame([[gre, toefl, uni_rating, sop, lor, cgpa, research]],
        columns=['GRE Score', 'TOEFL Score', 'University Rating',
                 'SOP', 'LOR', 'CGPA', 'Research'])

    # Prediction
    if st.button(" Predict Admission", use_container_width=True):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            "Outcome": ["Admitted", "Rejected"],
            "Probability": [probability[1], probability[0]]
        })
        st.markdown("---")
        st.subheader(" Prediction Result")

        gauge_placeholder = st.empty()
        final_value = probability[1] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_value,
            title={'text': "Admission Chance (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green", 'thickness': 0.2},
                'steps': [
                    {'range': [0, 50], 'color': "#ff4d4d"},
                    {'range': [50, 75], 'color': "#ffd11a"},
                    {'range': [75, 100], 'color': "#66ff66"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 6},
                    'thickness': 1,
                    'value': final_value
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
        if prediction == 1:
            st.success("## ADMITTED!")
        else:
            st.error("##  NOT ADMITTED")

        st.markdown("###  Profile Analysis")

        if probability[1] > 0.8:
            st.success(" Strong Profile — very high chances!")
        elif probability[1] > 0.65:
            st.info(" Good Profile — decent chance")
        elif probability[1] > 0.5:
            st.warning(" Mediocre Profile — improve SOP/LOR/CGPA")
        else:
            st.error(" Low chances — improve profile")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Admission Chance", f"{probability[1]*100:.1f}%")
        with col4:
            st.metric("Rejection Chance", f"{probability[0]*100:.1f}%")

        st.markdown("---")
        st.subheader("Student Performance Summary")
        summary = pd.DataFrame({
            "Parameter": ["GRE Score", "TOEFL Score", "University Rating",
                          "SOP", "LOR", "CGPA", "Research"],
            "Value": [gre, toefl, uni_rating, sop, lor, cgpa,
                      "Yes" if research == 1 else "No"]
        })

        st.dataframe(summary, use_container_width=True)
#Page 2
if st.session_state.page == "Visualizations":
    st.subheader(" Model Visualizations")
    st.markdown("Interactive insights from the model")
    st.markdown("All graphs are generated after ID3 training")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Decision Tree",
        "Confusion Matrix",
        "Feature Importance",
        "Accuracy vs Tree Depth"
    ])
    with tab1:
        try:
            img = Image.open("model/decision_tree.png")
            st.image(img, use_column_width=True)
        except:
            st.error("Run train.py first")
        st.info("This decision tree shows how the model makes decisions one step at a time. It puts CGPA and GRE at the top of the list of important factors that affect admission outcomes.")

    with tab2:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Not Admitted", "Admitted"],
            y=["Not Admitted", "Admitted"],
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("The confusion matrix shows that most of the predictions are along the diagonal. This means that the model correctly classifies a lot of students. There aren't many misclassifications, which means the model is reliable, but it might still have trouble with borderline profiles.")

    with tab3:
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance")
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This chart illustrates how CGPA and GRE Score have the biggest effect on whether or not someone gets in, while things like SOP and LOR have a smaller effect. This means that how well someone does in school is the most important factor in the model's predictions.")

    with tab4:
        st.markdown("###  Accuracy vs Tree Depth")
        depths = list(range(1, 11))
        accuracies = [
        accuracy_score(y_test, DecisionTreeClassifier(max_depth=d, criterion="entropy")
                   .fit(X_train, y_train)
                   .predict(X_test))
        for d in depths
]
        fig = px.line(x=depths, y=accuracies, markers=True,
              labels={"x": "Depth", "y": "Accuracy"},
              title="Accuracy vs Tree Depth")
        st.plotly_chart(fig, use_container_width=True)
        st.info("At first, accuracy goes up as the model learns more patterns, but after a certain depth, it levels off. This shows that adding more complexity is detrimental and may even cause overfitting.")

#page 3
if st.session_state.page == "Project Info":
 
    st.subheader(" About ID3 Algorithm")
    st.markdown("---")
 
    st.markdown("""
    ## What is ID3?
    **ID3 (Iterative Dichotomiser 3)** is a classic Decision Tree algorithm
    developed by **Ross Quinlan in 1986**.
 
    ---
 
    ## How does it work?
    ID3 builds a tree by selecting the best feature at each step using:
 
    ### 1️⃣ Entropy
    Measures the **impurity** or **disorder** in the dataset.
    """)
 
    st.latex(r"Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")
 
    st.markdown("""
    - **Entropy = 0** → Pure node (all same class)
    - **Entropy = 1** → Maximum disorder (50/50 split)
 
    ### 2️⃣ Information Gain
    Measures how much a feature **reduces entropy** after splitting.
    """)
 
    st.latex(r"IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)")
 
    st.markdown("""
    - The feature with the **highest Information Gain** is chosen as the split node
    - This continues recursively until all nodes are pure or max depth is reached
 
    ---
 
    ## Why ID3 for this project?
    | Property | Value |
    |---|---|
    | Algorithm | ID3 Decision Tree |
    | Library | scikit-learn |
    | Criterion | entropy |
    | Use case | Classification |
    | Output | Admitted / Not Admitted |
 
    ---
 
    ## In scikit-learn:
    ```python
    DecisionTreeClassifier(criterion='entropy')
    ```
    Setting `criterion='entropy'` makes scikit-learn use
    **Information Gain** — which is exactly how ID3 works.
    """)