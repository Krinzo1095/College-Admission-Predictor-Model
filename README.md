# 🎓 College Admission Predictor

A machine learning project that predicts whether a student will be admitted to college based on academic performance and extracurricular activities. Built as a case study on the ID3 Decision Tree algorithm for an Artificial Intelligence course.

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📋 Input Parameters
| Parameter | Description |
|---|---|
| 📚 12th Grade Marks | Percentage scored in 12th grade |
| 📝 Entrance Exam Score | Score in college entrance exam |
| 🏫 Attendance | Attendance percentage |
| 🏆 Extracurriculars | Participation in activities (Yes/No) |

---

## 🚀 How to Run

\```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
\```

---

## 🤖 Algorithm
ID3 Decision Tree uses **Entropy** and **Information Gain** to split nodes and build the tree. Implemented using `DecisionTreeClassifier(criterion='entropy')` from scikit-learn.

---

## 📊 Dataset
500 student records with admission outcome as the target variable.
