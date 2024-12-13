import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


s = pd.read_csv("social_media_usage.csv")


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']),
    'income': np.where(s['income'] > 9, np.nan, s['income']),
    'education': np.where(s['educ2'] > 8, np.nan, s['educ2']),
    'parent': np.where(s['par'] == 1, 1, 0),
    'married': np.where(s['marital'] == 1, 1, 0),
    'female': np.where(s['gender'] == 2, 1, 0),
    'age': np.where(s['age'] > 98, np.nan, s['age'])
})

ss = ss.dropna()

y = ss['sm_li']
x = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=9899) # set for reproducibility

lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)

def input(income, education, parent, married, female, age):
    user = [income, education, parent, married, female, age]
    predicted_class = lr.predict([user])
    probs = lr.predict_proba([user])
    return predicted_class[0], probs[0][1]

st.title("LinkedIn User Prediction")

st.write("""
This application predicts if a person is likely to be a LinkedIn user based on the following attributes:
- Income level (1: Less than $10k, 2: $10-20k, 3: $20-30k, 4: $30-40k, 5: $40-50k, 6: $50-75k, 7: $75-100k, 8: $100-150k, 9: More than $150k+)
- Education level (1: Less than high school, 2: High school incomplete, 3: High school degree, 4: Some college, 5: Associate's degree, 6: Bachelor's degree, 7: Some postgraduate schooling, 8: Postgraduate degree)
- Whether they are a parent
- Marital status
- Gender
- Age
""")

# Input fields
income = st.number_input("Income Level (1-9)", min_value=1, max_value=9, step=1)
education = st.number_input("Education Level (1-8)", min_value=1, max_value=8, step=1)
parent = st.number_input("Parent (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)
married = st.number_input("Married (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)
female = st.number_input("Female (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)
age = st.number_input("Age", min_value=18, max_value=98, step=1)

# Prediction
if st.button("Predict"):
    predicted_class, probability = input(income, education, parent, married, female, age)
    st.write(f"Predicted Class: {'LinkedIn User' if predicted_class == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of being a LinkedIn User: {probability:.4f}")