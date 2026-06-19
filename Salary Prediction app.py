from pathlib import Path
import pandas as pd
from joblib import load
import streamlit as st

# Dynamic root setup to look in the folder where this script lives
BASE_DIR = Path(__file__).resolve().parent


HTML_DIR = BASE_DIR / "html_pages"
CSV_DIR = BASE_DIR / "csv_files"
MODELS_DIR = BASE_DIR / "models"  


def build_input_frame(data):
    if data == None :
        return
    droped_columns_l = pd.DataFrame([data], columns=['job_title', 'location', 'remote_work', 'industry', 'experience_years', 'skills_count', 'company_size', 'education_level', 'certifications'])


    droped_columns_l['industry'] = droped_columns_l['industry'].fillna('Other')


    droped_columns_l['experience_years'] = droped_columns_l['experience_years'].fillna(0)
    

    droped_columns_l['skills_count'] = droped_columns_l['skills_count'].fillna(0)

    droped_columns_l['education_level'] = droped_columns_l['education_level'].replace({'Phd':'PhD','High school':'High School'})
    


    print(droped_columns_l)
    return droped_columns_l


def prepare_features(data, numeric_features, categorical_features, reference_columns=None):
    df = data.copy()


    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())


    for col in categorical_features:
        if col not in df.columns:
            df[col] = 'Unknown'

    dummies = pd.get_dummies(df[categorical_features], drop_first=True)
    features = pd.concat([df[numeric_features], dummies], axis=1)

    if reference_columns is not None:
        features = features.reindex(columns=reference_columns, fill_value=0)
    #print(features.head())
    

    return features



def prediction(frame):

    #cat_features = ['job_title', 'location', 'remote_work', 'industry', 'education_level', 'company_size']
    #num_features = ['experience_years', 'skills_count', 'certifications']

    cat_features = load(MODELS_DIR / 'cat_features.joblib')
    num_features = load(MODELS_DIR / 'num_features.joblib')

    model = load(MODELS_DIR / 'salary_prediction_model.joblib')
    X_train = load(MODELS_DIR / 'X_train.joblib')
    mae = load(MODELS_DIR / 'mean_error.joblib')
    #print("qwerty",X_train, X_train.dtypes)
    
    prepared_data = prepare_features(frame, num_features, cat_features, reference_columns=X_train.columns)

    prediction_value = model.predict(prepared_data)


    print('Model loaded successfully.')
    print(f'Predicted Salary: Rs.{prediction_value[0]:.2f} ± Rs.{mae:.2f}')

    st.success(f'Predicted Salary: Rs.{prediction_value[0]:.2f} ± Rs.{mae:.2f}')

    return prediction_value


def testing():

    st.title("Salary Prediction")

    title = st.text_input("Enter job title to predict salary:")

    loc = st.text_input("Location Country:")

    work = st.selectbox("Remote work:", ['Yes','No','Hybrid','Contract'])

    indust = st.selectbox("Industry:", ['Technology','Helpdesk','Finance','Consulting','Telecom', 'Other'])

    exp = st.number_input("Experience years (0-70):", min_value=0, max_value=70, step=1)

    skill_c = st.number_input("Skills count:", min_value=0, step=1)

    cert = st.number_input("Certifications count:", min_value=0, step=1)

    comp_s = st.selectbox("Company size:", ['Small','Medium','Large','Enterprise'])

    edu_l = st.selectbox("Education level:", ['High School','Diploma','Bachelor','Master','PhD'])

    if st.button("Predict"):

        if title=='' or loc=='':
            st.warning("Please fill job title and location.")
            return

        details = {
            'job_title': title.strip().capitalize(),
            'location': loc.strip().capitalize(),
            'remote_work': work,
            'industry': indust,
            'experience_years': exp,
            'skills_count': skill_c,
            'certifications': cert,
            'company_size': comp_s,
            'education_level': edu_l,
        }

        input_frame = build_input_frame(details)
        prediction(input_frame)



testing()