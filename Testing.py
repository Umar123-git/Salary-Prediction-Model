from pathlib import Path
import pandas as pd
from joblib import load

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

    return prediction_value


def testing():

    title = ''
    while(title==''):
        title = (input("Enter job title to predict salary: ")).strip().capitalize()
    
    loc=''
    while(loc==''):
        loc = (input("Location Country: ")).strip().capitalize()

    work = ''
    while(work=='' or work not in['Yes','No','Hybrid','Contract']):
        work = (input("Remote work (Yes/No/Hybrid/Contract): ")).strip().capitalize()

    indust = ''
    while(indust=='' or indust not in ['Technology','Helpdesk','Finance','Consulting','Telecom', 'Other']):
        indust = (input("Industry (Technology/Helpdesk/Finance/Consulting/Telecom/Other): ")).strip().capitalize()

    while True:
        try:
            exp = int(input("Experience years (0-70): "))
            if 0 <= exp <= 70:
                break
        except ValueError:
            pass
        print("Invalid input. Please enter a number between 0 and 70.")

    while True:
        try:
            skill_c = int(input("Skills count: "))
            if skill_c >= 0:
                break
        except ValueError:
            pass
        print("Invalid input. Please enter a positive number.")

    while True:
        try:
            cert = int(input("Certifications count: "))
            if cert >= 0:
                break
        except ValueError:
            pass
        print("Invalid input. Please enter a positive number.")
    
    
    comp_s = ''
    while(comp_s=='' or comp_s not in['Small','Medium','Large','Enterprise']):
        comp_s = (input("Company size (Small/Medium/Large/Enterprise): ")).strip().capitalize()

    edu_l = ''
    while(edu_l=='' or edu_l not in['high school','diploma','bachelor','master','phd']):
        edu_l = input("Education level (High School/Diploma/Bachelor/Master/PhD): ").strip().lower()
    edu_l = edu_l.capitalize()
        


    details = {
        'job_title': title,
        'location': loc,
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
