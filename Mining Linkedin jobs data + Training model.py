from matplotlib import text
import requests
from bs4 import BeautifulSoup 
from fake_useragent import UserAgent
import time
import re
import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


HTML_DIR = BASE_DIR / "html_pages"
CSV_DIR = BASE_DIR / "csv_files"
MODELS_DIR = BASE_DIR / "models"  


HTML_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)




def fetch_and_save_html(url, path):

    session = requests.Session()
    session.trust_env = False

    headers = {
        'User-Agent': UserAgent().random,
        'Accept-Language' : 'en-US, en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection' : 'keep-alive',
        'Referer': 'https://www.google.com'
    }
    
    time.sleep(2)
    
    response = session.get(url, headers=headers, timeout=30)
    #response.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(response.text)

    extarction(HTML_DIR/"Linkdin_Jobs.html")



def extarction(path):
    with open(path, "r", encoding="utf-8") as f:
        html_doc  = f.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    print(soup.title)

    with open(HTML_DIR / "links.txt", "w", encoding="utf-8") as f:
        for link in soup.find_all('a'):
            if(str(link.get('href'))[:28]=="https://pk.linkedin.com/jobs"):
                #print(str(link.get('href'))[-8:])
                f.write(str(link.get('href'))+"\n")

    with open(HTML_DIR/"company links.txt", "w", encoding="utf-8") as f:
        for link in soup.find_all('a'):
            #print(str(link.get('href'))[-43:])
            if(str(link.get('href'))[-43:]=="_jobs_jserp-result_job-search-card-subtitle"):
                f.write(str(link.get('href'))+"\n")

    Job_Detail_Extractor()

def Job_Detail_Extractor():
    job = {
        "job_id": ...,
        "job_title": ...,
        "company": ...,
        "location": ...,
        "remote_work": ...,
        "Seniority level": ...,
        "description":...,
        "industry": ...,
        "experience_years":...,
        "skills_count": ...,
        #"skills_list"
        "company_size": ...,
        "education_level": ...,
        "job_posting_url": ...
    }

    pf = pd.DataFrame(data=[], columns=job.keys())
    pf.to_csv(CSV_DIR/"linkedin_jobs.csv",index=False)


    with open(HTML_DIR/"links.txt", "r") as outer_f, open(HTML_DIR/"company links.txt", "r") as outer_f2:
        for i, (link,link2) in enumerate(zip(outer_f,outer_f2)):
            print("Reading and Extracting data from job link :",i+1)
            
            session = requests.Session()
            session.trust_env = False

            headers = {
                'User-Agent': UserAgent().random,
                'Accept-Language' : 'en-US, en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection' : 'keep-alive',
                'Referer': 'https://www.google.com'
            }

            try:
                time.sleep(2)

                response = session.get(link, headers=headers, timeout=30)

                time.sleep(2)

                response2 = session.get(link2, headers=headers, timeout=30)
            except Exception as e:
                print(f"Network issue on interation {i+1}: {e}")
                continue


            skills_l = [
                "python", "java", "c", "c++", "c#", "javascript", "typescript", "php", 
                "ruby", "go", "swift", "kotlin", "r", "scala", "dart", "rust", 
                "html", "css", "react", "reactjs", "angular", "vue", "nextjs", 
                "bootstrap", "tailwind", "jquery", "sass", "less", "nodejs", "express", 
                "django", "flask", "fastapi", "spring", "springboot", "laravel", 
                "asp.net", ".net", "mysql", "postgresql", "sqlite", "mongodb", "oracle", 
                "sql server", "redis", "firebase", "cassandra", "aws", "azure", "gcp", 
                "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "linux", "nginx", 
                "machine learning", "deep learning", "data science", "numpy", "pandas", 
                "matplotlib", "seaborn", "scikit learn", "tensorflow", "keras", "pytorch", 
                "nlp", "opencv", "xgboost", "git", "github", "gitlab", "bitbucket", 
                "jira", "postman", "shell scripting", "android", "ios", "flutter", 
                "react native", "data structures", "algorithms", "oop", "system design", 
                "design patterns", "agile", "scrum", "testing", "unit testing", 
                "communication", "teamwork", "problem solving", "leadership", 
                "time management", "collaboration", "analytical thinking"
            ]

            education_list = ['High school','Diploma','Bachelor','Master','PhD']

            with open(HTML_DIR/"{i+1}.html", "w+", encoding="utf-8") as f, open(HTML_DIR/"company{i+1}.html", "w+", encoding="utf-8") as f2:
                f.write(response.text)
                f2.write(response2.text)
                f.seek(0)
                f2.seek(0)

                soup = BeautifulSoup(f.read(), 'html.parser')
                soup2 = BeautifulSoup(f2.read(), 'html.parser')
                #job_title = soup.find('div', class_='topbar__company-info-headersearch-bar__full-placeholder font-sans text-md text-color-text max-w-[calc(100%-40px)] text-left whitespace-nowrap overflow-hidden text-ellipsis')
                job_title = (
                    soup.select_one('.topbar__company-info-header')
                    or soup.select_one('.top-card-layout__title')
                    or soup.select_one('.topcard__title')
                    or soup.find('h1')
                    or soup.find('h3')
                )
                company = soup.find('a',class_='topcard__org-name-link topcard__flavor--black-link')
                location = soup.find('span', class_='topcard__flavor--bullet')
                Seniority = soup.find('span',class_='description__job-criteria-text description__job-criteria-text--criteria')
                description = soup.find('div',class_='show-more-less-html__markup') 

                #for key, value in skills_list.items():
                #    if key in description.get_text(separator=' ', strip=True) or key.upper() in description.get_text(separator=' ', strip=True) or key.capitalize() in description.get_text(separator=' ', strip=True):
                #        skills_list[key] = 1
                #        print(skills_list[key])
                #    else:
                #        #del skills_list[key]
                #        #print("false")
                #        skills_list[key] = 0
                description_text = description.get_text(separator=' ', strip=True) if description else ''
                pattern = rf"(\d+)\s*(to|-)?\s*(\d+)?\s*years?"
                experience = re.search(pattern, description_text, re.IGNORECASE)

                skill=""
                for key in skills_l:
                    pattern = rf"\b{re.escape(key)}\b"
                    
                    if re.search(pattern, description_text, re.IGNORECASE):
                        #skills_list[key] = 1
                        skill+=key+", "
                        #print(skills_list[key])
                    

                edu=[]
                for key in education_list:
                    pattern = rf"\b{re.escape(key)}\b"

                    if re.search(pattern, description_text, re.IGNORECASE):
                        #skills_list[key] = 1
                        edu.append(key)



                json_script = soup.find('script', type='application/ld+json')
                job_metadata = json.loads(json_script.string)
        
                #remote_worke = soup.find_all(class_='description__job-criteria-text description__job-criteria-text--criteria')
                remote_work = job_metadata.get("employmentType", "N/A")
                #print(remote_work) 
                category = job_metadata.get("industry") or job_metadata.get("category") or "N/A"

                #print(category)

                employee_element = soup2.find(attrs={"data-test-id": "about-us__size"})

                # Method B: Fallback search if the structured tag is missing
                if not employee_element:
                    # Look for any definition list metric containing the keyword "employees"
                    employee_element = soup2.find(lambda tag: tag.name in ["dd"] and "employees" in tag.text.lower())

                pattern =  r"\b\d{1,3}(?:,\d{3})*(?:\s*-\s*\d{1,3}(?:,\d{3})*|\s*\+)\s*employees"
                company_s = re.search(pattern, str(employee_element), re.IGNORECASE)

                #print(i+1, company_info.get_text(strip=True) if company_info else 'not found')
                job['job_id'] = i+1
                job['job_title'] = job_title.get_text(strip=True) if job_title else 'not found'
                job['company'] = company.get_text(strip=True) if company else 'not found'
                loc = location.get_text(strip=True) if location else 'not found'

                job['location'] = loc.split(",")[-1].strip() if loc != 'not found' else 'not found' 
                job['Seniority level'] = Seniority.get_text(strip=True) if Seniority else 'not found'
                job['remote_work'] = remote_work
                job['description'] = description.get_text(strip=True) if description else 'not found'
                job['experience_years'] = experience.group() if experience else 'not found'
                job['industry'] = category
                job['skills_count'] = len(skill[:-2].split(", ")) if skill else 0
                #job['skills_list'] = skill
                job['job_posting_url'] = link
                job['education_level'] = edu[0] if edu else ''
                #print(edu)
                job['company_size'] = company_s.group()[:-10] if company_s else 'not found'
                #print(employee_element)


                #print("Title :", job['title'])
                #print("Location :", job['location'])
                #print("Company :",job['company'])
                #print("Worktype :", job['WorkType'])
                #print("Skills :", job['skills'])
                #print("Experience :", job['experience'])
                #print("Description :", job["description"])
                #print("Url :", job["url"])

                writing_data_in_Excel(job)
            os.remove(HTML_DIR/"{i+1}.html")
            os.remove(HTML_DIR/"company{i+1}.html")
    cleaning()        


def writing_data_in_Excel(job):
    output_path = CSV_DIR/"linkedin_jobs.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pf = pd.DataFrame(data=[job])
    write_header = (not os.path.exists(output_path)) or os.path.getsize(output_path) == 0
    pf.to_csv(output_path, mode='a', header=write_header, index=False)



def merging():
    p1 = pd.read_csv(CSV_DIR/'cleaned_testing_data_linkedin_main2.csv')
    p2 = pd.read_csv(CSV_DIR/'cleaned_training_data_kaggle_main2.csv')
    df_concatenated = pd.concat([p1,p2], join='outer', ignore_index=True)
    df_concatenated.to_csv(CSV_DIR/'merged.csv', index=False)


    print("merged 100%")


def cleaning_for_input_data(data):
    if data == None :
        return
    droped_columns_l = pd.DataFrame([data], columns=['job_title', 'location', 'remote_work', 'industry', 'experience_years', 'skills_count', 'company_size', 'education_level'])

    droped_columns_l['remote_work']=droped_columns_l['remote_work'].replace({'full_time': 'No','part_time': 'hybrid', 'contract':'Contract','volunteer':'Hybrid'})


    droped_columns_l['industry'] = droped_columns_l['industry'].fillna('Other')


    droped_columns_l['experience_years'] = droped_columns_l['experience_years'].fillna(0)
    

    droped_columns_l['skills_count'] = droped_columns_l['skills_count'].fillna(0)

    droped_columns_l['education_level'] = droped_columns_l['education_level'].replace({'Phd':'PhD'})
    


    #print(droped_columns_l)
    return droped_columns_l




def cleaning():
    pfk = pd.read_csv(BASE_DIR/'job_salary_prediction_dataset.csv', low_memory=False)
    pfl = pd.read_csv(CSV_DIR/'linkedin_jobs.csv', low_memory=False)
    #print(pf.head())
    
    #columns_to_drop = ['skills', 'job_posting_url', 'description', 'company', 'publication_date', 'education_level', 'company_size', 'Seniority level', 'job_id']
    #droped_columns_k = pfk.drop(columns=['education_level', 'company_size',], errors='ignore')
    droped_columns_k = pfk.copy()
    droped_columns_l = pfl.drop(columns=['job_posting_url', 'description', 'company', 'Seniority level', 'job_id'], errors='ignore')
    #droped_columns = droped_columns.replace('not found', pd.NA)
    droped_columns_l['remote_work']=droped_columns_l['remote_work'].replace({'FULL_TIME': 'No','PART_TIME': 'Hybrid', 'CONTRACTOR':'Contract','VOLUNTEER':'Hybrid'})
    droped_columns_l['industry'] = droped_columns_l['industry'].apply(lambda x: x.lower() if pd.notnull(x) else x)
    droped_columns_l['industry'] = droped_columns_l['industry'].replace({
        'it services and it consulting': 'Technology',
        'software development': 'Technology',
        'mobile gaming apps': 'Technology',
        'internet publishing': 'Technology',
        'automation machinery manufacturing': 'Technology',
        'consumer services': 'Helpdesk',
        'accounting': 'Finance',
        'business consulting and services': 'Consulting',
        'sap/erp consulting': 'Consulting',
        'e-Learning providers': 'Technology',
        'telecommunications': 'Telecom',
        
    })

    tech_condition = (droped_columns_l['industry'].isna()) & (droped_columns_l['job_title'].str.contains('engineer|developer|dev', case=False, na=False))
    droped_columns_l.loc[tech_condition, 'industry'] = 'Technology'

    # 3. Check for "support" or "helpdesk" to fill "Helpdesk"
    help_condition = (droped_columns_l['industry'].isna()) & (droped_columns_l['job_title'].str.contains('support|help|desk', case=False, na=False))
    droped_columns_l.loc[help_condition, 'industry'] = 'Helpdesk'

    # 4. Fallback: If anything is still empty after keyword checking, assign 'Other'
    droped_columns_l['industry'] = droped_columns_l['industry'].fillna('Other')


    #numbers = re.findall(r'\d+', str(droped_columns_l['experience_required']))
    #print(droped_columns_l['experience_required'])
    droped_columns_l['company_size'] = droped_columns_l['company_size'].apply(calculate_experience_mean)
    droped_columns_l['company_size'] = droped_columns_l['company_size'].apply(calculate_caterory_from_mean)

    droped_columns_l['experience_years'] = droped_columns_l['experience_years'].apply(calculate_experience_mean)

    industry_means_l = droped_columns_l.groupby('industry')['experience_years'].transform('mean')
    industry_means_k = droped_columns_k.groupby('industry')['experience_years'].transform('mean')
    
    droped_columns_l['experience_years'] = droped_columns_l['experience_years'].fillna(industry_means_l.round())
    droped_columns_k['experience_years'] = droped_columns_k['experience_years'].fillna(industry_means_k.round())
    droped_columns_l['experience_years'] = droped_columns_l['experience_years'].fillna(0)
    droped_columns_k['experience_years'] = droped_columns_k['experience_years'].fillna(0)
    droped_columns_k['experience_years'] = droped_columns_k['experience_years'].apply(lambda x: int(x) if pd.notnull(x) else x)

    droped_columns_l = droped_columns_l.dropna(subset=['education_level'])
    #droped_columns_l['skills_count'] = droped_columns_l[]

    skills_means_k = droped_columns_k.groupby('industry')['skills_count'].transform('mean')
    skills_means_l = droped_columns_l.groupby('industry')['skills_count'].transform('mean')
    droped_columns_l['skills_count'] = droped_columns_l['skills_count'].fillna(skills_means_l)
    droped_columns_k['skills_count'] = droped_columns_k['skills_count'].fillna(skills_means_k)
    

    droped_columns_k['location'] = droped_columns_k['location'].replace({'India':'Pakistan'})

    droped_columns_k.to_csv(CSV_DIR/'cleaned_training_data_kaggle_main2.csv', index=False)
    droped_columns_l.to_csv(CSV_DIR/'cleaned_testing_data_linkedin_main2.csv', index=False)

    #merging()

def calculate_caterory_from_mean(text):    
    numbers = text

    if numbers == 0:
        return ''
    
    if numbers < 50:
        return 'Small'
    
    elif numbers < 200:
        return 'Medium'
    
    elif numbers < 1000:
        return 'Large'
    
    else:
        return 'Enterprise'

def calculate_experience_mean(text):    
    numbers = [int(x) for x in re.findall(r'\d+', str(text))]
    numbers.remove(0) if 0 in numbers else None
    #print(f"Extracted numbers from '{text}': {numbers}")

    if len(numbers) == 2:
        return sum(numbers) / 2
    
    elif len(numbers) == 1:
        return numbers[0]
    
    else:
        return None
    

def split_train_test(data, test_ratio):
    shuffled_idx = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)

    test_idx = shuffled_idx[:test_size]
    train_idx = shuffled_idx[test_size:]

    return data.iloc[train_idx], data.iloc[test_idx]


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
    

    return features


def training(data=None):


    pf = pd.read_csv(CSV_DIR/'cleaned_training_data_kaggle_main2.csv', low_memory=False)
    pf = pf.dropna(subset=['salary']).copy()

    train_set, test_set = split_train_test(pf, 0.2)


    cat_features = ['job_title', 'location', 'remote_work', 'industry', 'education_level', 'company_size']
    num_features = ['experience_years', 'skills_count', 'certifications']

    X = prepare_features(train_set, num_features, cat_features)
    y = train_set['salary']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dump(X_train, MODELS_DIR/'X_train.joblib')



    pip = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
#
    print("Training model...")
    #pip = load(MODELS_DIR/'salary_prediction_model.joblib')
    pip.fit(X_train, y_train)
    print("Done.")
    #print(X_train)

    dump(pip, MODELS_DIR/'salary_prediction_model.joblib')
    dump(num_features, MODELS_DIR/'num_features.joblib')
    dump(cat_features, MODELS_DIR/'cat_features.joblib')
        

    # evaluation
    preds = pip.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    plt.scatter(y_test, preds)
    plt.xlabel("Actual Salaries")
    plt.ylabel("Predicted Salaries")
    plt.title("Salary Prediction Performance")
    #plt.show()

    mean_error = mean_absolute_error(y_test, preds)
    dump(mean_error,MODELS_DIR/'mean_error.joblib')

    # external predictions
    if(data is not None):
        data_pf = data

        data_fea = prepare_features(data_pf, num_features, cat_features, reference_columns=X_train.columns)
        pred = pip.predict(data_fea)
        data_pf['predicted_salary'] = pred
        data_pf['min_predicted_salary'] = data_pf['predicted_salary'] - mean_absolute_error(y_test, preds)
        data_pf['max_predicted_salary'] = data_pf['predicted_salary'] + mean_absolute_error(y_test, preds)
        data_pf[['predicted_salary', 'min_predicted_salary', 'max_predicted_salary']] = data_pf[['predicted_salary', 'min_predicted_salary', 'max_predicted_salary']].astype("int")

        data_pf.to_csv(CSV_DIR/'test_input.csv',mode='a', index=False)
        print(f'Predicted Salary : Rs {pred} with the error : {mean_absolute_error(y_test, preds)}')

    ext_data = pd.read_csv(CSV_DIR/'cleaned_testing_data_linkedin_main2.csv', low_memory=False)
    ext_features = prepare_features(ext_data, num_features, cat_features, reference_columns=X_train.columns)
    ext_preds = pip.predict(ext_features)
    ext_data['predicted_salary'] = ext_preds
    ext_data['min_predicted_salary'] = ext_data['predicted_salary'] - mean_absolute_error(y_test, preds)
    ext_data['max_predicted_salary'] = ext_data['predicted_salary'] + mean_absolute_error(y_test, preds)

    ext_data[['predicted_salary', 'min_predicted_salary', 'max_predicted_salary']] = ext_data[['predicted_salary', 'min_predicted_salary', 'max_predicted_salary']].astype("int")
    ext_data.to_csv(CSV_DIR/'linkedin_jobs_with_salary_predictions.csv', index=False)
    print(r'Linkedin job file with predicted salary is generated at (D:\Projects\csv files\linkedin_jobs_with_salary_predictions.csv)')


def testing():
    #details = {
    #    'job_title': (input("Enter job title to predict salary: ") or 'not found').strip().capitalize(),
    #    'location': (input("Location: ") or 'not found').strip().capitalize(),
    #    'remote_work': (input("Remote work (Yes/No/Hybrid/Contract): ") or 'not found').strip().lower(),
    #    'industry': (input("Industry (Technology/Helpdesk/Finance/Consulting/Telecom/Other): ") or 'not found').strip().capitalize(),
    #    'experience_years': int((input("Experience years: ") or '0').strip()),
    #    'skills_count': int((input("Skills count: ") or '0').strip()),
    #    'certifications': int((input("Certifications count: ") or '0').strip()),
    #    'company_size': input("Company size (Small/Medium/Large/Enterprise) or 'not found').strip().capitalize()",
    #    'education_level': input("Education level (High School/Diploma/Bachelor/Master/PhD)" or 'not found').strip().capitalize()
    #}
    # Hard-coded test dictionary replacing the input() prompts
    details = {
        'job_title': 'Data scientist',
        'location': 'New york',
        'remote_work': 'full_time',
        'experience_years': 0,
        'industry': 'Technology',
        'certifications': 0,
        'skills_count': 0,
        'company_size': 'Large',
        'education_level': 'High School'
    }

    cleaned_data = cleaning_for_input_data(details)
    
    training(cleaned_data)


#main()
#url = "https://www.amazon.com/s?k=laptop&crid=1WXV3GPAFKK5L&sprefix=laptop%2Caps%2C445&ref=nb_sb_noss_1"
url = "https://www.linkedin.com/jobs/search?keywords=software&location=Lahore&geoId=104112529&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0"
#url = "https://pk.indeed.com/l-lahore-jobs.html?vjk=248404bb0b795f18"

path = HTML_DIR/"Linkdin_Jobs.html"




# ============================================================
# NOTE:
# - To mine fresh LinkedIn job data AND predict salaries for
#   each mined job -> UNCOMMENT the line below (fetch_and_save_html)
# - To SKIP mining and just train/evaluate the model on existing
#   cleaned data -> KEEP the line below COMMENTED OUT
# ============================================================
# fetch_and_save_html(url, path)



#fetch_and_save_html(url, path)

training()

#testing()


