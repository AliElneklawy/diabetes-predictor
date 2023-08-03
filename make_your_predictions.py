import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
import joblib


def get_input():

    user_inputs = {}
    user_inputs['gender'] = input("Gender (Male/Female): ").title()
    user_inputs['age'] = float(input("Age: "))
    user_inputs['hypertension'] = input("Does the patient have hypertension (yes/no)? ")
    user_inputs['heart_disease'] = input("Does the patient have history of heart disease (yes/no)? ")
    user_inputs['smoking_history'] = input("Is the patient smoker (never/no_info/former/current/not_current/ever)? ")
    user_inputs['bmi'] = float(input("BMI: "))
    user_inputs['HbA1c_level'] = float(input('HbA1c level: '))
    user_inputs['blood_glucose_level'] = float(input("Blood glucose level: "))

    user_data = pd.DataFrame([user_inputs])
    user_data = feat_eng(user_data)

    return user_data

def feat_eng(df):
    df['AgeCat'] = pd.cut(df['age'],
                           bins=[-np.inf, 1, 12, 18, 65, np.inf],
                           labels=['infant', 'child', 'teenager', 'adult', 'older_adult'])

    df['BMICat'] = pd.cut(df['bmi'],
                            bins=[-np.inf, 18.5, 25, 30, np.inf],
                            labels=['underweight', 'normal', 'overweight', 'obese'])

    df['GlucoseCat'] = pd.cut(df['blood_glucose_level'],
                                bins=[-np.inf, 140, 200, np.inf],
                                labels=['normal', 'impaired', 'diabetic'])
    
    df['HbA1cCat'] = pd.cut(df['HbA1c_level'],
                            bins=[0, 5.6, 6.4, np.inf],
                            labels=['normal', 'prediabetic', 'diabetic'])
    return df


reloaded_model = joblib.load("/home/elneklawy/Desktop/diabetes prediction (better data set)/final_model.pkl")
input = get_input()
numerical_columns_except_bmi = list(make_column_selector(dtype_include=np.number)(input))
numerical_columns_except_bmi.remove('bmi')
print()
print(input.loc[:, 'AgeCat':'HbA1cCat'])
print()
prediction = reloaded_model.predict(input)    
print(f"There is a {(reloaded_model.predict_proba(input)[0][1] * 100).round(2)}% chance that the patient is diabetic.")
print('My guess: diabetic.') if prediction == 1 else print('My guess: non-diabetic.')


num_pl = Pipeline([
    ('imputenum', SimpleImputer(strategy='median')),
    ('stdscale', StandardScaler())
])

cat_pl = Pipeline([
    ('imputecat', SimpleImputer(strategy='most_frequent')),
    ('onehotencode', OneHotEncoder(handle_unknown='ignore'))
])

log_pl = Pipeline([
    ('imputelog', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
    ('stdscalelog', StandardScaler())
])


preprocessing = ColumnTransformer([
    ('num_pipeline', num_pl, numerical_columns_except_bmi),
    ('log_transform', log_pl, ['bmi']),
    ('cat_pipeline', cat_pl, make_column_selector(dtype_include=['category', 'object']))
])
