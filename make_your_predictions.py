import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
import joblib

def feat_eng(df):
    df['AgeCat'] = pd.cut(df['Age'],
                           bins=[0, 1, 12, 18, 65, np.inf],
                           labels=['infant', 'child', 'teenager', 'adult', 'older_adult'])

    df['BMICat'] = pd.cut(df['BMI'],
                            bins=[-np.inf, 18.5, 25, 30, np.inf],
                            labels=['underweight', 'normal', 'overweight', 'obese'])

    df['GlucoseCat'] = pd.cut(df['Glucose'],
                                bins=[-np.inf, 140, 200, np.inf],
                                labels=['normal', 'impaired', 'diabetic'])

    df['BloodPressCat'] = pd.cut(df['BloodPressure'],
                                    bins=[-np.inf, 80, 89, 120, 140],
                                    labels=['normal', 'stage_1', 'stage_2', 'Hypertensive_Crisis'])
    return df

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

preprocessing = make_column_transformer(
    (num_pl, ['Glucose', 'BloodPressure', 'BMI']),
    (log_pl, ['Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']),
    (cat_pl, make_column_selector(dtype_include='category'))
)


def get_input():

    user_inputs = {}
    user_inputs['Pregnancies'] = float(input("Number of Pregnancies: "))
    user_inputs['Glucose'] = float(input("Glucose level: "))
    user_inputs['BloodPressure'] = float(input("Diastolic blood pressure: "))
    user_inputs['SkinThickness'] = float(input("Skin thickness: "))
    user_inputs['Insulin'] = float(input("Insulin level: "))
    user_inputs['BMI'] = float(input("BMI: "))
    user_inputs['DiabetesPedigreeFunction'] = float(input('Diabetes pedigree function: '))
    user_inputs['Age'] = float(input("Age: "))

    user_data = pd.DataFrame([user_inputs])
    user_data = feat_eng(user_data)

    return user_data


if __name__ == '__main__':
    
    reloaded_model = joblib.load("/home/elneklawy/Desktop/New Folder 1/final_model.pkl")
    input = get_input()
    print()
    print(input.loc[:, 'AgeCat':'BloodPressCat'])
    print()
    prediction = reloaded_model.predict(input)    
    print(f"There is a {(reloaded_model.predict_proba(input)[0][1] * 100).round(2)}% chance that the patient is diabetic.")
    print('My guess: diabetic.') if prediction == 1 else print('My guess: non-diabetic.')

    
