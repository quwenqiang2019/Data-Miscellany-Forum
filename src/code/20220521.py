import numpy as np
import pandas as pd

def Read_data(file):
    dt = pd.read_csv(file)
    dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
                  'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina',
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    data =dt
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    print(data.head())
    return data

def data_encoding(data):
    #========================数据编码===========================
    data = data[["age", 'sex', "chest_pain_type", "resting_blood_pressure", "cholesterol", "fasting_blood_sugar", "rest_ecg",
         "max_heart_rate_achieved", "exercise_induced_angina", "st_depression", "st_slope", "num_major_vessels","thalassemia"]]
    Discretefeature=['sex',"chest_pain_type", "fasting_blood_sugar", "rest_ecg",
          "exercise_induced_angina",  "st_slope", "thalassemia"]
    Continuousfeature=["age", "resting_blood_pressure", "cholesterol","max_heart_rate_achieved","st_depression","num_major_vessels"]

    df = pd.get_dummies(data,columns=Discretefeature)
    print(df.head())

    df[Continuousfeature]=(df[Continuousfeature]-df[Continuousfeature].mean())/(df[Continuousfeature].std())
    print(df.head())
    return df

if __name__=="__main__":
    data1=Read_data("F:\数据杂坛\\0504\heartdisease\Heart-Disease-Data-Set-main\\UCI Heart Disease Dataset.csv")
    data2=data_encoding(data1)
