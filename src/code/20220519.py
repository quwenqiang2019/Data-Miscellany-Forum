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
    print(data.head().append(dt.tail()))
    return data

def Segment_statistics(data):
    age = data[["age"]]
    bins = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    age2 = pd.cut(age.values.flatten(), bins=bins)
    print(age2)
    print(age2.value_counts())
    age2 = pd.DataFrame(age2, columns=["年龄段"])  #
    age3 = pd.concat([age, age2], axis=1)
    print(age3)
    return

if __name__=="__main__":
    data1=Read_data("F:\数据杂坛\\0504\heartdisease\Heart-Disease-Data-Set-main\\UCI Heart Disease Dataset.csv")
    Segment_statistics(data1)
