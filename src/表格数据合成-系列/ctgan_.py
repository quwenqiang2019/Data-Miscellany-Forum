import pandas as pd
from sklearn import preprocessing
from ctgan import CTGAN
from table_evaluator import TableEvaluator
from sdv.evaluation.single_table import evaluate_quality


# 1、准备数据
hci_data=pd.read_csv(r'HCI_Datasheet.csv')
hci_data=hci_data.drop(columns=['S. No','Decision Date','Application Date'])
hci_data['University_Program']=hci_data.University+' '+hci_data.Programme
hci_data.University=hci_data.University_Program
hci_data=hci_data.drop(columns=['Programme','University_Program','Year of Entry'])
le = preprocessing.LabelEncoder()
hci_data['Decision'] = le.fit_transform(hci_data['Decision']) ### 0-Accepted, 1-Rejected
hci_data['Research Experience'] = le.fit_transform(hci_data['Research Experience']) ###1-Yes, 0-No
hci_data['Submitted Portfolio'] = le.fit_transform(hci_data['Submitted Portfolio']) ###1-Yes, 0-No
hci_data['Student Status'] = le.fit_transform(hci_data['Student Status']) ### 0-Domestic, 1-International, 2-Undergrad domestic
hci_data['GRE']=hci_data['GRE'].fillna(330)
hci_data['TOEFL']=hci_data['TOEFL'].fillna(120)

#2、创建CTGAN对象并拟合数据
columns=list(hci_data.columns)
ctgan = CTGAN(epochs=100)
ctgan.fit(hci_data, columns)

#3、生成模拟数据
synthetic_data = ctgan.sample(50)
print(synthetic_data[:10])

#4、效果评价
table_evaluator = TableEvaluator(hci_data, synthetic_data)
try:
    table_evaluator.visual_evaluation()
except:
    print()
# 元数据
metadata ={
    "primary_key": "guest_email",
    "alternate_keys": [ "credit_card_number" ],
    "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
    "columns": {
        "guest_email": { "sdtype": "email", "pii": true },
        "has_rewards": { "sdtype": "boolean" },
        "room_type": { "sdtype": "categorical" },
        "amenities_fee": { "sdtype": "numerical" },
        "checkin_date": { "sdtype": "datetime", "datetime_format": "%d %b %Y" },
        "checkout_date": { "sdtype": "datetime", "datetime_format": "%d %b %Y" },
        "room_rate": { "sdtype": "numerical" },
        "billing_address": { "sdtype": "address", "pii": true },
        "credit_card_number": { "sdtype": "credit_card_number", "pii": true }
    }
}
print(evaluate_quality(hci_data, synthetic_data, metadata=metadata))

hci_data.describe()
synthetic_data.describe()