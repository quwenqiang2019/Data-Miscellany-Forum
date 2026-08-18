import pandas as pd
from sklearn import preprocessing
from ctgan import CTGAN

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
hci_data.describe()
synthetic_data.describe()


#4、TableEvaluator效果评价
from table_evaluator import TableEvaluator
table_evaluator = TableEvaluator(hci_data, synthetic_data)
try:
    table_evaluator.visual_evaluation()
except:
    print()



#5、sdv.evaluation效果评价
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
# 元数据
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(hci_data)
print(metadata)

# 1. perform basic validity checks
diagnostic = run_diagnostic(hci_data, synthetic_data, metadata)
# 2. measure the statistical similarity
quality_report = evaluate_quality(hci_data, synthetic_data, metadata)
quality_report.get_details(property_name='Column Shapes')
print(quality_report.get_details(property_name='Column Shapes'))
# 3. plot the data
fig = get_column_plot(
    real_data=hci_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='CGPA'
)

fig.show()
