
import pandas as pd

data = pd.DataFrame({'name':['wencky','stany','barbio'],
                      'age':[29,29,3],
                      'gender':['w','m','m']})

print(data)
print('age去重',data["age"].unique(),sep='\n')
print('去重后数量',len(data["age"].unique()),sep='\n')