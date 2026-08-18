import pandas as pd

data={'state':[1,1,2,2,1,2,2],'pop':['a','b','c','d','b','c','d']}
frame=pd.DataFrame(data)

frame=frame.drop_duplicates(subset=['pop','state'])
print(frame)
