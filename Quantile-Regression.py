import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
import matplotlib.pyplot as plt
%matplotlib inline
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
orama import Fore, Back, Style
import seaborn as sns
sns.set(style="whitegrid")
import pydicom
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from statsmodels.formula.api import quantreg
import os

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test_df["Sex"] = test_df["Sex"].apply(lambda x: 0 if str(x)=='Male' else 1)
test_df["SmokingStatus"] = test_df["SmokingStatus"].apply(lambda x: 1 if str(x)=='Ex-smoker' else 2 if str(x)=='Never smoked' else 0)


train_df["Sex"] = train_df["Sex"].apply(lambda x: 0 if str(x)=='Male' else 1)
train_df["SmokingStatus"] = train_df["SmokingStatus"].apply(lambda x: 1 if str(x)=='Ex-smoker' else 2 if str(x)=='Never smoked' else 0)


train_df['Percent']       = (train_df['Percent'] - train_df['Percent'].mean()) / train_df['Percent'].std()
train_df['Age']           = (train_df['Age'] - train_df['Age'].mean()) / train_df['Age'].std()
train_df['Sex']           = (train_df['Sex'] - train_df['Sex'].mean()) / train_df['Sex'].std()
train_df['SmokingStatus'] = (train_df['SmokingStatus'] - train_df['SmokingStatus'].mean()) / train_df['SmokingStatus'].std()

test_df['Percent']       = (test_df['Percent'] - test_df['Percent'].mean()) / test_df['Percent'].std()
test_df['Age']           = (test_df['Age'] - test_df['Age'].mean()) / test_df['Age'].std()
if test_df['Sex'].std() == 0:
    test_df['Sex']           = (test_df['Sex'] - test_df['Sex'].mean()) / (test_df['Sex'].std() + 1)
else:
    test_df['Sex']           = (test_df['Sex'] - test_df['Sex'].mean()) / test_df['Sex'].std()
# test_df['Sex']           = (test_df['Sex'] - test_df['Sex'].mean()) / test_df['Sex'].std()
test_df['SmokingStatus'] = (test_df['SmokingStatus'] - test_df['SmokingStatus'].mean()) / test_df['SmokingStatus'].std()

ffff_test = pd.merge(test_df, train_df, on=['Patient'], how='left')
ffff_test = ffff_test[["Patient","Weeks_y","FVC_y","Percent_y","Age_y","Sex_y","SmokingStatus_y"]]

ffff_test = ffff_test[["Patient","Weeks_y","FVC_y","Percent_y","Age_y","Sex_y","SmokingStatus_y"]]
ffff_test.rename(columns={'Weeks_y': 'Weeks', 'FVC_y': 'FVC', 'Percent_y': 'Percent','Age_y': 'Age','Sex_y': 'Sex','SmokingStatus_y': 'SmokingStatus'}, inplace=True)



modelL = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.15 )
model  = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.50 )
modelH = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.85 )

new_test_data = test_df.copy()
new_test_data = new_test_data[0:0]
for i, D in enumerate(test_df.Patient): 
    for week in range(-12,134):
        new_row = [test_df.iloc[i,0],week, test_df.iloc[i,2], test_df.iloc[i,3], test_df.iloc[i,4],test_df.iloc[i,5],test_df.iloc[i,6]]
        new_test_data.loc[len(new_test_data)] = new_row

i = 0
for row in new_test_data.iterrows():
    cur_week = row[1][1]
    temp_dataframe = ffff_test[ffff_test.Patient == row[1][0]]
    result_index = temp_dataframe['Weeks'].sub(cur_week).abs().idxmin()
    result = temp_dataframe.loc[result_index]
    result.at['Weeks'] = cur_week
    new_test_data.loc[i] = result
    i = i + 1
    
new_test_data['ypredL'] = modelL.predict( new_test_data ).values
new_test_data['ypred']  = model.predict( new_test_data ).values
new_test_data['ypredH'] = modelH.predict( new_test_data ).values
new_test_data['ypredstd'] = 0.5*np.abs(new_test_data['ypredH'] - new_test_data['ypred'])+0.5*np.abs(new_test_data['ypred'] - new_test_data['ypredL'])



submission = new_test_data[["Patient","Weeks","ypred","ypredstd"]]

for i in range(0,len(submission)):
    submission.loc[i,"Patient"] = str(submission.loc[i,"Patient"])+"_" + str(submission.loc[i,"Weeks"])
    
submission = submission[["Patient","ypred","ypredstd"]]
submission.rename(columns={'Patient': 'Patient_Week',"ypred":"FVC","ypredstd":"Confidence"}, inplace=True)

submission.to_csv('submission.csv', index=False)
