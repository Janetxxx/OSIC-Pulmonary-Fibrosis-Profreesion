# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#color
from colorama import Fore, Back, Style

import seaborn as sns
sns.set(style="whitegrid")

#pydicom
import pydicom

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()

from sklearn import preprocessing
from statsmodels.formula.api import quantreg
import os

# List files available
list(listdir("../input/osic-pulmonary-fibrosis-progression"))

IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
train_df.head(9)

test_df

train_df_fvc = np.array(train_df.FVC)
train_df_FVC = train_df_fvc.reshape(-1, 1) #transfer to 2D array
train_df_FVC

scaler = preprocessing.StandardScaler().fit(train_df_FVC)
scaler

scaler.mean_

scaler.scale_

train_df_fvc_scaled = scaler.transform(train_df_FVC)
train_df_fvc_scaled

test_df_fvc = np.array(test_df.FVC)
test_df_FVC = test_df_fvc.reshape(-1, 1) #transfer to 2D array
test_df_FVC

test_df_fvc_scaled = scaler.transform(test_df_FVC)
test_df_fvc_scaled

train_df_fvc_scaled

train_df_percent = np.array(train_df.Percent)
train_df_Percent = train_df_percent.reshape(-1, 1) #transfer to 2D array
train_df_Percent

#standardization percentage column
scaler = preprocessing.StandardScaler().fit(train_df_Percent)
scaler

scaler.mean_
scaler.scale_

train_df_percent_scaled = scaler.transform(train_df_Percent)
train_df_percent_scaled

train_df_percent_scaled.mean(axis=0) #zero mean

train_df_percent_scaled.std(axis=0) #unit variance

test_df_percent = np.array(test_df.Percent)
test_df_Percent = test_df_percent.reshape(-1, 1) #transfer to 2D array
test_df_Percent

#reapply to test_df in Percent column
test_df_percent_scaled = scaler.transform(test_df_Percent)
test_df_percent_scaled

# Encoding categorical features for sex
enc = preprocessing.OrdinalEncoder()

train_df_sex = np.array(train_df.Sex)
train_df_SEX = train_df_sex.reshape(-1, 1) #transfer to 2D array
train_df_SEX

enc.fit(train_df_SEX)

train_df_sex_encode = enc.transform(train_df_SEX)
train_df_sex_encode

#reapply to sex attribute on test set 
test_df_sex = np.array(test_df.Sex)
test_df_SEX = test_df_sex.reshape(-1, 1) #transfer to 2D array
test_df_SEX

enc.fit(test_df_SEX)

test_df_sex_encode = enc.transform(test_df_SEX)
test_df_sex_encode

# Encoding categorical features for SmokingStatus
train_df_smokingStatus = np.array(train_df.SmokingStatus)
train_df_SmokingStatus = train_df_smokingStatus.reshape(-1, 1) #transfer to 2D array
train_df_SmokingStatus

enc.fit(train_df_SmokingStatus)

train_df_smokingStatus_encode = enc.transform(train_df_SmokingStatus)
train_df_smokingStatus_encode

#reapply to sex attribute on test set 
test_df_smokingStatus = np.array(test_df.SmokingStatus)
test_df_SmokingStatus = test_df_smokingStatus.reshape(-1, 1) #transfer to 2D array
test_df_SmokingStatus

enc.fit(train_df_SmokingStatus)

test_df_smokingStatus_encode = enc.transform(test_df_SmokingStatus)
test_df_smokingStatus_encode

Age_column = np.array(train_df.Age).reshape(-1, 1)
Age_column

Weeks_column = np.array(train_df.Weeks).reshape(-1, 1)
Weeks_column

Patient_column = np.array(train_df.Patient).reshape(-1, 1)
Patient_column

newTrainTable = np.concatenate((Patient_column, Weeks_column,train_df_fvc_scaled,train_df_percent_scaled,Age_column,train_df_sex_encode,train_df_smokingStatus_encode),axis=1)
newTrainTable

Age_column_test = np.array(test_df.Age).reshape(-1, 1)
Age_column_test

Weeks_column_test = np.array(test_df.Weeks).reshape(-1, 1)
Weeks_column_test

Patient_column_test = np.array(test_df.Patient).reshape(-1, 1)
Patient_column_test

newTestTable = np.concatenate((Patient_column_test, Weeks_column_test,test_df_fvc_scaled,test_df_percent_scaled,Age_column_test,test_df_sex_encode,test_df_smokingStatus_encode),axis=1)
newTestTable

pd.DataFrame(newTrainTable).to_csv("traindata.csv")

# Analyse tabular data attribute distributions and interactions between sex groups 

g = sns.pairplot(train_df[['FVC', 'Weeks', 'Percent', 'Age']], aspect=1.4, height=5, diag_kind='kde', kind='reg')

g.axes[3, 0].set_xlabel('FVC', fontsize=30)
g.axes[3, 1].set_xlabel('Weeks', fontsize=30)
g.axes[3, 2].set_xlabel('Percent', fontsize=30)
g.axes[3, 3].set_xlabel('Age', fontsize=30)
g.axes[0, 0].set_ylabel('FVC', fontsize=30)
g.axes[1, 0].set_ylabel('Weeks', fontsize=30)
g.axes[2, 0].set_ylabel('Percent', fontsize=30)
g.axes[3, 0].set_ylabel('Age', fontsize=30)

g.axes[3, 0].tick_params(axis='x', labelsize=20)
g.axes[3, 1].tick_params(axis='x', labelsize=20)
g.axes[3, 2].tick_params(axis='x', labelsize=20)
g.axes[3, 3].tick_params(axis='x', labelsize=20)
g.axes[0, 0].tick_params(axis='y', labelsize=20)
g.axes[1, 0].tick_params(axis='y', labelsize=20)
g.axes[2, 0].tick_params(axis='y', labelsize=20)
g.axes[3, 0].tick_params(axis='y', labelsize=20)

g.fig.suptitle('Tabular Data Attribute Distributions and Interactions', fontsize=40, y=1.08)

plt.show()

# Analyse tabular data attribute distributions and interactions between smoking status groups.
g = sns.pairplot(train_df[['FVC', 'Weeks', 'Percent', 'Age', 'Sex']], hue='Sex', aspect=1, height=5, diag_kind='kde', kind='reg')

g.axes[3, 0].set_xlabel('FVC', fontsize=30)
g.axes[3, 1].set_xlabel('Weeks', fontsize=30)
g.axes[3, 2].set_xlabel('Percent', fontsize=30)
g.axes[3, 3].set_xlabel('Age', fontsize=30)
g.axes[0, 0].set_ylabel('FVC', fontsize=30)
g.axes[1, 0].set_ylabel('Weeks', fontsize=30)
g.axes[2, 0].set_ylabel('Percent', fontsize=30)
g.axes[3, 0].set_ylabel('Age', fontsize=30)

g.axes[3, 0].tick_params(axis='x', labelsize=20)
g.axes[3, 1].tick_params(axis='x', labelsize=20)
g.axes[3, 2].tick_params(axis='x', labelsize=20)
g.axes[3, 3].tick_params(axis='x', labelsize=20)
g.axes[0, 0].tick_params(axis='y', labelsize=20)
g.axes[1, 0].tick_params(axis='y', labelsize=20)
g.axes[2, 0].tick_params(axis='y', labelsize=20)
g.axes[3, 0].tick_params(axis='y', labelsize=20)

plt.legend(prop={'size': 20})
g._legend.remove()
g.fig.suptitle('Tabular Data Attribute Distributions and Interactions Between Sex Groups', fontsize=40, y=1.08)

plt.show()

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

sub

#process sample_submission
#split Patient_Week into Patient and Weeks
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub = sub[['Patient','Weeks','Confidence','Patient_Week']]

sub
print(test_df)
print("drop column Weeks in test_df temporary and merge with sample_submission")
sub = sub.merge(test_df.drop('Weeks', axis=1), on="Patient")
print (sub)

# mark train set as ‘train’, test set as ‘value’ and sample_submission set as ‘test’
# then append value set and test set after train set.
train_df['WHERE'] = 'train'
test_df['WHERE'] = 'value'
sub['WHERE'] = 'test' 
data = train_df.append([test_df, sub])

sub

print(train_df.shape, test_df.shape, sub.shape, data.shape)
print(train_df.Patient.nunique(), sub.Patient.nunique(), test_df.Patient.nunique(), data.Patient.nunique())

data

data['min_week'] = data['Weeks']
data

#set week sample submission be not a number
data.loc[data.WHERE=='test','min_week'] = np.nan
data

# min_week is the #week per patient measure 
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
data

#create a base to record the case weeks == min_week
base = data.loc[data.Weeks == data.min_week]
base

base = base[['Patient','FVC']].copy()
base

base.columns = ['Patient','initial_FVC']
base

# Return cumulative sum over DataFrame axis.
# if has lower one before, replace
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)
base

data = data.merge(base, on='Patient', how='left')
data['week_from_min_week'] = data['Weeks'] - data['min_week']
data

# extract sex and smoking status 
COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

# create new table column name
FE

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE_FVC'] = (data['initial_FVC'] - data['initial_FVC'].min() ) / ( data['initial_FVC'].max() - data['initial_FVC'].min() )
data['week'] = (data['week_from_min_week'] - data['week_from_min_week'].min() ) / ( data['week_from_min_week'].max() - data['week_from_min_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE_FVC']
FE

tr = data.loc[data.WHERE=='train']
ts= data.loc[data.WHERE=='value']
sample = data.loc[data.WHERE=='test']

# check
tr.shape, ts.shape, sample.shape

# Load image data
def load_scan(df, how="train"):
    xo = []
    p = []
    w  = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i,0]
        week = df.iloc[i,1]
        try:
            img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 
            im = np.array(im)
            xo.append(im[np.newaxis,:,:])
            p.append(patient)
            w.append(week)
        except:
            pass
    data = pd.DataFrame({"Patient":p,"Weeks":w})
    return np.concatenate(xo, axis=0), data

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
 
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

# build CNN model
def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="r1")(z)
    p1 = L.Dense(3, activation="linear", name="l1")(x)
    p2 = L.Dense(3, activation="relu", name="r3")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer="adam", metrics=[score])
    return model

net = make_model()
print(net.summary())
print(net.count_params())

x = tr['FVC'].values
z = tr[FE].values
ze = sample[FE].values
pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))

# predict the result by using cross validation, 
# K-fold cross validation. Wrap the total 7 rounds predict process, 
# get the mean value, to be the final predicted result.
from sklearn.model_selection import KFold
NFOLD = 7
kf = KFold(n_splits=NFOLD)

count = 0
for tr_idx, val_idx in kf.split(z):
    count += 1
    print(f"FOLD {count}")
    net = make_model()
    net.fit(z[tr_idx], x[tr_idx], batch_size=200, epochs=1000, 
            validation_data=(z[val_idx], x[val_idx]), verbose=0) #
    print("train value", net.evaluate(z[tr_idx], x[tr_idx], verbose=0, batch_size=300))
    print("value predict", net.evaluate(z[val_idx], x[val_idx], verbose=0, batch_size=300))
    pred[val_idx] = net.predict(z[val_idx], batch_size=300, verbose=0)
    pe += net.predict(ze, batch_size=300, verbose=0) / NFOLD


sample.head()

sample['FVC1'] = pe[:, 1]
sample['Confidence1'] = pe[:, 2] - pe[:, 0]

subm = sample[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

subm.loc[~subm.FVC1.isnull()].head(10)

unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

subm.head()

subm.describe().T

subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)