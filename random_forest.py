import os
import pydicom

import pandas as pd
import numpy as np
import random
import plotly.graph_objs as go

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from skimage import measure, morphology
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas_profiling as pdp

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,make_scorer

#plot
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

    

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

    
train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

train_df.head(10)
    
train_df.groupby(['Patient']).count().head(10)

# get a overview of the train.csv
profile_train_df = pdp.ProfileReport(train_df)
profile_train_df
profile_train_df.to_file(output_file="overview")

# ========================== PLOTING ==============================

# create unique patients dataset for exploring relation between 
# features ["Patient", "Age", "Sex", "SmokingStatus"]
train_df_unique = train_df[["Patient", "Age", "Sex", "SmokingStatus"]]
train_df_unique = train_df_unique.drop_duplicates()

# pie chart of smokingstatus distribution
train_df_unique['SmokingStatus'].value_counts().plot.pie(autopct = '%1.2f%%')

# histgram of FVC distribution
plt.hist(train_df['FVC'], bins=30)
plt.xlabel('FVC')
plt.ylabel('count')
plt.show()

# Age distribution wrt Sex 
fig = px.histogram(train_df_unique, x='Age',color = 'Sex',color_discrete_map={'Male':'#2684b2','Female':'#f7b4a9'},marginal = 'rug',hover_data = train_df_unique.columns)
fig.update_layout(title = 'Age distribution wrt Sex')
fig.update_traces(marker_line_color='blue',marker_line_width=1, opacity=0.85)
fig.show()

# line chart of weeks wrt FVC on a particular patient 
p=train_df[train_df.Patient == 'ID00419637202311204720264']
figure = px.line(p, x="Weeks", y="FVC")
figure.show()

# kde ploting of SmokingStatus wrt FVC Percent
for x in ["Currently smokes","Ex-smoker", "Never smoked"]:
    train_df.Percent[train_df.SmokingStatus == x].plot(kind="kde")
plt.title("SmokingStatus and Percent")
plt.legend(("Currently smokes", "Ex-smoker", "Never smoked"))

    
# ========================== DATA CLEANING AND PREPROCESSING ==============================

def data_cleaning(df):
    df.loc[df['Sex'] == "Male", "Sex"] = 0
    df.loc[df['Sex'] == "Female", "Sex"] = 1

    # a more simple and elegant way to clean the data: map
    df['SmokingStatus'] = df['SmokingStatus'].map( {'Ex-smoker': 1, 'Never smoked': 2,'Currently smokes': 0} ).astype(int)


    
train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

    
ID = train_df.Patient
train_df = train_df.drop(['Patient'], axis=1)
data_cleaning(train_df)
col_names = train_df.columns

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# transform to 2D array
train_df = sc.fit_transform(train_df)

# transform back to final dataframe
train_df = pd.DataFrame(train_df)
train_df.columns = col_names

# copy the dataframe, set a check point
c = train_df


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)

    
# ============================== RANDOM FOREST MODEL ==============================

X = train_df.drop(["FVC"],axis=1).values
y = train_df['FVC'].values

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=0)

rgs = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
rgs.fit(test_x, test_y)

y_hat = rgs.predict(test_x)
laplace_log_likelihood(test_y, y_hat, np.std(y_hat))


    
train_df_nos = c.iloc[:,:]
v = train_df_nos.iloc[:, :]

# dataframe with scaling back to train_df
train_df_nos = sc.inverse_transform(train_df_nos)
train_df_nos = pd.DataFrame(train_df_nos)
train_df_nos.columns = col_names

# dataframe contains FCV_pred
y_p = rgs.predict(X)
v.FVC = y_p
v = sc.inverse_transform(v)
v = pd.DataFrame(v)
v.columns = ['Weeks', 'FVC_pred', 'Percent', 'Age', 'Sex', 'SmokingStatus']

laplace_log_likelihood(train_df_nos['FVC'].values, v['FVC_pred'].values, np.std(train_df_nos['FVC'].values))


from sklearn.model_selection import cross_val_predict

X = train_df.drop(["FVC"],axis=1)
y = train_df['FVC']

kf = KFold(n_splits=5, shuffle= True)
Rs = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)

scores = []

for i in range(10):
    result = next(kf.split(X), None)
    train_x = X.iloc[result[0]]
    test_x = X.iloc[result[1]]
    train_y = y.iloc[result[0]]
    test_y = y.iloc[result[1]]
    model = Rs.fit(train_x, train_y)
    y_hat = Rs.predict(test_x)
    print(laplace_log_likelihood(test_y, y_hat, np.std(y)))
    
    scores.append(model.score(test_x, test_y))
print('\nScores from each iteraion: ', scores)
print('Average kFold score: ', np.mean(scores))



    
# ======================== Quantile regression for comparison ========================

import statsmodels.formula.api as smf
modelL = smf.quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.15 )
model  = smf.quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.50 )
modelH = smf.quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.85 )

    
train_df['ypred_L'] = modelL.predict(train_df).values
train_df['ypred']  = model.predict(train_df).values
train_df['ypred_H'] = modelH.predict(train_df).values

    
v1 = train_df.iloc[:, :6]
v1.FVC = train_df['ypred']
v2 = sc.inverse_transform(v1)
v3 = pd.DataFrame(v2)
v3.columns = ['Weeks', 'FVC_pred', 'Percent', 'Age', 'Sex', 'SmokingStatus']
v3

    
train_df_nos = train_df.iloc[:, :6]
train_df_nos = sc.inverse_transform(train_df_nos)
train_df_nos = pd.DataFrame(train_df_nos)
train_df_nos.columns = col_names

train_df_nos['FVC_pred'] = v3['FVC_pred']
train_df_nos

    
laplace_log_likelihood(train_df_nos['FVC'].values, train_df_nos['FVC_pred'].values, np.std(train_df_nos['FVC'].values))

# ============================ Prepare submission.csv ============================

sample_df = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )
sample_df['Weeks']   = sample_df['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )
sample_df['Patient'] = sample_df['Patient_Week'].apply( lambda x: x.split('_')[0] ) 
sample_df = sample_df.drop(['FVC'], axis=1)
sample_df

    
cp = train_df_nos

cp = cp.iloc[:, 1:7]
cp['Patient'] = ID
test = pd.merge(sample_df, cp, on='Patient', how='left' )
test.sort_values( ['Patient','Weeks'], inplace=True )
test

    
test['Confidence'] = np.abs(test['FVC'] - test['FVC_pred'])

test[['Patient_Week','FVC', 'FVC_pred','Confidence']].head(10)
sub = test[['Patient_Week','FVC_pred','Confidence']]
sub.columns = ['Patient_Week', 'FVC', 'Confidence']
sub

    
# calculate confidence
# import forestci as fci


sub.to_csv('submission.csv', index=False)
sub.head(10)

# ====================== CT Scan Image Proccesing Functions (haven't use yet!) =========================

# try use the CT scan
import pandas as pd
import numpy as np
import random as rnd
import re
%matplotlib inline
from skimage import measure, morphology
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import pydicom


#load the slices for each patient
def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


    
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


    
def get_tissue(patient_pixels,tissue_hu):
    total_pixel=patient_pixels.flatten().tolist()
    #count the pixel of a certain tissue
    count = total_pixel.count(tissue_hu)
    #total number of pixel in the scan
    #slice number*pixel row number*pixel column number
    total=patient_pixels.shape[0]*patient_pixels.shape[1]*patient_pixels.shape[2]
    #Normalise
    count_normal=(count/total)*100000
    count_normal=round(count_normal, 4)
    return count,count_normal

    

def get_disease_tissue(patient_pixels):
    total_pixel=patient_pixels.flatten().tolist()
    #count the pixel of a disease tissue
    #disease tisssue has a hu range from 50 to 70
    count=0
    for i in list(total_pixel):
        if i>50 and i<70:
            count=count+1
    
    #total number of pixel in the scan
    #slice number*pixel row number*pixel column number
    total=patient_pixels.shape[0]*patient_pixels.shape[1]*patient_pixels.shape[2]
    #Normalise
    count_normal=(count/total)*1000
    count_normal=round(count_normal, 4)
    return count,count_normal

    
