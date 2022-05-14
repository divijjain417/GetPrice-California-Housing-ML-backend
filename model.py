import numpy as np # linear algebra
import pandas as pd # data processin/microsoft/pylance-release/blob/main/DIAGNOSTIC_SEVERITY_RULES.mdg, CSV file I/O (e.g. pd.read_csv)

import pickle 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:34.787811Z","iopub.execute_input":"2022-05-07T02:05:34.788486Z","iopub.status.idle":"2022-05-07T02:05:36.040999Z","shell.execute_reply.started":"2022-05-07T02:05:34.788448Z","shell.execute_reply":"2022-05-07T02:05:36.039829Z"}}
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:38.289868Z","iopub.execute_input":"2022-05-07T02:05:38.290180Z","iopub.status.idle":"2022-05-07T02:05:38.478450Z","shell.execute_reply.started":"2022-05-07T02:05:38.290149Z","shell.execute_reply":"2022-05-07T02:05:38.477788Z"}}
import seaborn as sns
import matplotlib.pyplot as plt 

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:41.775853Z","iopub.execute_input":"2022-05-07T02:05:41.776703Z","iopub.status.idle":"2022-05-07T02:05:41.863283Z","shell.execute_reply.started":"2022-05-07T02:05:41.776666Z","shell.execute_reply":"2022-05-07T02:05:41.861591Z"}}
data = pd.read_csv("housing.csv") #reading in the csv_data on housing
data.head()


# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:44.245326Z","iopub.execute_input":"2022-05-07T02:05:44.246244Z","iopub.status.idle":"2022-05-07T02:05:44.310414Z","shell.execute_reply.started":"2022-05-07T02:05:44.246177Z","shell.execute_reply":"2022-05-07T02:05:44.309486Z"}}
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:07:35.741732Z","iopub.execute_input":"2022-05-07T02:07:35.742529Z","iopub.status.idle":"2022-05-07T02:07:35.749660Z","shell.execute_reply.started":"2022-05-07T02:07:35.742480Z","shell.execute_reply":"2022-05-07T02:07:35.748785Z"}}
data['total_bedrooms']=data['total_bedrooms'].fillna('bfill') #fill in the empty columns 


# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:51.621866Z","iopub.execute_input":"2022-05-07T02:05:51.622195Z","iopub.status.idle":"2022-05-07T02:05:51.627747Z","shell.execute_reply.started":"2022-05-07T02:05:51.622160Z","shell.execute_reply":"2022-05-07T02:05:51.626974Z"}}
print(np.shape(data)) #describes the number of rows and columns

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:05:54.468409Z","iopub.execute_input":"2022-05-07T02:05:54.469161Z","iopub.status.idle":"2022-05-07T02:07:19.371926Z","shell.execute_reply.started":"2022-05-07T02:05:54.469120Z","shell.execute_reply":"2022-05-07T02:07:19.371108Z"}}
#sns.lineplot(data=data, x="median_house_value", y="households")
#relationship between household desnity and median_house_value

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:07:46.828088Z","iopub.execute_input":"2022-05-07T02:07:46.828826Z","iopub.status.idle":"2022-05-07T02:09:11.684246Z","shell.execute_reply.started":"2022-05-07T02:07:46.828779Z","shell.execute_reply":"2022-05-07T02:09:11.683360Z"}}
#sns.lineplot(data=data, x="median_house_value", y="total_rooms")

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:11:38.943656Z","iopub.execute_input":"2022-05-07T02:11:38.943973Z","iopub.status.idle":"2022-05-07T02:11:39.144829Z","shell.execute_reply.started":"2022-05-07T02:11:38.943924Z","shell.execute_reply":"2022-05-07T02:11:39.143733Z"}}
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:11:44.009553Z","iopub.execute_input":"2022-05-07T02:11:44.009854Z","iopub.status.idle":"2022-05-07T02:13:09.189798Z","shell.execute_reply.started":"2022-05-07T02:11:44.009819Z","shell.execute_reply":"2022-05-07T02:13:09.188902Z"}}
#sns.lineplot(data=data, x="median_house_value", y="median_income")
# line plot to show the relation between median_house_value and the median_income

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:16:52.159919Z","iopub.execute_input":"2022-05-07T02:16:52.160352Z","iopub.status.idle":"2022-05-07T02:16:52.538700Z","shell.execute_reply.started":"2022-05-07T02:16:52.160321Z","shell.execute_reply":"2022-05-07T02:16:52.537712Z"}}

#sns.displot(data=data,x="median_house_value", kind='kde')

# plotting a desnity graph to show median_house_value 

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:16:41.844668Z","iopub.execute_input":"2022-05-07T02:16:41.845019Z","iopub.status.idle":"2022-05-07T02:16:41.852802Z","shell.execute_reply.started":"2022-05-07T02:16:41.844981Z","shell.execute_reply":"2022-05-07T02:16:41.852144Z"}}
features = ['longitude', 'latitude', 'total_rooms', 'median_income']
X = data[features]
y = data['median_house_value']





# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:17:39.403847Z","iopub.execute_input":"2022-05-07T02:17:39.404872Z","iopub.status.idle":"2022-05-07T02:17:39.413787Z","shell.execute_reply.started":"2022-05-07T02:17:39.404812Z","shell.execute_reply":"2022-05-07T02:17:39.413022Z"}}
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.6,random_state=0)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:17:41.395995Z","iopub.execute_input":"2022-05-07T02:17:41.396277Z","iopub.status.idle":"2022-05-07T02:17:44.180230Z","shell.execute_reply.started":"2022-05-07T02:17:41.396246Z","shell.execute_reply":"2022-05-07T02:17:44.179117Z"}}
model = RandomForestRegressor(random_state=0)
model.fit(X_train,y_train)

model2 = RandomForestRegressor(random_state=0)

print(X_train.head())
# %% [code] {"execution":{"iopub.status.busy":"2022-05-06T23:03:44.759703Z","iopub.execute_input":"2022-05-06T23:03:44.759967Z","iopub.status.idle":"2022-05-06T23:03:45.000523Z","shell.execute_reply.started":"2022-05-06T23:03:44.759932Z","shell.execute_reply":"2022-05-06T23:03:44.999323Z"}}
model.predict(X_test)

pickle.dump(model,open('model.pkl','wb'))

test_case = pd.DataFrame(data={'longitude':-118.35, 'latitude': 34.22, '1':1560.0, 'median_income':4.0000}, index=[0])

# %% [code] {"execution":{"iopub.status.busy":"2022-05-07T02:17:48.929420Z","iopub.execute_input":"2022-05-07T02:17:48.929727Z","iopub.status.idle":"2022-05-07T02:17:49.224594Z","shell.execute_reply.started":"2022-05-07T02:17:48.929696Z","shell.execute_reply":"2022-05-07T02:17:49.223715Z"}}
print(mean_absolute_error(model.predict(X_test), y_test))
print(model.predict(test_case))
print(model.predict(X_test))
