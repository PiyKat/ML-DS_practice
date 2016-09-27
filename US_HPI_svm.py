import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt   # Standard name for matplotlib
from matplotlib.pyplot import style
import numpy as np
from statistics import mean
from sklearn import svm,preprocessing,cross_validation

style.use('fivethirtyeight')

key = open("quandl_api_key.txt").read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]

def grab_initial_state_data():
    states = state_list()
    main_df = pd.DataFrame()
    for abbv in states:
        
        query = "FMAC/HPI_" + str(abbv) 
        df = quandl.get(query,authtoken = key)
        df.rename(columns = {"Value":abbv},inplace=True)
        df[abbv] = (df[abbv] - df[abbv][0])/df[abbv][0]*100
        
        if main_df.empty:       # empty used to check whether pf is empty
          main_df = df
        else:
          main_df = main_df.join(df)
    
    # Pickling to store data
    return main_df
    pickle_in = open('pickle1_pct_change2.pickle','wb')
    pickle.dump(main_df,pickle_in)
    pickle.close()
    

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA",authtoken = key)
    df.rename(columns = {"Value":"United States"},inplace = True)
    df["United States"] = (df["United States"] - df["United States"][0])/df["United States"][0]*100
    return df

def mortgage_30y():
    df = quandl.get("FMAC/MORTG",authtoken = key,trim_start = "1975-01-01")
    df.rename(columns = {"Value":"M30"},inplace = True)
    df["M30"] = (df["M30"] - df["M30"][0])/df["M30"][0]*100
    #df = df.resample('D')
    df = df.resample('M').mean() # Resample data to month so that the index is the last day of month 
    return df

def sp500():
    df = quandl.get("YAHOO/INDEX_GSPC",authtoken = key,trim_start = "1975-01-01")
    df.resample('M').mean()
    df.rename(columns = {"Adjusted Close":"s&p500"},inplace=True)
    df["s&p500"] = (df["s&p500"] - df["s&p500"][0])/df["s&p500"][0]*100
    df = df['s&p500']
    return df 

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=key)
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    #df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df

def create_labels(curr_hpi,fut_hpi):
    if fut_hpi > curr_hpi:
        return 1
    else:
        return 0

def moving_averages(values):
    return mean(values)

unemployment_data = us_unemployment()

us_benchmark = HPI_Benchmark()

gdp = gdp_data()

HPI_data = grab_initial_state_data()

M30 = mortgage_30y()

stock_data = sp500()

HPI = HPI_data.join([M30,stock_data,gdp,unemployment_data,us_benchmark])

print("Final Consolidated Data: ")
print(HPI.head())

HPI.to_pickle('HPI.pickle')

housing_data = pd.read_pickle('HPI.pickle')

housing_data = housing_data.pct_change()

housing_data.replace([np.inf, -np.inf],np.nan,inplace = True) # Use 
housing_data.dropna(inplace = True)

housing_data['US_HPI_future'] = housing_data['United States'].shift(-1) # -ve means shifting column
housing_data.dropna(inplace = True)

housing_data['label'] = list(map(create_labels,housing_data['United States'],housing_data['US_HPI_future']))

# Note - by default both for numpy and pandas, axis = 0 means along the rows and axis = 1 means along the column

X = np.array(housing_data.drop(['US_HPI_future','label'],1)) # Note , 1 denotes delete the column
X = preprocessing.scale(X)

y = np.array(housing_data['label'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf  = svm.SVC(kernel='linear')

clf.fit(X_train,y_train) 
print("Accuracy:")
print(clf.score(X_test,y_test))
print("Coefficients(using SVM):")
print(clf.coef_)
