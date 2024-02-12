import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def loadConsumptionAndPVforCustomer(pathTrain,pathEval,pathTest,customer):
    df_load_train = pd.read_csv(pathTrain, header=0, parse_dates=['date'], date_format="%d/%m/%y")
    df_load_eval = pd.read_csv(pathEval, header=0, parse_dates=['date'], date_format="%d/%m/%y")
    df_load_test = pd.read_csv(pathTest, header=0, parse_dates=['date'], date_format="%d/%m/%y")

    df = pd.concat([df_load_train, df_load_eval, df_load_test])
    return df[df['Customer']==customer].drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])

def combineControlledAndGeneralConsumption(df):
    if (df == 'CL').any().any():
        combinedLoad = pd.DataFrame(columns=df.columns)
        cl = df[df['Consumption Category'] == 'CL'].set_index(pd.RangeIndex(0, 1096))
        gc = df[df['Consumption Category'] == 'GC'].set_index(pd.RangeIndex(0, 1096))
        clgc = pd.concat([cl,gc])
        for i in range(0,1096):
            sum = clgc.loc[i].sum()
            sum['date'] = i
            combinedLoad.loc[i] = sum
        final_load = combinedLoad
    else:
        final_load = df[df['Consumption Category']=='GC']
   
    load_array = np.array(final_load.drop(columns=['Consumption Category', 'date'])).flatten()
    days = np.empty([52608,])
    start = 3
    for i in range(0,52608, 48):
        days[i:i+48] = np.repeat(start,48)
        start = (start+1) % 7
    return pd.DataFrame({0:days, 1:load_array})

def getPV(df):
    return np.array(df[df['Consumption Category']=='GG'].drop(columns=['Consumption Category', 'date']).set_index(pd.RangeIndex(0, 1096))).flatten()

def scale(arr,scaler):
    return scaler.fit_transform(arr.reshape(-1,1)).flatten()

def getPrice(pricePath):
    df = pd.read_csv(pricePath, header=0, parse_dates=['SETTLEMENTDATE'], date_format="%y/%d/%m")
    price = df.drop(columns=['REGION', 'SETTLEMENTDATE', 'TOTALDEMAND', 'PERIODTYPE']).div(1000)
    return np.array(price).flatten()

def getCustomerData(pathTrain,pathEval,pathTest,pathPrice,customer):
    data = loadConsumptionAndPVforCustomer(pathTrain,pathEval,pathTest,customer)
    load_pv = combineControlledAndGeneralConsumption(data)
    pv = getPV(data)
    load_pv.insert(2,2,pv)
    price = getPrice(pathPrice)

    train = (load_pv[0:17520],price)
    eval = (load_pv[17520:35088],price)
    test = (load_pv[35088:52608],price)
    
    return train, eval, test