import pandas as pd
import numpy as np


def loadData(data_path):
    """
    Load the load and pv data from the csv

    :param data_path: path to csv
    :return: dataframe with load and pv
    """
    dfload = pd.read_csv(data_path, header=0, parse_dates=['date'], date_format="%d/%m/%y")
    return dfload


def loadPrice(data_path):
    """
    Load the price from the csv

    :param data_path: path to csv
    :return: dataframe with price
    """
    dfprice = pd.read_csv(data_path, header=0, parse_dates=['SETTLEMENTDATE'], date_format="%y/%d/%m")
    return dfprice

def loadMix(data_path):
    dfmix = pd.read_csv(data_path, header=0)
    return dfmix

def process_fuelmix(dfmix):
    dfmix_sum = pd.DataFrame(columns=dfmix.columns)
    for i in range(0,35039,2):
        sum = dfmix.iloc[i:i+2].sum()
        dfmix_sum.loc[i] = sum
    dfmix_processed = dfmix_sum.drop(columns=['Date', 'Start', 'End'])
    return dfmix_processed.set_index(pd.RangeIndex(0, 17520))

def get_customer_data(dfload, dfprice, dfmix, customer=1):
    """
    Prepare the customer and price data

    :param dfload: dataframe with load and pv
    :param dfprice: dataframe with price
    :param customer: customer ID
    :return: dataframes for the given customer
    """
    customer_all_data = dfload[(dfload['Customer'] == customer)]
    customer_reduced_data = customer_all_data.drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])
    gc = customer_reduced_data[customer_reduced_data['Consumption Category'] == 'GC'].set_index(pd.RangeIndex(0, 365)) \
        .drop(columns=['Consumption Category', 'date'])
    if len(customer_reduced_data[customer_reduced_data['Consumption Category'] == 'CL']) != 0:
        cl = customer_reduced_data[customer_reduced_data['Consumption Category'] == 'CL'].set_index(
            pd.RangeIndex(0, 365)).drop(columns=['Consumption Category', 'date'])
    values = pd.DataFrame()
    customer_load_data = pd.DataFrame(columns=gc.columns)
    for day in range(0, 365):
        values.iloc[0:0]
        values['gc'] = gc.iloc[day]
        if len(customer_reduced_data[customer_reduced_data['Consumption Category'] == 'CL']) != 0:
            values['cl'] = cl.iloc[day]
        else:
            values['cl'] = 0.0
        summed_values = values.sum(axis=1)
        customer_load_data.loc[day] = summed_values.T
    customer_pv_data = customer_reduced_data[customer_reduced_data['Consumption Category'] == 'GG'].set_index(
        pd.RangeIndex(0, 365)) \
        .drop(columns=['Consumption Category', 'date'])
    price_data = dfprice.drop(columns=['REGION', 'SETTLEMENTDATE', 'TOTALDEMAND', 'PERIODTYPE']).div(1000)
    load_array = pd.DataFrame(np.array(customer_load_data).flatten())
    pv_array = pd.DataFrame(np.array(customer_pv_data).flatten())
    price_array = pd.DataFrame(np.array(price_data).flatten())
    return load_array, pv_array, price_array#, process_fuelmix(dfmix)


