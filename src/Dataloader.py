import pandas as pd


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


def get_customer_data(dfload, dfprice, customer=1):
    """
    Prepare the customer and price data

    :param dfload: dataframe with load and pv
    :param dfprice: dataframe with price
    :param customer: customer ID
    :return: dataframes for the given customer
    """
    customer_all_data = dfload[(dfload['Customer'] == customer)]
    customer_reduced_data = customer_all_data.drop(columns=['Customer', 'Generator Capacity', 'Postcode'])
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
    return customer_load_data, customer_pv_data, price_data
