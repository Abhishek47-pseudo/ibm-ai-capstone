import os
import re
import shutil
import time

import pandas as pd
import numpy as np

from .processing import convert_to_ts


def fetch_data(data_dir):
    """
    load all json formatted files into a dataframe
    
    based on original from: https://github.com/aavail/ai-workflow-capstone
    """

    ## input testing
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search("\.json",f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    ## read data into a temp structure
    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    ## ensure the data are formatted with correct columns
    for f,df in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
             df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price':'price'},inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("columns name could not be matched to correct cols")

    ## concat all of the data
    df = pd.concat(list(all_months.values()),sort=True)
    years,months,days = df['year'].values,df['month'].values,df['day'].values 
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    df['invoice'] = [re.sub("\D+","",i) for i in df['invoice'].values]
    
    ## sort by date and reset the index
    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    return(df)



def fetch_ts(data_dir, clean=False):
    """
    convenience function to read in new data
    uses csv to load quickly
    use clean=True when you want to re-create the files

    based on original from: https://github.com/aavail/ai-workflow-capstone
    """

    ts_data_dir = os.path.join(data_dir,"ts-data")
    
    if clean:
        shutil.rmtree(ts_data_dir)
    if not os.path.exists(ts_data_dir):
        os.mkdir(ts_data_dir)

    ## if files have already been processed load them        
    if len(os.listdir(ts_data_dir)) > 0:
        print("... loading ts data from files")
        result = {re.sub("\.csv","",cf)[3:]:pd.read_csv(os.path.join(ts_data_dir,cf)) for cf in os.listdir(ts_data_dir)}
        for country, df in result.items():
            result[country].date = pd.to_datetime(result[country].date)
        return(result)

    ## get original data
    print("... processing data for loading")
    df = fetch_data(data_dir)

    ## find the top ten countries (wrt revenue)
    table = pd.pivot_table(df,index='country',values="price",aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue',inplace=True,ascending=False)
    top_ten_countries =  np.array(list(table.index))[:10]

    ## load the data
    dfs = {}
    dfs['all'] = convert_to_ts(df)
    for country in top_ten_countries:
        country_id = re.sub("\s+","_",country.lower())
        dfs[country_id] = convert_to_ts(df,country=country)

    ## save the data as csvs    
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir,"ts-"+key+".csv"),index=False)
        
    return(dfs)

def get_country_names(data_dir=None):
    if not data_dir:
        data_dir = os.path.join("data", "cs-train")

    df = fetch_data(data_dir)

    m = {}

    for country in df['country'].unique().tolist():
        country_id = re.sub("\s+","_",country.lower())
        m[country_id] = country

    return m

if __name__ == "__main__":

    run_start = time.time() 
    print(os.listdir('..'))
    data_dir = os.path.join("data","cs-train")
    print("...fetching data")

    ts_all = fetch_ts(data_dir,clean=False)

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

    for key,item in ts_all.items():
        print(key,item.shape)