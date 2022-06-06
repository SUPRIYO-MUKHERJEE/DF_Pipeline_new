# Import Libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from azureml.core import Workspace, Dataset, Datastore
import itertools
import multiprocessing
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')
# ============================================================================

class HW_ProdSales_Forecast:
    # Load the dataset
    # ----------------------------------------------------------------------------
    def import_original_dataset(subscription_id, resource_group, workspace_name):
        workspace = Workspace(subscription_id, resource_group, workspace_name)
        dataset = Dataset.get_by_name(workspace, name='DMDFRCST_FRCST_INPUT')
        df = dataset.to_pandas_dataframe()
        
        return df

    # ============================================================================

    # Rename the columns
    # ----------------------------------------------------------------------------
    def rename_col(df):
        df.rename(columns={'Column1':'ProductNumber'}, inplace=True)
        df.rename(columns={'Column2':'ProductName'}, inplace=True)
        df.rename(columns={'Column3':'SalesDistCode'}, inplace=True)
        df.rename(columns={'Column4':'SalesDistName'}, inplace=True)
        df.rename(columns={'Column5':'ProdSalesCode'}, inplace=True)
        df.rename(columns={'Column6':'cust_segment'}, inplace=True)
        df.rename(columns={'Column7':'yr'}, inplace=True)
        df.rename(columns={'Column8':'mo'}, inplace=True)
        df.rename(columns={'Column9':'SalesDate'}, inplace=True)
        df.rename(columns={'Column10':'qty_kg'}, inplace=True)
        
        return df
    # ============================================================================

    # Preprocessing
    # ----------------------------------------------------------------------------
    def preprocess(df):
        df['dt'] = pd.to_datetime(df['SalesDate'], format = '%Y%m%d')
        df['qty_kg'] = df['qty_kg'].astype('float')
        df['qty_kg'] = df['qty_kg'].fillna(1.0)

        return df
    # ============================================================================

    # Impute values
    # ----------------------------------------------------------------------------
    def impute(temp_df, start_date, end_date):
        zerodf = pd.DataFrame()
        zerodf['dt'] = pd.date_range(start = start_date, end = end_date, freq= 'MS')

        zero_merged = pd.merge(temp_df, zerodf, how = 'right', on = 'dt')
        zero_merged['qty_kg'] = zero_merged['qty_kg'].fillna(0.0)
        zero_merged = zero_merged.sort_values(by = ['dt'])
        
        return zero_merged
    # ============================================================================

    # Train - Test splitting
    # ----------------------------------------------------------------------------
    def train_test_splitting(df, split_date):
        df_train = df[['dt', 'qty_kg']].loc[df['dt'] <= split_date]
        df_test = df[['dt', 'qty_kg']].loc[df['dt'] > split_date]
        
        return df_train, df_test
    # ============================================================================

    # Compute Hyper-parameters
    # ----------------------------------------------------------------------------
    def HW_model(train, test, lag_check):
        ret_model = []

        for t in ['add', 'mul', None]:
            for s in ['add', 'mul', None]:
                for sp in [0, 3, 6, 9 ,12]:
                    try:
                        model = ExponentialSmoothing(train['qty_kg'], trend=t, seasonal = s, seasonal_periods=sp)
                        model_fit = model.fit()
                        forecast_val = model_fit.forecast(lag_check)

                        mae = mean_absolute_error(list(test['qty_kg'])[:lag_check], list(forecast_val))
                        ret_model.append([model_fit, mae, t, s, sp])
                    except:
                        ret_model.append([None, +math.inf, t, s, sp])
        
        best_model = [None, +math.inf]
        for model in ret_model:
            if model[1] < best_model[1]:
                best_model = model
            
        return best_model
    # ============================================================================

    # Make CSV and store it to workspace
    # ----------------------------------------------------------------------------
    def toCSV(output,filename):
        output_list = []
        for item in output:
            if item not in output_list:
                output_list.append(item)

        output_df = pd.DataFrame(output_list, columns = ['ProdSalesCode', 'ProductNumber', 'Productdescr', 'SalesDistCode' , 
                                                        'SalesDistrict', 'cust_segment', 'Last record date', 'yr', 'month', 
                                                        'forecast_date', 'lag', 'qty_kg', 'frcst_qty', 'acc', 'model'])
        output_df = output_df.sort_values(by=['ProdSalesCode', 'lag'])
        output_df.to_csv('../Datasets/'+filename, header=True, index=False)
    # ============================================================================

    # Make CSV and store it to container
    # ----------------------------------------------------------------------------
    def to_container(filename):
        subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6'
        resource_group = 'AI-QAT1-RG'
        workspace_name = 'mlworkspaceingredion'

        workspace = Workspace(subscription_id, resource_group, workspace_name)
        datastore = Datastore(workspace, 'aiml_deltamart')
        datastore.upload_files(['../Datasets/'+filename], target_path = 'mlmodel/MLDemandForecast/final_fcst_output', overwrite = True)
    # ============================================================================

    # Calculate per lag accuracy
    # ----------------------------------------------------------------------------
    def lag_acc(act, frcst):
        if act == 0:
            acc = 0
        else:
            acc = (1 - abs(act - frcst)/act)*100
        
        return acc
    # ============================================================================

    # Processing 
    # ----------------------------------------------------------------------------
    def processing(df, prod_sales_dist, frcst_start, last_rec_dt, frcst_end, nsteps, lag_check):
        # Extract the needful dataframe from original dataframe
        # **********************************************************************
        temp_df = df[(df['ProdSalesCode'] == prod_sales_dist)]

        # **********************************************************************
        prod_num = temp_df['ProductNumber'].unique()
        proddesc = temp_df['ProductName'].unique()
        sales_dist_code = temp_df['SalesDistCode'].unique()
        sales_dist = temp_df['SalesDistName'].unique()
        cust = temp_df['cust_segment'].unique()
        
        imputed_df = HW_ProdSales_Forecast.impute(temp_df, frcst_start, frcst_end)
        # **********************************************************************

        train, test = HW_ProdSales_Forecast.train_test_splitting(imputed_df, last_rec_dt)
        # **********************************************************************
        
        best_model = HW_ProdSales_Forecast.HW_model(train, test, lag_check)

        if best_model[0] != None:
            frcst = best_model[0].forecast(nsteps)
            forecast = list(frcst)

            actual_qty_kg_list = list(test['qty_kg'])
            actual_date_list = list(test['dt'])

            return [[prod_sales_dist, prod_num[0], proddesc[0], sales_dist_code[0], sales_dist[0], cust[0], last_rec_dt, actual_date_list, actual_qty_kg_list, forecast]]
    # ============================================================================

    # Final Processing function
    # ----------------------------------------------------------------------------
    def final_processing():
        df = HW_ProdSales_Forecast.import_original_dataset(subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6',
                                    resource_group = 'AI-QAT1-RG', 
                                    workspace_name = 'mlworkspaceingredion')
        
        df = HW_ProdSales_Forecast.rename_col(df)

        df = HW_ProdSales_Forecast.preprocess(df)

        # Groupping
        df = df[(df["cust_segment"].isin(['GROUP_2', 'GROUP_3']))]

        product_sales_dist_list = df['ProdSalesCode'].unique()

        # *************************************************************************
        output = []
        results = []

        # **********************************************************************
        nsteps = 26
        lag_check =  1
        
        # *************************************************************************
        # Forecast Start Date
        frcst_start = datetime(2015,1,1).date()

        # Last Record Date
        first_date_of_current_month = datetime.today().replace(day = 1)
        last_rec_dt = first_date_of_current_month + relativedelta(months = -2)
        last_rec_dt = last_rec_dt.date()

        # Forecast End Date
        frcst_end = last_rec_dt + relativedelta(months = nsteps)
        # **********************************************************************

        with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
            for prod_sales_dist in product_sales_dist_list:
                print(prod_sales_dist)
                results.append(executor.submit(HW_ProdSales_Forecast.processing, df, prod_sales_dist, frcst_start, last_rec_dt, frcst_end, nsteps, lag_check))
            
            for f in concurrent.futures.as_completed(results):  
                try:
                    res = f.result()

                    if res != None:
                        temp_ret = res[0]

                        for i in range(len(temp_ret[-1])):
                            if temp_ret[-1][i] < 0:
                                temp_ret[-1][i] = 0.0
                            acc = HW_ProdSales_Forecast.lag_acc(temp_ret[8][i], temp_ret[-1][i])

                            output.append([temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3], temp_ret[4], temp_ret[5], temp_ret[6], temp_ret[7][i].year, temp_ret[7][i].month, temp_ret[7][i], i+1, temp_ret[8][i], temp_ret[-1][i], acc, 'HW'])

                except:
                    continue

        return output
    
    def main():
        start = time.time()
        output = HW_ProdSales_Forecast.final_processing()
        end = time.time()
        print('Training Completed!')
        # **********************************************************************
        # Forecast Output
        forecast_file_name = '00_IS50_HW_ProdSales.csv'
        HW_ProdSales_Forecast.toCSV(output, forecast_file_name)
        HW_ProdSales_Forecast.to_container(forecast_file_name)
        print('Forecast for ProdSalesCode has been sent to Container!')
        # **********************************************************************
        print('Time taken = ', end - start)
        print('Done!')