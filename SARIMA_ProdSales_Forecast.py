# Import Libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from azureml.core import Workspace, Dataset, Datastore
import itertools
import multiprocessing
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')
# ============================================================================

class SARIMA_ProdSales_Forecast:
    # Load the datasets
    # Forecast Input
    # ----------------------------------------------------------------------------
    def import_original_dataset(subscription_id, resource_group, workspace_name):
        workspace = Workspace(subscription_id, resource_group, workspace_name)
        dataset = Dataset.get_by_name(workspace, name='DMDFRCST_FRCST_INPUT')
        df = dataset.to_pandas_dataframe()
        
        return df
    
    # Parameter list
    # ----------------------------------------------------------------------------
    def import_params(subscription_id, resource_group, workspace_name):
        workspace = Workspace(subscription_id, resource_group, workspace_name)
        dataset = Dataset.get_by_name(workspace, name='DMDFRCST_ARIMA_PARAM')
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
        df['dt'] =pd.to_datetime(df['SalesDate'], format = '%Y%m%d')
        df['qty_kg']=df['qty_kg'].astype('float')
        df['qty_kg'] = df['qty_kg'].fillna(0.0)

        return df
    # ============================================================================

    # Impute qty_kg values
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
    def compute_hyper_param(train, test):
        p = range(1,4)
        d = range(0,2)
        q = range(1,4)
        pdq = list(itertools.product(p,d,q))

        P = range(1,4)
        D = range(0,2)
        Q = range(1,4)
        M = range(0,13,3)
        seasonal_pdq = list(itertools.product(P,D,Q,M))

        # p = range(1,2)
        # d = range(0,1)
        # q = range(1,2)
        # pdq = list(itertools.product(p,d,q))

        # P = range(1,2)
        # D = range(0,1)
        # Q = range(1,2)
        # M = [12]
        # seasonal_pdq = list(itertools.product(P,D,Q,M))

        prev_error = +math.inf
        forecast = None    
        best_model = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(train['qty_kg'], order = param, seasonal_order = param_seasonal, enforce_stationarity=False, intercept=True)
                    model_fit = model.fit(max_iter = 50)

                    nsteps = 2 # Revert to 2
                    frcst = model_fit.forecast(nsteps)

                    cal_error = mean_absolute_error(list(test['qty_kg'])[:nsteps], list(frcst))

                    if cal_error < prev_error:
                        prev_error = cal_error
                        forecast = frcst
                        best_model = [param, param_seasonal]
                except Exception as e: 
                    continue

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

    def toCSV_param(output,filename):
        output_list = []
        for item in output:
            if item not in output_list:
                output_list.append(item)

        output_df = pd.DataFrame(output_list, columns = ['ProdSalesCode', 'p', 'd', 'q', 'P', 'D', 'Q', 'M'])
        output_df = output_df.sort_values(by=['ProdSalesCode'])
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

    # Find out the parameters
    # ----------------------------------------------------------------------------
    def find_param(prod_sales_dist, param_list):
        matched_param = None

        for x in param_list:
            if prod_sales_dist == x[0]:
                matched_param = x[1:]
                break
        
        return matched_param
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

    # Calculate total accuracy
    # ----------------------------------------------------------------------------
    def calculate_tot_acc(act, frcst, n):
        act_s = sum(act[:n])
        diff = []
        for i in range(n):
            diff.append(abs(act[i] - frcst[i]))
        diff_s = sum(diff)
        if act_s == 0:
            acc = 0
        else:
            acc = (1 -  (diff_s/act_s))*100

        return acc
    # ============================================================================

    # Create new dataset for parameters
    # ----------------------------------------------------------------------------
    def create_dataset_param(path, dataset_name):
        datastore_name = 'aiml_deltamart'
        workspace = Workspace.from_config()
        datastore = Datastore.get(workspace, datastore_name)
        datastore_paths = [(datastore, path)]
        dataset = Dataset.Tabular.from_delimited_files(path = datastore_paths, header = True)
        print(dataset.register(workspace, dataset_name, create_new_version=True))

    # ============================================================================

    # Processing 
    # ----------------------------------------------------------------------------
    def processing(df, prod_sales_dist, param_list, frcst_start, last_rec_dt, frcst_end, nsteps, lag_check, thresold):
        # Extract the needful dataframe from original dataframe
        # **********************************************************************
        temp_df = df[(df['ProdSalesCode'] == prod_sales_dist)]

        # **********************************************************************
        prod_num = temp_df['ProductNumber'].unique()
        proddesc = temp_df['ProductName'].unique()
        sales_dist_code = temp_df['SalesDistCode'].unique()
        sales_dist = temp_df['SalesDistName'].unique()
        cust = temp_df['cust_segment'].unique()
        
        imputed_df = SARIMA_ProdSales_Forecast.impute(temp_df, frcst_start, frcst_end)
        # **********************************************************************

        train, test = SARIMA_ProdSales_Forecast.train_test_splitting(imputed_df, last_rec_dt)
        # **********************************************************************
        # Find out the best param from the param list
        best_param = SARIMA_ProdSales_Forecast.find_param(prod_sales_dist, param_list)

        # the product_sales_dist is not in the param list
        if best_param == None:
            # Call Hyper-parameter tuning
            best_model = SARIMA_ProdSales_Forecast.compute_hyper_param(train, test)

            if best_model != None:
                model = SARIMAX(train['qty_kg'], order = best_model[0], seasonal_order = best_model[1], enforce_stationarity=False, intercept=True)
                model_fit = model.fit(max_iter = 500)

                frcst = model_fit.forecast(nsteps)
                forecast = list(frcst)
                actual_qty_kg_list = list(test['qty_kg'])
                actual_date_list = list(test['dt'])

                return [[prod_sales_dist, best_model[0][0], best_model[0][1], best_model[0][2], best_model[1][0], best_model[1][1], best_model[1][2], best_model[1][3]], [prod_sales_dist, prod_num[0], proddesc[0], sales_dist_code[0], sales_dist[0], cust[0], last_rec_dt, actual_date_list, actual_qty_kg_list, forecast]]
            
        # **************************************************************************
        # the product_sales_dist is in the param list
        else:
            best_p, best_d, best_q, best_P, best_D, best_Q, best_M = best_param[0], best_param[1], best_param[2], best_param[3], best_param[4], best_param[5], best_param[6]    

            # **********************************************************************

            model = SARIMAX(train['qty_kg'], order = (best_p, best_d, best_q), seasonal_order = (best_P, best_D, best_Q, best_M), enforce_stationarity=False, intercept=True)
            model_fit = model.fit(max_iter = 500)

            frcst = model_fit.forecast(nsteps)
            forecast = list(frcst)
            actual_qty_kg_list = list(test['qty_kg'])
            actual_date_list = list(test['dt'])

            # **********************************************************************
            # If avg accuracy of 'lag_check' months is less than thresold % then we will do a 
            # hyper-param tuning for the product_sales_dist, otherwise we will append 
            # it to the output dataframe
            # **********************************************************************
            if SARIMA_ProdSales_Forecast.calculate_tot_acc(actual_qty_kg_list, forecast, lag_check) < thresold:
                # Call Hyper parameter tuning
                best_model = SARIMA_ProdSales_Forecast.compute_hyper_param(train, test)

                if best_model != None:
                    model = SARIMAX(train['qty_kg'], order = best_model[0], seasonal_order = best_model[1], enforce_stationarity=False, intercept=True)
                    model_fit = model.fit(max_iter = 500)

                    frcst = model_fit.forecast(nsteps)
                    forecast = list(frcst)
                    actual_qty_kg_list = list(test['qty_kg'])
                    actual_date_list = list(test['dt'])

                    return [[prod_sales_dist, best_model[0][0], best_model[0][1], best_model[0][2], best_model[1][0], best_model[1][1], best_model[1][2], best_model[1][3]], [prod_sales_dist, prod_num[0], proddesc[0], sales_dist_code[0], sales_dist[0], cust[0], last_rec_dt, actual_date_list, actual_qty_kg_list, forecast]]
                
            else:
                return [[prod_sales_dist, best_p, best_d, best_q, best_P, best_D, best_Q, best_M], [prod_sales_dist, prod_num[0], proddesc[0], sales_dist_code[0], sales_dist[0], cust[0], last_rec_dt, actual_date_list, actual_qty_kg_list, forecast]]

    # ============================================================================

    # Final processing function
    # ----------------------------------------------------------------------------
    def final_processing():
        df = SARIMA_ProdSales_Forecast.import_original_dataset(subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6',
                                                                resource_group = 'AI-QAT1-RG', 
                                                                workspace_name = 'mlworkspaceingredion')
        df = SARIMA_ProdSales_Forecast.rename_col(df)

        df = SARIMA_ProdSales_Forecast.preprocess(df)

        # Groupping
        df = df[(df["cust_segment"].isin(['GROUP_2', 'GROUP_3']))]

        product_sales_dist_list = df['ProdSalesCode'].unique()

        # *************************************************************************

        param = SARIMA_ProdSales_Forecast.import_params(subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6',
                                                        resource_group = 'AI-QAT1-RG', 
                                                        workspace_name = 'mlworkspaceingredion')

        param_list = param.values.tolist()   

        output = []
        results = []
        new_param_list = []

        # **********************************************************************
        nsteps = 26
        lag_check =  2
        thresold = 70
        
        # *************************************************************************
        # Forecast Start Date
        frcst_start = datetime(2015,1,1).date()

        # Last Record Date
        first_date_of_current_month = datetime.today().replace(day = 1)
        last_rec_dt = first_date_of_current_month + relativedelta(months = -4)
        last_rec_dt = last_rec_dt.date()

        # Forecast End Date
        frcst_end = last_rec_dt + relativedelta(months = nsteps)
        # **********************************************************************

        with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
            for prod_sales_dist in product_sales_dist_list:
                results.append(executor.submit(SARIMA_ProdSales_Forecast.processing, df, prod_sales_dist, param_list, frcst_start, last_rec_dt, frcst_end, nsteps, lag_check, thresold))
            
            for f in concurrent.futures.as_completed(results):  
                try:
                    res = f.result()

                    if res != None:
                        new_param_list.append(res[0])
                        temp_ret = res[1]

                        for i in range(len(temp_ret[-1])):
                            if temp_ret[-1][i] < 0:
                                temp_ret[-1][i] = 0.0
                            acc = SARIMA_ProdSales_Forecast.lag_acc(temp_ret[8][i], temp_ret[-1][i])

                            output.append([temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3], temp_ret[4], temp_ret[5], temp_ret[6], temp_ret[7][i].year, temp_ret[7][i].month, temp_ret[7][i], i+1, temp_ret[8][i], temp_ret[-1][i], acc, 'SARIMA'])

                except:
                    continue

        return output, new_param_list
    
    def main():
        start = time.time()
        output, param_list = SARIMA_ProdSales_Forecast.final_processing()
        end = time.time()
        print('Training Completed!')
        # **********************************************************************
        # Forecast Output
        forecast_file_name = '02_0_IS50_SARIMA_ProdSales.csv'
        SARIMA_ProdSales_Forecast.toCSV(output, forecast_file_name)
        SARIMA_ProdSales_Forecast.to_container(forecast_file_name)
        print('Forecast for ProdSalesCode has been sent to Container!')
        # **********************************************************************
        # SARIMA parameters
        param_filename = '02_1_IS50_SARIMA_ProdSales_PARAM.csv'
        SARIMA_ProdSales_Forecast.toCSV_param(param_list, param_filename)
        SARIMA_ProdSales_Forecast.to_container(param_filename)
        print('Parameter combinations for ProdSalesCode has been sent to Container!')

        # Create new dataset for parameters
        path = 'mlmodel/MLDemandForecast/final_fcst_output/' + param_filename
        dataset_name = 'DMDFRCST_ARIMA_PARAM'
        SARIMA_ProdSales_Forecast.create_dataset_param(path, dataset_name)
        print('New dataset has been created with new parameter list!')
        # **********************************************************************
        print('Time taken = ', end - start)
        print('Done!')