# Import Libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from azureml.core import Workspace, Dataset, Datastore
import warnings
warnings.filterwarnings('ignore')
# ============================================================================

class XGB_ProdSales_Forecast:
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
        df['dt'] =pd.to_datetime(df['SalesDate'], format = '%Y%m%d')
        df['qty_kg']=df['qty_kg'].astype('float')
        df['qty_kg'] = df['qty_kg'].fillna(0.0)

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

    # Transform a time series dataset into a Supervised Learning dataset
    # ----------------------------------------------------------------------------
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):    
        df_temp = pd.DataFrame(data)
        cols = list()    
        
        for i in range(n_in, 0, -1):
            cols.append(df_temp.shift(i))
        
        for i in range(0, n_out):
            cols.append(df_temp.shift(-i))

        agg = pd.concat(cols, axis=1)
        
        if dropnan:
            agg.dropna(inplace=True)
        
        return agg.values
    # ============================================================================

    # XGBoost Model train
    # ----------------------------------------------------------------------------
    def xgboost_train(train):
        train = np.asarray(train)    
        trainX, trainy = train[:, :-1], train[:, -1]

        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(trainX, trainy)

        return model
    # ============================================================================

    # XGBoost Model forecast
    # ----------------------------------------------------------------------------
    def xgboost_frcst(model, testX):
        testX = np.array(testX).reshape(1,-1)
        yhat = model.predict(testX)

        return yhat
    # ============================================================================

    # Forecasting using XGB
    # ----------------------------------------------------------------------------
    def make_predictions(series, pred_steps, n_in, n_out = 1):
        data = XGB_ProdSales_Forecast.series_to_supervised(series, n_in = n_in, n_out = n_out)
        
        model = XGB_ProdSales_Forecast.xgboost_train(data)
        frcst = []
        pred_input = data[-1][1:]
        
        for _ in range(pred_steps+1):
            y_hat = XGB_ProdSales_Forecast.xgboost_frcst(model, pred_input)
            pred_input = list(pred_input)
            pred_input.append(y_hat)
            pred_input = pred_input[1:]
            frcst.append(y_hat[0])
        
        return frcst
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
        print(prod_sales_dist)
        # Extract the needful dataframe from original dataframe
        # **********************************************************************
        temp_df = df[(df['ProdSalesCode'] == prod_sales_dist)]

        # **********************************************************************
        prod_num = temp_df['ProductNumber'].unique()
        proddesc = temp_df['ProductName'].unique()
        sales_dist_code = temp_df['SalesDistCode'].unique()
        sales_dist = temp_df['SalesDistName'].unique()
        cust = temp_df['cust_segment'].unique()

        
        imputed_df = XGB_ProdSales_Forecast.impute(temp_df, frcst_start, frcst_end)
        imputed_df = imputed_df[['dt','qty_kg']]
        # **********************************************************************

        train, test = XGB_ProdSales_Forecast.train_test_splitting(imputed_df, last_rec_dt)
        # **********************************************************************

        n_features = [1, 3, 6, 9, 12]
        prev_acc = +math.inf
        best_forecast = None

        actual_date_list = list(test['dt'])
        actual_qty_kg_list = list(test['qty_kg'])

        for n in n_features:
            forecast = XGB_ProdSales_Forecast.make_predictions(train['qty_kg'], pred_steps = nsteps, n_in = n, n_out = 1)

            if mean_absolute_error(actual_qty_kg_list[:lag_check], forecast[1:lag_check+1]) < prev_acc:
                best_forecast = forecast

        return [prod_sales_dist, prod_num[0], proddesc[0], sales_dist_code[0], sales_dist[0], cust[0], last_rec_dt, actual_date_list, actual_qty_kg_list, best_forecast[1:]]

    # ============================================================================

    # Final processing function
    # ----------------------------------------------------------------------------
    def final_processing():
        df = XGB_ProdSales_Forecast.import_original_dataset(subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6',
                                    resource_group = 'AI-QAT1-RG', 
                                    workspace_name = 'mlworkspaceingredion')
        
        df = XGB_ProdSales_Forecast.rename_col(df)

        df = XGB_ProdSales_Forecast.preprocess(df)

        # Groupping
        df = df[(df["cust_segment"].isin(['GROUP_2', 'GROUP_3']))]

        product_sales_dist_list = df['ProdSalesCode'].unique()

        # ************************************************************************

        output = []
        results = []
        # **********************************************************************
        
        nsteps = 26 
        lag_check = 1 
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

        for prod_sales_dist in product_sales_dist_list:
            temp_ret = XGB_ProdSales_Forecast.processing(df, prod_sales_dist, frcst_start, last_rec_dt, frcst_end, nsteps, lag_check)

            for i in range(len(temp_ret[-1])):
                if temp_ret[-1][i] < 0:
                    temp_ret[-1][i] = 0.0
                acc = XGB_ProdSales_Forecast.lag_acc(temp_ret[8][i], temp_ret[-1][i])

                output.append([temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3], temp_ret[4], temp_ret[5], temp_ret[6], temp_ret[7][i].year, temp_ret[7][i].month, temp_ret[7][i], i+1, temp_ret[8][i], temp_ret[-1][i], acc, 'XGB'])

        return output
    
    def main():
        start = time.time()
        output = XGB_ProdSales_Forecast.final_processing()
        end = time.time()
        print('Training Completed!')
        # **********************************************************************
        # Forecast Output
        forecast_file_name = '03_IS50_XGB_ProdSales.csv'
        XGB_ProdSales_Forecast.toCSV(output, forecast_file_name)
        XGB_ProdSales_Forecast.to_container(forecast_file_name)
        print('Forecast for ProdSalesCode has been sent to Container!')
        # **********************************************************************
        print('Time taken = ', end - start)
        print('Done!')