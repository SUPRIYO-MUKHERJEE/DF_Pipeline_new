# Import Libraries
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from azureml.core import Workspace, Datastore, Dataset
from Preprocess import Preprocess
import warnings
warnings.filterwarnings("ignore")
# ============================================================================

class Best_Model_Selection:
    # Preprocessing
    # ----------------------------------------------------------------------------
    def Preprocess(ip):
        ip=ip[['ProdSalesCode','ProductNumber','Productdescr','SalesDistCode','SalesDistrict','cust_segment','Last record date','yr','month','forecast_date','lag','qty_kg','frcst_qty','model']]
        ip['SalesDistCode']=ip['SalesDistCode'].astype(str)
        ip.loc[ ip['SalesDistCode']=='0', ['SalesDistCode']]='0000'
        return ip
    # ============================================================================

    # Lag wise accuracy
    # ----------------------------------------------------------------------------
    def lagwiseacc(ip,nlag):
        accs=ip.groupby(['lag']).agg({"ProdSalesCode":pd.Series.nunique,"qty_kg": np.sum, "frcst_qty": np.sum})
        accs['Acc']=round((1-(abs(accs['qty_kg']-accs['frcst_qty'])/accs['qty_kg']))*100,2)
        accs['Acc']=accs['Acc'].replace([np.inf, -np.inf], np.nan)
        accs.loc[:, "Acc"] = accs["Acc"].map('{:.2f}'.format)
        accs["qty_kg"]=round(accs["qty_kg"],0)
        accs["frcst_qty"]=round(accs["frcst_qty"],0)
        accs.rename(columns={"ProdSalesCode":"CustomerCount","qty_kg":"Hist_Qty","frcst_qty":"Fcst_Qty"},inplace=True)
        m=ip["model"].unique().tolist()
        print("[-------------------- Accuracy of  %s --------------------]"%m[0])
        accs=accs.head(nlag)
        return accs
    # ============================================================================

    # Import Original Dataset
    # ---------------------------------------------------------------------------
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

    # Make CSV and store it to workspace
    # ----------------------------------------------------------------------------
    def to_container(filename):
        subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6'
        resource_group = 'AI-QAT1-RG'
        workspace_name = 'mlworkspaceingredion'

        workspace = Workspace(subscription_id, resource_group, workspace_name)
        datastore = Datastore(workspace, 'aiml_deltamart')
        datastore.upload_files(['../Datasets/'+filename], target_path = 'mlmodel/MLDemandForecast/final_fcst_output', overwrite = True)
    # ============================================================================

    # Main
    # ----------------------------------------------------------------------------
    def main():
        ws = Workspace.from_config()
        # Default datastore 
        def_data_store = ws.get_default_datastore()
        # Get the blob storage associated with the workspace
        def_blob_store = Datastore(ws, "aiml_deltamart")

        dataset = Dataset.File.from_files([(def_blob_store, '/mlmodel/MLDemandForecast/final_fcst_output/*.csv')])

        download_paths = dataset.download(target_path = None, overwrite=True)

        print('Download Paths')
        print('----------------------------------------------------')
        print('\n'.join(download_paths))
        
        # ************************************************************************
        # Variables
        nlag = 2

        # ************************************************************************
        # Holt - Winters
        holtwinter=pd.read_csv(download_paths[0], index_col = None)
        holtwinter=Best_Model_Selection.Preprocess(holtwinter)
        print(Best_Model_Selection.lagwiseacc(holtwinter, nlag))

        # ************************************************************************
        # SARIMA
        sarima=pd.read_csv(download_paths[1], index_col = None)
        sarima=Best_Model_Selection.Preprocess(sarima)
        print(Best_Model_Selection.lagwiseacc(sarima, nlag))

        # ************************************************************************
        # XGBoost
        xgb=pd.read_csv(download_paths[3], index_col = None)
        xgb=Best_Model_Selection.Preprocess(xgb)
        print(Best_Model_Selection.lagwiseacc(xgb, nlag))

        # ************************************************************************
        # Master table
        master = pd.DataFrame()
        master = master.append(sarima)
        master = master.append(holtwinter)
        master = master.append(xgb)

        # ************************************************************************
        # Master table logic
        master['mape']=abs(master['qty_kg']-master['frcst_qty'])/master['qty_kg']
        master['mape']=master['mape'].replace([np.inf, -np.inf],1)
        master['mape']=master['mape'].replace([np.nan],0)
        master['Acc']=1-master['mape']

        master=master[['ProdSalesCode', 'ProductNumber', 'Productdescr', 'SalesDistCode', 'SalesDistrict', 'cust_segment', 'Last record date', 'yr', 'month', 'forecast_date', 'lag', 'qty_kg', 'frcst_qty', 'mape', 'Acc', 'model']]

        list=master['ProdSalesCode'].unique().tolist()
        a=master.columns.values.tolist()
        op=pd.DataFrame(columns=a)
        for i in list:
            mdl = master['model'].loc[master['ProdSalesCode']==i].unique().tolist()
            mdlc = master['model'].loc[master['ProdSalesCode']==i].nunique()
            arr_reslt=np.zeros((mdlc,3),dtype=float)
            x=0
            mape=[]
            for n in mdl :
                for m in [1,2]:
                    d= master.loc[(master['ProdSalesCode']==i) & (master['model']==n) & (master['lag']==m)]
                    r=d['mape'].values
                    if len(r)==0: r=0
                    else: r=float(r)
                    arr_reslt[x][m-1]=r
                arr_reslt[x][m]=(arr_reslt[x][m-2]+arr_reslt[x][m-1])/2
                x=x+1
            
            for cnt in range(0,mdlc) :
                mape.insert(cnt,arr_reslt[cnt][2])
            win = mdl[mape.index(min(mape))]
            op=op.append(master.loc[(master['ProdSalesCode']==i) & (master['model']==win)])

        op.rename(columns={"cust_segment":"Cust_Segment"
                            ,"Last record date":"Last_Record_Date"
                            ,"yr":"Year"
                            ,"month":"Month"
                            ,"forecast_date":"Forecast_Date"
                            ,"lag":"Lag"
                            ,"qty_kg":"Hist_Qty"
                            ,"frcst_qty":"Fcst_Qty"
                            ,"mape":"MAPE"
                            ,"Acc":"Accuracy"
                            ,"model":"Model"},inplace=True)

        
        df = Best_Model_Selection.import_original_dataset(subscription_id = '33105e1c-8af8-4d39-b1d1-dd5700f168c6',
                                    resource_group = 'AI-QAT1-RG', 
                                    workspace_name = 'mlworkspaceingredion')
        df = Best_Model_Selection.rename_col(df)
        df = Best_Model_Selection.preprocess(df) 

        df = df[(df["cust_segment"].isin(['GROUP_2', 'GROUP_3']))]
        product_sales_dist_list = df['ProdSalesCode'].unique()

        # *************************************************************************
        nsteps = 26
        # Forecast Start Date
        frcst_start = datetime(2015,1,1).date()

        # Last Record Date
        first_date_of_current_month = datetime.today().replace(day = 1)
        last_rec_dt = first_date_of_current_month + relativedelta(months = -4)
        last_rec_dt = last_rec_dt.date()

        # Forecast End Date
        frcst_end = last_rec_dt + relativedelta(months = nsteps)
        # **********************************************************************

        for comb in product_sales_dist_list:
            temp_df = df[(df['ProdSalesCode'] == comb)]
            output_df = op[(op['ProdSalesCode'] == comb)]

            imputed_df = Best_Model_Selection.impute(temp_df, frcst_start, frcst_end)
            imputed_df = imputed_df[['dt', 'qty_kg']]
            train, test = Best_Model_Selection.train_test_splitting(imputed_df, last_rec_dt)
            
            try:
                train_max = train['qty_kg'].max()
                fcst_max = output_df['Fcst_Qty'].max()
                ratio = fcst_max/train_max

                if ratio >= 5:
                    op.drop(output_df.index, inplace = True)
            except:
                continue

        # ************************************************************************
        # Store the Master table in CSV format
        filename = '04_DmdFrcstMaster.csv'
        op.to_csv('../Datasets/' + filename,index='False',header='True')
        print('CSV file saved in workspace')
        Best_Model_Selection.to_container(filename)
        print('CSV file saved in container')
        
        # ************************************************************************
        # Model wise count
        group=op.groupby(['ProdSalesCode','Model'])
        group=group.size()
        df=group.groupby(['Model'])
        CountP=op['ProdSalesCode'].nunique()
        print('----------------------------------------------------')
        print("Unique ProductSalesCode Count = ",CountP)
        print(df.count().sort_values(ascending=False))

        # ************************************************************************
        # Master table lagwise accuracy
        Accuracy=op.groupby(['Lag']).agg({"ProdSalesCode":pd.Series.nunique,"Hist_Qty": np.sum, "Fcst_Qty": np.sum})
        Accuracy.rename(columns={"ProdSalesCode":"CustomerCount"},inplace=True)
        Accuracy['Accuracy']=round((1-(abs(Accuracy['Hist_Qty']-Accuracy['Fcst_Qty'])/Accuracy['Hist_Qty']))*100,2)
        Accuracy['Accuracy']=Accuracy['Accuracy'].replace([np.inf, -np.inf], np.nan)
        Accuracy.loc[:, "Accuracy"] = Accuracy["Accuracy"].map('{:.2f}'.format)
        print('-------------------------------------------------------')
        print(Accuracy.head(nlag))
        
    # ============================================================================