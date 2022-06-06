# Import libraries and forecast models
# ---------------------------------------------------------------------------
from azureml.core import Workspace, Datastore, Dataset
from HW_ProdSales_Forecast import HW_ProdSales_Forecast
from XGB_ProdSales_Forecast import XGB_ProdSales_Forecast
from SARIMA_ProdSales_Forecast import SARIMA_ProdSales_Forecast
from Best_Model_Selection import Best_Model_Selection
# ============================================================================

# Create DMD_FRCST_INPUT dataset
# ---------------------------------------------------------------------------
def create_input_dataset(datastore_name, init_path, dataset_name):
    workspace = Workspace.from_config()
    def_blob_store = Datastore.get(workspace, datastore_name)

    dataset = Dataset.File.from_files([(def_blob_store,  init_path + '*.csv')])
    download_paths = dataset.download(target_path = None, overwrite=True)

    path = init_path + download_paths[0].split('/')[-1]

    datastore_paths = [(def_blob_store, path)]

    try:
        dataset = Dataset.Tabular.from_delimited_files(path = datastore_paths, header = False)
        print(dataset.register(workspace, dataset_name, create_new_version=True))

    except Exception as e:
        print(e)
# ============================================================================

def main():
    # Create input dataset (DMD_FRCST_INPUT)
    # **********************************************************************
    datastore_name = 'aiml_deltamart'
    init_path = '/mlmodel/MLDemandForecast/sap_dmdfrcst_input/'
    dataset_name = 'DMDFRCST_FRCST_INPUT'

    create_input_dataset(datastore_name, init_path, dataset_name)
    # **********************************************************************

    # Holt - Winters Model
    HW_ProdSales_Forecast.main()
    # **********************************************************************

    # XGBoost Model
    XGB_ProdSales_Forecast.main()
    # **********************************************************************

    # SARIMA Model
    SARIMA_ProdSales_Forecast.main()
    # **********************************************************************

    # Select the best model
    # Best_Model_Selection.main()
    # **********************************************************************

if __name__ == '__main__':
    main()