import pathlib
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.logger import infologger

infologger.info('*** Executing: visualize.py ***')
from src.models.predict_model import load_model
from src.data.make_dataset import load_data

def roc_curve() -> None : 
     pass

def conf_matrix(y_test: pd.Series, y_pred: pd.Series, labels: np.ndarray, path: str) -> None : 
     try : 
          current_time = datetime.now().strftime('%d%b%y-%H.%M.%S')
          pathlib.Path(path).mkdir(parents = True, exist_ok = True)
          # directory_path = pathlib.Path(path)
          # directory_path.mkdir(parents = True, exist_ok = True)
          # pathlib.Path(f"{directory_path}/cm_{current_time}").mkdir()
     except Exception as e : 
          infologger.info(f'there\'s an issue in directory [check conf_metrix()]. exc: {e}')
     else :
          infologger.info('directories are all set!')
          try :
               cm = confusion_matrix(y_test, y_pred, labels = labels)
               disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
               disp.plot(cmap = plt.cm.Blues)
               plt.title('Confusion Matrix')
               plt.xlabel('Predicted Label')
               plt.ylabel('True Label')
               plt.savefig(f'{path}/cM-{current_time}.png')
               plt.close()
          except Exception as e : 
               infologger.info(f'there\'s an issue in ploting confusion metrix [check conf_metrix()]. exc: {e}')
          else :
               infologger.info(f'confusion metrix saved at [{path}]')

def main() -> None :
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent
     dir_path = pathlib.Path(f'{home_dir.as_posix()}/plots')
     # dir_path.mkdir(parents = True, exist_ok = True)

     try : 
          params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))
     except Exception as e : 
          infologger.info(f'there\'s some issue while loading params.yaml [check main()]. exc: {e}')
     else :
          data_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"
          model_dir = f'{home_dir.as_posix()}{params["train_model"]["model_dir"]}/model.joblib'
          
          TARGET = params['base']['target']

          test_data = load_data(data_dir)
          x_test = test_data.drop(columns = [TARGET]).values
          y_test = test_data[TARGET]
          
          model = load_model(model_dir)
          labels = model.classes_
          try : 
               y_pred = model.predict(x_test)     # return class
               # y_pred_prob = model.predict_proba(x_test)    # return probability
          except Exception as e : 
               infologger.info(f'there\'s an issue while prediction [check main()]. exc: {e}')
          else :
               conf_matrix(y_test, y_pred, labels, dir_path)
               infologger.info('program terminated normally!')
               
if __name__ == '__main__' :
     main()
