import pathlib
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dvclive import Live
from src.logger import infologger
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.base import BaseEstimator
from typing import Tuple

infologger.info('*** Executing: train_model.py ***')
# writing import after infologger to log the info precisely 
from src.data.make_dataset import load_data

def train_model(training_feat: pd.DataFrame, y_true: pd.Series, n_estimators: int, criterion: str, max_depth: int, random_state: int) -> Tuple[BaseEstimator, np.ndarray] :
     try : 
          model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth,
                                        random_state = random_state)
          model.fit(training_feat, y_true)
     except Exception as e :
          infologger.info(f'there\'s an issue while training model [check train_model()]. exc: {e}')
     else :
          infologger.info(f'trained {type(model).__name__} model')
          y_pred = model.predict(training_feat)
          y_pred_prob = model.predict_proba(training_feat)
          accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
          precision = metrics.precision_score(y_true, y_pred, zero_division = 1, average = 'macro')
          recall = metrics.recall_score(y_true, y_pred, average = 'macro')
          roc_score = metrics.roc_auc_score(y_true, y_pred_prob, average = 'macro', multi_class = 'ovr')

          try : 
               with Live(resume = True) as live : 
                    live.log_param('n_estimators', n_estimators)
                    live.log_param('criterion', criterion)
                    live.log_param('max_depth', max_depth)
                    live.log_param('random_state', random_state)

                    live.log_metric('training/bal_accuracy', float('{:.2f}'.format(accuracy)))
                    live.log_metric('training/roc_score', float('{:.2f}'.format(roc_score)))
                    live.log_metric('training/precision', float("{:.2f}".format(precision)))
                    live.log_metric('training/recall', float("{:.2f}".format(recall)))
          except Exception as ie : 
               infologger.info(f'there\'s an issue while tracking metrics/parameters using dvclive [check train_model()]. exc: {ie}')
          else : 
               infologger.info('parameters/metrics tracked by dvclive')
               return model, y_pred

def save_model(model: BaseEstimator, model_dir: str) -> None : 
     try : 
          joblib.dump(model, f'{model_dir}/model.joblib')
     except Exception as e : 
          infologger.info(f'there\'s an issue while saving the model [check save_model(). exc: {e}')
     else :
          infologger.info(f'model saved at {model_dir}')

def main() -> None : 
     curr_path = pathlib.Path(__file__) 
     home_dir = curr_path.parent.parent.parent
     params_loc = f'{home_dir.as_posix()}/params.yaml'
     try : 
          params = yaml.safe_load(open(params_loc, encoding = 'utf8'))
     except Exception as e :
          infologger.info(f'there\'s an issue while loading params.yaml [check main()]. exc: {e}')
     else : 
          parameters = params['train_model']
          TARGET = params['base']['target']

          train_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
          model_dir = f"{home_dir.as_posix()}{parameters['model_dir']}"
          pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
          
          data = load_data(train_data)
          X_train = data.drop(columns = [TARGET])
          Y = data[TARGET]

          model, _ = train_model(X_train, Y, parameters['n_estimators'], parameters['criterion'], parameters['max_depth'],
                                   parameters['seed'])
          save_model(model, model_dir)
          infologger.info('program terminated normally!')

if __name__ == '__main__' : 
     infologger.info('train_model.py as __main__')
     main()
     