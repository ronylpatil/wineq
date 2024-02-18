import pathlib
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn import metrics
from src.logger import infologger
from dvclive import Live

infologger.info('*** Executing: predict_model.py ***')
from src.data.make_dataset import load_data

def load_model(model_dir: str) -> BaseEstimator :
     try : 
          model = joblib.load(model_dir)
     except Exception as e : 
          infologger.info(f'there\'s an issue while loading the model from {model_dir} [check load_model()]. exc: {e}')
     else : 
          infologger.info(f'model loaded successfully from {model_dir}')
          return model


def evaluate(x_test: pd.DataFrame, y_test: np.ndarray, model: BaseEstimator) -> None : 
     # try : 
     #      y_pred = model.predict(x_test)     # return class
     #      y_pred_prob = model.predict_proba(x_test)    # return probability
     #      accuracy = metrics.accuracy_score(y_test, y_pred)
     #      precision = metrics.precision_score(y_test, y_pred, zero_division = 1, average = 'macro')
     #      recall = metrics.recall_score(y_test, y_pred, average = 'macro')
     #      roc_score = metrics.roc_auc_score(y_test, y_pred_prob, average = 'macro', multi_class = 'ovr')
     #      infologger.info('model evalution done')
     #      try : 
     #           with Live(resume = True) as live : 
     #                live.log_metric('testing/bal_accuracy', float('{:.2f}'.format(accuracy)))
     #                live.log_metric('testing/roc_score', float('{:.2f}'.format(roc_score)))
     #                live.log_metric('testing/precision', float("{:.2f}".format(precision)))
     #                live.log_metric('testing/recall', float("{:.2f}".format(recall)))
     #           infologger.info('performance metrics tracked by dvclive')
     #      except Exception as ie : 
     #           infologger.info(f'there\'s an issue while tracking the performance metrics [check evaluate()]. exc: {ie}')
     # except Exception as oe : 
     #      infologger.info(f'there\'s an issue while evalution [check evaluate()]. exc: {oe}')

     try : 
          y_pred = model.predict(x_test)     # return class
          y_pred_prob = model.predict_proba(x_test)    # return probability
     except Exception as oe : 
          infologger.info(f'there\'s an issue while prediction [check evaluate()]. exc: {oe}')
     else : 
          try : 
               bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
               precision = metrics.precision_score(y_test, y_pred, zero_division = 1, average = 'macro')
               recall = metrics.recall_score(y_test, y_pred, average = 'macro')
               roc_score = metrics.roc_auc_score(y_test, y_pred_prob, average = 'macro', multi_class = 'ovr')
          except Exception as e :
               infologger.info(f'there\'s an issue while evalution [check evaluate()]. exc: {e}')
          else : 
               infologger.info('model evalution done')
               try : 
                    with Live(resume = True) as live : 
                         live.log_metric('testing/bal_accuracy', float('{:.2f}'.format(bal_acc)))
                         live.log_metric('testing/roc_score', float('{:.2f}'.format(roc_score)))
                         live.log_metric('testing/precision', float("{:.2f}".format(precision)))
                         live.log_metric('testing/recall', float("{:.2f}".format(recall)))
               except Exception as ie : 
                    infologger.info(f'there\'s an issue while tracking the performance metrics [check evaluate()]. exc: {ie}')
               else :
                    infologger.info('performance metrics tracked by dvclive')

def main() -> None : 
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent
     try : 
          params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))
     except Exception as e :
          infologger.info(f'there\'s an issue while loading params.yaml [check main()]. exc: {e}')
     else :
          data_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"
          model_dir = f'{home_dir.as_posix()}{params["train_model"]["model_dir"]}/model.joblib'
          test_data = load_data(data_dir)
          TARGET = params['base']['target']
          x_test = test_data.drop(columns = [TARGET]).values
          y_test = test_data[TARGET]

          evaluate(x_test, y_test, load_model(model_dir))
          infologger.info('program terminated normally')

if __name__ == '__main__' :
     infologger.info('predict_model.py as __main__')
     main()
