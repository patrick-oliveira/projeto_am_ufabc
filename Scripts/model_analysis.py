from typing import List, Dict, Tuple

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

random_seed = 42
np.random.seed(random_seed)

class ModelEvaluator(object):
    def __init__(self, classification_models: List[Tuple], regression_models: List[Tuple]) -> None:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        super(ModelEvaluator, self).__init__()
        
        # Os modelos são passados por uma lista de tuplas (classe, argumentos),
        # onde "classe" é a classe do modelo sem ser instanciado, e "argumentos"
        # é um dicionário com os argumentos a serem utilizados na instanciação de cada modelo.
        # Assim fica mais fácil de instanciar ao mesmo tempo vários modelos com combinações diferentes
        self.classification_models = [model_class(**args) for model_class, args in classification_models]
        self.regression_models     = [model_class(**args) for model_class, args in regression_models]
        
        
    def evaluate_classification_models(self, X_train: np.array, y_train: np.array, 
                                             X_test: np.array = None, y_test: np.array = None,
                                             evaluate_by: str = 'train_test_split', test_size: float = 0.33, plot_roc_curve: bool = False,
                                             choose_by: str = 'accuracy') -> pd.DataFrame:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        
       
        if evaluate_by == 'train_test_split':
            performance_scores = pd.DataFrame()
            for model in self.classification_models:
                print(f"Evaluating model: {str(model)}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy, precision, recall, roc_score, f1 = classification_metrics(y_test, y_pred)

                performance_scores.loc[str(model), 'accuracy']  = accuracy
                performance_scores.loc[str(model), 'precision'] = precision
                performance_scores.loc[str(model), 'recall']    = recall
                performance_scores.loc[str(model), 'roc_score'] = roc_score
                performance_scores.loc[str(model), 'f1']        = f1

                if plot_roc_curve:
                    plot_auc(y_test, y_pred, str(model))
                    
            best_model = self.classification_models[np.argmax(performance_scores[choose_by].values)]

        elif evaluate_by == 'cross_validation':
            performance_scores = {}
            choice_metric = pd.DataFrame()
            for model in self.classification_models:
                scores = cross_validate(model, X_train, y_train, scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1'])
                
                scores = {'accuracy': scores['test_accuracy'],
                          'precision': scores['test_precision'],
                          'recall': scores['test_recall'],
                          'roc_auc': scores['test_roc_auc'],
                          'f1': scores['test_f1']}
                
                performance_scores[str(model)] = scores
                choice_metric.loc[str(model), choose_by] = scores[choose_by].mean()
            
            best_model = self.classification_models[np.argmax(choice_metric[choose_by].values)]
                
        return performance_scores, best_model
        
    def evaluate_regression_models(self, X_train: np.array, y_train: np.array, 
                                         X_test: np.array = None, y_test: np.array = None,
                                         evaluate_by: str = 'train_test_split',  choose_by: str = 'mse') -> pd.DataFrame:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
       
        
        
        if evaluate_by == 'train_test_split':
            performance_scores = pd.DataFrame()
            for model in self.regression_models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae, mse, rmse = regression_metrics(y_test, y_pred)

                performance_scores.loc[str(model), 'mae']     = mae
                performance_scores.loc[str(model), 'mse']      = mse
                performance_scores.loc[str(model), 'rmse'] = rmse
                
            best_model = self.regression_models[np.argmin(performance_scores[choose_by].values)]

        elif evaluate_by == 'cross_validation':
            performance_scores = {}
            choice_metric = pd.DataFrame()
            for model in self.regression_models:
                scores = cross_validate(model, X_train, y_train, scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'])

                scores = {'mae': -1 * scores['test_neg_mean_absolute_error'],
                          'mse': -1 * scores['test_neg_mean_squared_error'],
                          'rmse': -1 * scores['test_neg_root_mean_squared_error']}

                performance_scores[str(model)] = scores
                choice_metric.loc[str(model), choose_by] = scores[choose_by].mean()
        
            best_model = self.regression_models[np.argmin(choice_metric[choose_by].values)]
        
        return performance_scores, best_model
        
def plot_auc(y_test: np.array, y_pred: np.array, model_name: str) -> None:
    """
    Comentário.

    Args:
        None.

    Returns:
        None.
    """
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    ax.plot(fpr, tpr, color='red', lw=2)
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {model_name}')
    plt.show()

    # salva a figura
        
        
def classification_metrics(y_test: np.array, y_pred: np.array) -> Tuple[float]:
    """
    Comentário.

    Args:
        None.

    Returns:
        None.
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    roc_score = metrics.roc_auc_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    return accuracy, precision, recall, roc_score, f1

def regression_metrics(y_test: np.array, y_pred = np.array) -> Tuple[float]:
    """
    Comentário.

    Args:
        None.

    Returns:
        None.
    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared = False)
    
    return mae, mse, rmse
    