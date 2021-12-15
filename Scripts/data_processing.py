import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SMOTENC

random_seed = 42
np.random.seed(random_seed)

from copy import deepcopy


#     """
#     Comentário.

#     Args:
#         None.

#     Returns:
#         None.
#     """

class DataManager(object):
    def __init__(self, path: str):
        super(DataManager, self).__init__()
        self._raw_data = pd.read_csv(path)
        self._processed_data = None        # Recebe informação pelo método preprocess() 
        
        self.preprocess()
        
    @property
    def raw_data(self) -> pd.DataFrame:
        return self._raw_data
    
    @property
    def data(self) -> pd.DataFrame:
        return self._processed_data
    
    @property
    def users(self) -> List[str]:
        return list(self.data['id_cliente'].drop_duplicates())
    
    def user_data(self, user: str) -> pd.DataFrame:
        """
        Coleta os dados correspondentes a um usuário especificado. Para a execução do programa caso os dados sejam inexistentes.
        
        **
        Parar a execução não é a melhor solução mas comumente seguir com um DataFrame vazio gera outros problemas.
        **
        
        Args:
            user (str): ID do usuário.
            
        Returns:
            pd.DataFrame: Um DataFrame do Pandas com os dados - pré-processados - do usuário especificado.
            
        """
        data = self.data[self.data['id_cliente'] == user]
        
        assert len(data) > 0, 'Dados inexistentes para o usuário especificado.'
        
        return data
        
    def preprocess(self) -> None:
        """
        Executa todas as operações de pré-processamento sobre os dados puros. Estes são salvos na memória como o atributo "raw_data", e uma cópia é gerada para o pré-processamento.
        
        Args:
            None.
            
        Returns:
            None.
        """
        self._processed_data = deepcopy(self.raw_data)
        
        self._processed_data['PctCantada'] = self._processed_data['PctCantada'] / 100
        self._processed_data['PctRap']     = self._processed_data['PctRap'] / 100
        self._processed_data['duracao']    = self._processed_data['duracao'] / (60*1000)
        self._processed_data['duracao']    = self._processed_data['duracao'].abs()
        self._processed_data['VolMedio']   = self._processed_data['VolMedio'].abs()
        self._processed_data['modo']       = self._processed_data['modo'].fillna('K')
        self._processed_data['escala_maior'] = self._processed_data['modo'].apply(lambda mode: scale_to_bool(mode))
        self._processed_data = bool_to_int(self._processed_data)
        self._processed_data = bateria_to_bool(self._processed_data)
        self._processed_data = scale_to_one_hot(self._processed_data)
        
        columns = ['Tem_Instr_Violao_Viola', 'Tem_Instr_Guitarra', 'Tem_Instr_Cavaco',
                   'Tem_Instr_Sintetizador_Teclado', 'Tem_Instr_Piano', 'Tem_Instr_Metais',
                   'Tem_Instr_Madeiras', 'Tem_Instr_Cordas', 'escala_maior', 'bateria_eletronica', 
                   'bateria_acustica', 'bateria_nenhuma', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 
                   'F', 'F#', 'G', 'G#', 'K', 'a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 
                   'g', 'g#', 'ano_lancamento', 'BPM', 'VolMedio', 'PctCantada', 'PctRap', 'duracao',
                   'id_cliente', 'data_curtida', 'n_reproducao', 'gostou']

        self._processed_data = self._processed_data[columns]
        
    def data_arrays(self, user: str = None) -> Tuple[np.array]:
        """
        Retorna arrays com as features e targets, removendo informações desnecessárias. Se especificado, seleciona os dados de um usuário.

        Args:
            user (str): Default to None. Especifica um usuário cujas features serão obtidas

        Returns:
            Tuple[np.array]: Um array X de features e um array Y de targets. O array Y de targets contém duas colunas, a primeira relativa à coluna "gostou" e a segunda relativa à coluna "n_reproducao".
         """
        if user != None:
            data = self.user_data(user)
        else:
            data = self.data
            
        data = data.drop(columns = ['data_curtida', 'id_cliente'])
        
        X = data.drop(columns = ['gostou', 'n_reproducao']).to_numpy()
        Y = data[['gostou', 'n_reproducao']].to_numpy()
        
        return X, Y
    
    def get_training_data(self, user: str, test_size: float = None, classification: bool = True, oversampling: str = None) -> Tuple[np.array]:
        """
        Separa os dados em conjunto de treinamento e teste para o ajuste de um modelo de regressão ou classificação. Se especificado, seleciona os dados de um usuário.

        Args:
            user (str): Especifica o ID de um usuário para seleção dos dados.
            kwargs (Dict): Um dicionário com os argumentos a serem passados para a função train_test_split do sklearn.
            classification (bool): Default to True. Especifica se os dados correspondem ao ajuste do modelo de classificação ou regressão, de modo a escolher o target adequado.
            balance (bool): Default to False. Especifica se os dados de treinamento devem ser balanceados.

        Returns:
            Tuple[np.array]: Os arrays de atributos de treinamento e teste (X_train, X_test) e os targets de treinamento e teste (y_train, y_test).
        """
        X, Y = self.data_arrays(user)
        
        if classification:
            Y = Y[:, 0]
            if oversampling == 'SMOTENC':
                X, Y = SMOTENC_oversampling(X, Y)
#             elif oversampling == 'SMOTE':
#                 X, Y = SMOTE_oversampling(X, Y)
        else:
            Y = Y[:, 1]
        
        Y = Y.ravel()
        
        if test_size != None:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size)
        
            return X_train, X_test, y_train, y_test
        else:
            return X, Y
        
    
def bateria_to_bool(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma a coluna "bateria" para ser representada por one-hot-encoding.

    Args:
        dataframe (pd.DataFrame): Um DataFrame do Pandas com os dados a serem processados.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas com as informações da bateria codificadas como one-hot-encoding; a coluna original é descartada.
    """
    new_df = deepcopy(dataframe)
    new_df['bateria_eletronica'] = (new_df['bateria'] == 'Eletrônica').astype(int)
    new_df['bateria_acustica']   = (new_df['bateria'] == 'Acústica').astype(int)
    new_df['bateria_nenhuma']    = (new_df['bateria'] == 'Nenhuma').astype(int)
    new_df = new_df.drop(columns = ['bateria'])
    
    return new_df
    
        
def bool_to_int(dataframe: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Converte as colunas do tipo bool para o tipo int (0 e 1).

    Args:
        dataframe (pd.DataFrame): Um DataFrame do Pandas com os dados a serem processados.
        columns (List[str]): Uma lista das colunas que devem ser convertidas. Default = None. Se columns = None, um conjunto padrão de colunas será considerada para o processamento.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas com os dados processados.
    """

    if columns == None:
        columns = ['Tem_Instr_Violao_Viola', 'Tem_Instr_Guitarra', 'Tem_Instr_Cavaco', 
                   'Tem_Instr_Sintetizador_Teclado', 'Tem_Instr_Piano', 'Tem_Instr_Metais', 
                   'Tem_Instr_Madeiras', 'Tem_Instr_Cordas', 'gostou']
    
    new_df = deepcopy(dataframe)
    
    for col in columns:
        new_df[col] = new_df[col].astype(int)
        
    return new_df


def scale_to_one_hot(data: pd.DataFrame) -> pd.DataFrame:
    """
    Representa os dados de escala como one-hot-encodings.

    Args:
        data (pd.DataFrame): Um DataFrame do Pandas contendo os dados das músicas.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas com as informações de escalas transformadas para one-hot-encodings; a coluna original é descartada.
    """
    new_df = deepcopy(data)
    
    mode_df = pd.get_dummies(new_df['modo'])
    new_df = new_df.drop(columns = ['modo'])
    
    new_df = pd.concat([new_df, mode_df], axis = 1)
    
    return new_df

def scale_to_bool(scale: str) -> int:
    """
    Retorna como um booleano se uma dada escala é maior ou menor.

    Args:
        scale (str): Escala.

    Returns:
        (int): Um inteiro representando um booleano que especifica se a escala é maior.
    """
    string = {
        'K'  : 1,
        'c'  : 0,
        'c#' : 0, 
        'C'  : 1,
        'C#' : 1,
        'd'  : 0,
        'd#' : 0,
        'D'  : 1,
        'D#' : 1,
        'e'  : 0,
        'E'  : 1,
        'f'  : 0,
        'f#' : 0,
        'F'  : 1,
        'F#' : 1,
        'g'  : 0,
        'g#' : 0,
        'G'  : 1,
        'G#' : 1,
        'a'  : 0,
        'a#' : 0,
        'A'  : 1,
        'A#' : 1,
        'b'  : 0,
        'B'  : 1
    }[scale]
    return string


def SMOTENC_oversampling(X_train: np.array, y_train: np.array) -> Tuple[np.array]:
    """
    Faz upsampling utilizando o algoritmo SMOTENC para balancear os dados de treinamento.
    
    ** Essa função pode ser aprimorada incluindo mais de uma estratégia, mas provavelmente não vai dar tempo **

    Args:
        X_train (np.array): Array de features.
        y_train (np.array): Array de targets.
        
    Returns:
        Tuple[np.array]: A tupla (X_train_res, Y_train_res) com os dados balanceados.
    """
# colocar uns parametros uteis de entrada
# melhorar esse hardcode
    a = np.arange(0,8)
    b = np.array([11])
    c = np.arange(15,43)
    categorical_index = np.concatenate([a,c])
    categorical_index = np.concatenate([categorical_index,b])
    categorical_index = list(categorical_index)

    # Teste SMOTE-NC
    smote_nc_over = SMOTENC(categorical_features=categorical_index, random_state=0)

    under = RandomUnderSampler(sampling_strategy='majority',random_state=0)

    steps = [('o', smote_nc_over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_train_res, Y_train_res = pipeline.fit_resample(X_train, y_train)
    return X_train_res, Y_train_res

def SMOTE_oversampling(X_train: np.array, y_train: np.array) -> Tuple[np.array]:
    """
    Faz upsampling utilizando o algoritmo SMOTE para balancear os dados de treinamento.
    
    ** Essa função pode ser aprimorada incluindo mais de uma estratégia, mas provavelmente não vai dar tempo **

    Args:
        X_train (np.array): Array de features.
        y_train (np.array): Array de targets.
        
    Returns:
        Tuple[np.array]: A tupla (X_train_res, Y_train_res) com os dados balanceados.
    """
    # colocar uns parametros uteis de entrada
    raise ValueError("Do not use SMOTE to a categorical dataset: The result can be biased")
    # Teste do SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, Y_train_res = sm.fit_resample(X_train, y_train)

    under = RandomUnderSampler(sampling_strategy='majority',random_state=0)

    steps = [('o', ), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_train_res, Y_train_res = pipeline.fit_resample(X_train, y_train)
    return X_train_res, Y_train_res

def check_balancing(Y: np.array) -> Tuple[int]:
    """
    Conta o número de elementos em cada classe.

    Args:
        Y (np.array): Array de targets.

    Returns:
        Tuple[int]: Tupla (m, n) onde m é o número de elementos na classe 0 e n o número de elementos na classe 1.
    """
    count_plus = 0
    count_minus = 0
    
    for y_val in Y:
        if y_val == 1:
            count_plus += 1
        else:
            count_minus += 1
    
    return (count_minus, count_plus)
    