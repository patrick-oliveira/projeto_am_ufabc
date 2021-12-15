import numpy as np
import pandas as pd
from copy import deepcopy           # Necessário para copiar os dados dentro de funções e evitar alterações inplace dos dados
                                    # Isso para que as funções recebam um dado e gerem um dado novo, mantendo o original inalterado.
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


# TODO: converter para um arquivo requirements.txt 
# !pip install -U imbalanced-learn
# !pip install pandas
# !pip install numpy
# !pip install -U scikit-learn
# !pip install seaborn

class Preprocessor():
    '''
        Saída final esperada: um dicionário com as saídas 
    '''
    users_data = {}
        # colocar as variáveis de interesse da classe:
        # especialmente o dataframe base, e dados de trabalho
    def __init__(self, Data_path = None):
        self.raw_data = 0
        self.output_data = 0
        users_data = {}
        if not (Data_path is None):
            self.load_dataset(Data_path)
        # colocar as variáveis de interesse da classe:
        # especialmente o dataframe de entrada
        pass

    def load_dataset(self,Data_path):
            self.raw_data = pd.read_csv(Data_path)
            self.output_data = self.raw_data

    def bool_to_int(self):
        '''
        
        Converte as colunas do tipo bool para o tipo int (0 e 1).
        
        '''
        columns = ['Tem_Instr_Violao_Viola', 'Tem_Instr_Guitarra', 'Tem_Instr_Cavaco', 
                'Tem_Instr_Sintetizador_Teclado', 'Tem_Instr_Piano', 'Tem_Instr_Metais', 
                'Tem_Instr_Madeiras', 'Tem_Instr_Cordas', 'gostou']                      # Adicionar as colunas da bateria aqui depois
        # self.output_data = deepcopy(dataframe)

        for col in columns:
            self.output_data[col] = self.output_data[col].astype(int)

    

    def bateria_to_bool(self):
        '''
        
        Transforma a coluna bateria para ser representada por binários
        Essa função pode ser modificada para processar a coluna bateria de formas distintas.
        
        '''
        # TODO melhorar internamente
        new_df = self.output_data
        new_df['bateria_eletronica'] = (new_df['bateria'] == 'Eletrônica').astype(int)
        new_df['bateria_acustica']   = (new_df['bateria'] == 'Acústica').astype(int)
        new_df['bateria_nenhuma']     = (new_df['bateria'] == 'Nenhuma').astype(int)
        new_df = new_df.drop(columns = ['bateria'])
        self.output_data = new_df

    
    def scale_to_bool(self,mode):
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
        }[mode]
        return string


    def mode_to_one_hot(self):
        ''''
            Aplica One hot encoding nos dados de modo, criando um conjunto de colunas binárias
        '''
        # TODO melhorar internamente 
        new_df = deepcopy(self.output_data)
        mode_df = pd.get_dummies(new_df["modo"])
        new_df = new_df.drop(columns = ['modo'])
        #  Place the DataFrames side by side
        new_df = pd.concat([new_df,mode_df], axis=1)
        self.output_data = new_df

    def generate_users_dict(self):
        users_list = self.output_data["id_cliente"].value_counts()
        print(f"Usuarios: {[x for x in users_list.keys()]}")
        for user in users_list.keys():
            user_data = self.output_data[self.output_data['id_cliente'] == user]
            self.users_data[user] = user_data
        # print(users_data[users_list.keys()[1]])


    def preprocess_dataset(self, use_major_scale_col = False):    
        '''
            Aplica todas as estratégias de preprocessamento para adequar à estrutura entendida pelo sklearn
        '''
        self.output_data['PctCantada'] = self.output_data['PctCantada'] / 100
        self.output_data['PctRap'] = self.output_data['PctRap'] / 100
        self.output_data['duracao'] = self.output_data['duracao'] / (60*1000)
        self.output_data['VolMedio'] = self.output_data['VolMedio'].abs()
        self.output_data['duracao']  = self.output_data['duracao'].abs()
        self.output_data['modo'] = self.output_data['modo'].fillna('K')
        self.bool_to_int()
        self.bateria_to_bool()
        self.mode_to_one_hot() # Não gostei muito da solução, mas até faz sentido

        if use_major_scale_col:
            pass
            # self.output_data['escala_maior'] = self.output_data['modo'].apply((lambda mode: self.scale_to_bool(mode)))
        self.generate_users_dict()


    def get_user_data(self, user_ID):
        # definir melhor a interface
        user_data = self.users_data[user_ID]
        # user_data = self.output_data[self.output_data['id_cliente'] == USER]
        return user_data


    def train_test_split(self,user_data, oversampling = None): #adicionar parametros de entrada
        input_data = user_data.drop(columns = ["data_curtida", "id_cliente"])
        X = input_data.drop(columns = ["gostou"]).to_numpy()
        Y = input_data["gostou"].to_numpy()
        Y = Y.ravel()
        print(f"Labels shape: {Y.shape}")
        print(f"Features shape{X.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=109) # 70% training and 30% testv

        if oversampling is not None:
            print(f"with {oversampling} oversampling")
            if oversampling == "SMOTE":
                X_train_res, y_train_res = self.apply_SMOTE_oversampling(X_train, y_train)          
                return X_train_res, X_test, y_train_res, y_test 

            elif oversampling =="SMOTENC":
                X_train_res, y_train_res = self.apply_SMOTENC_oversampling(X_train, y_train)
                return X_train_res, X_test, y_train_res, y_test 

        return X_train, X_test, y_train, y_test 


    def apply_SMOTENC_oversampling(self, X_train, y_train):
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

    def apply_SMOTE_oversampling(self, X_train, y_train):
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


    def check_balancing(self):
        ## Count elements
        count_plus = 0
        count_minus = 0

        for y_val in Y_train_res:
            
            if y_val == 1:
                count_plus +=1
            else: 
                count_minus +=1
        print("Positive examples:",count_plus)
        print("Negative examples:",count_minus)

