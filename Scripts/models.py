import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from typing import List, Tuple, Dict

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from skorch import NeuralNet
from skorch.dataset import CVSplit

from .data_processing import DataManager


torch.cuda.is_available()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
random_seed = 42
np.random.seed(random_seed)
    
####
#
# Classes para avaliação das músicas
#
###

class Valuer(object):
    def __init__(self, classification_model, regression_model) -> None:
        super(Valuer, self).__init__()
        self.classifier = classification_model
        self.regressor  = regression_model
        
    def predict(self, X: np.array) -> np.array:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        return np.array([self.classifier.predict(X), self.regressor.predict(X)]).T
        
####
#
# Classes para o Autoencoder Variacional
#
###

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 43, latent_dim: int = 30, hidden_dims: List[int] = [40, 35]) -> None:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.Dropout(0.1),
                nn.ReLU()
            ))
            input_dim = h_dim
            
        self.Encoder = nn.Sequential(*modules)
        
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.N = torch.distributions.Normal(0, 1)
        
        modules = []
        modules.append(nn.Linear(latent_dim, hidden_dims[-1]))
        hidden_dims.reverse()
        hidden_dims.append(self.input_dim)
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ))
        self.Decoder = nn.Sequential(*modules)
        
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
            self.Encoder = self.Encoder.cuda()
            self.Decoder = self.Decoder.cuda()
        
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        X = self.Encoder(X)
        mu = self.mu(X)
        log_var = self.log_var(X)
        sigma = torch.exp(0.5 * log_var)
        
        Z = mu + sigma * self.N.sample(mu.shape)
        
        X = self.Decoder(Z)
        
        return X, mu, sigma
    
class VAELoss(nn.Module):
    def __init__(self) -> None:
        super(VAELoss, self).__init__()
    
    def forward(self, model_output, X) -> torch.Tensor:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        Xhat, mu, log_var = model_output
        KL_Divergence = - 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        Reconstruction_Loss = F.mse_loss(Xhat, X)
        
        loss = Reconstruction_Loss + KL_Divergence
        
        return loss
    
####
#
# Classe responsável por gerar a lista de músicas (e possivelmente fazer algum processamento)
#
### 
    
class Recommender(object):
    def __init__(self, generativeModel: VariationalAutoencoder, evaluationModel: Valuer, scaler: MinMaxScaler, user: str):
        # Adicionar o modelo de avaliação
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        super(Recommender, self).__init__()
        
        self.generativeModel = generativeModel
        self.evaluationModel = evaluationModel
        self.latent_dim = generativeModel.latent_dim
        self.scaler = scaler
        self.user = user
        
    def sample(self, n_samples: int) -> np.array:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        Z = torch.cat([self.generativeModel.N.sample(sample_shape = (1, self.latent_dim)) for n in range(n_samples)])
        Xhat = self.generativeModel.Decoder(Z).squeeze().detach().cpu().numpy()
        Xhat = self.scaler.inverse_transform(Xhat)
        return Xhat
    
    def getMusicList(self, n_musics: int, music_dataset: pd.DataFrame) -> np.array:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        # usa o modelo generativo para obter uma lista de novas músicas
        samples = self.sample(2*n_musics)
        
        # assumindo que esse dataset já está no formato necessário, i.e. sem as colunas do ID do usuário, curtida, data da curtida e n_reprod.
        # transforma em array do numpy
        music_att_array = music_dataset.values
        
        # aplica uma função nos rows comparando as distâncias angulares de cada sample com as músicas do dataset
            # pega o índice da música mais próxima
            # gera uma lista com os índices, um para cada amostra
        min_distances_indices = [np.apply_along_axis(lambda x: np.linalg.norm(x - sample), 
                                                     1,
                                                     music_att_array).argmin() for sample in samples]
        
        # seleciona as músicas reais
        music_list = music_dataset.loc[min_distances_indices]
        
        # Filtra as músicas
        if self.evaluationModel != None:
            music_list['evaluation'] = [tuple(x) for x in self.evaluationModel.predict(music_list.values)]
            music_list = music_list.sort_values('evaluation', ascending = False)
            music_list = music_list.iloc[:n_musics]
        
        # retorna a lista de músicas reais
        return music_list
    
    def test_model(self, user_dataset: pd.DataFrame, n_musics: int = 20) -> Tuple[float]:
        """
        Comentário.

        Args:
            None.

        Returns:
            None.
        """
        # usa o modelo generativo para obter uma lista de novas músicas
        samples = self.sample(2*n_musics)
        
        music_att_array = user_dataset.drop(columns = ['id_cliente', 'gostou', 'data_curtida', 'n_reproducao']).values
        
        min_distances_indices = [np.apply_along_axis(lambda x: np.linalg.norm(x - sample),
                                                     1,
                                                     music_att_array).argmin() for sample in samples]
        
        music_list = user_dataset.iloc[min_distances_indices]
        
        if self.evaluationModel != None:
            evaluation = [tuple(x) for x in self.evaluationModel.predict(music_list.drop(columns = ['id_cliente', 'gostou', 'data_curtida', 'n_reproducao']).values)]
            music_list['evaluation'] = evaluation
            music_list['predict_gostou'] = [predict[0] for predict in evaluation]
            music_list['predict_n_reproducao'] = [predict[1] for predict in evaluation]
            music_list = music_list.sort_values('evaluation', ascending = False)
            music_list = music_list.iloc[:n_musics]
            
        gostou_real = music_list['gostou'].values
        gostou_predict = music_list['predict_gostou'].values
        n_reprod_real = music_list['n_reproducao'].values
        n_reprod_predict = music_list['predict_n_reproducao'].values
        
        classification_error = metrics.accuracy_score(gostou_real, gostou_predict)
        regression_error = metrics.mean_squared_error(n_reprod_real, n_reprod_predict)
        
        return classification_error, regression_error
            
####
#
# Funções para o treinamento dos modelos
#
### 
        
def train_evaluation_system(manager: DataManager, USER: str) -> Tuple[DecisionTreeClassifier, LinearRegression]:
    """
    Comentário.

    Args:
        None.

    Returns:
        None.
    """
    X_train, X_test, y_train, y_test = manager.get_training_data(USER, test_size = 0.2, oversampling = 'SMOTENC')
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    print(f"Classifier Model: {classifier}")
    print(f"Accuracy Score: {metrics.accuracy_score(classifier.predict(X_test), y_test)}")
    print()
    
    X_train, X_test, y_train, y_test = manager.get_training_data(USER, test_size = 0.2, classification = False)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(f"Regression Model: {regressor}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(regressor.predict(X_test), y_test)}")
    print()
    
    return classifier, regressor


def train_generative_system(manager: DataManager, USER: str, verbose: bool = True) -> Tuple[NeuralNet, MinMaxScaler]:
    """
    Comentário.

    Args:
        None.

    Returns:
        None.
    """
    
    skorch_model = NeuralNet(
        module = VariationalAutoencoder,
        module__latent_dim = 35,
        module__hidden_dims = [40],
        criterion = VAELoss,
        optimizer = torch.optim.Adam,
        lr = 0.0001,
        max_epochs = 200,
        batch_size = 100,
        iterator_train__shuffle = True,
        train_split = CVSplit(0.3),
        device = device,
        verbose = verbose
    )
    
    print("Generative Model:")
    print(skorch_model)
    
    # Ajuste de escala "treinado" para os dados do usuário selecionado
    scaler = MinMaxScaler()
    
    # Seleciona o conjunto de dados com curtida positiva, normaliza e transforma em torch.Tensor
    X, Y = manager.data_arrays(USER)
    X = X[Y[:, 0] == 1]
    scaler.fit(X)
    X = scaler.transform(X)
    X = torch.tensor(X).float().to(device)

    # Ajusta o modelo
    skorch_model.fit(X, X)
    print()
    
    return skorch_model, scaler