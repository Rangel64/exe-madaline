import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score

entradas_train = np.asarray(pd.read_csv('all/archive/train/targets/train.csv'))/255 * 2 -1 
targets_train = np.asarray(pd.read_csv('all/archive/train/targets/targetsTrain.csv'))

entradas_test = np.asarray(pd.read_csv('all/archive/test/targets/test.csv'))/255 * 2 -1 
targets_test = np.asarray(pd.read_csv('all/archive/test/targets/targetsTest.csv'))

# entradas_validation = np.asarray(pd.read_csv('all/archive/validation/targets/validation.csv'))/255 * 2 -1 
# targets_validation = np.asarray(pd.read_csv('all/archive/validation/targets/targetsValidation.csv'))

class Perceptron:
    entrada = None
    target = None
    
    entrada_test = None
    target_test = None
    
    entradas = None
    saidas = None
    
    alpha = 0.01
    
    neuronios = 100
    
    camadaY = None
    
    Y = None
    
    XY = None
    
    biasY = None
    
    targetsTeste = []
    
    y_pred = []
    y_pred_test = []
    
    ciclo = 0
    
    erroTotal = np.Infinity
    erroTotal_test = np.Infinity
    trained = 0
    
    listaCiclo = []
    erros = []
    
    listaCiclo_test = []
    erros_test = []
    
    firstErro = 0
    
    erro_anterior = 0
    
    aleatorio = 2
    
    accuracy_train = 0
    accuracy_test = 0
    
    list_accuracy_train = []
    list_accuracy_test = []
    
    aleatorio = 0.5
    
    def get_weights_biases(self):
        return {
            'XY': self.XY,
            'biasY': self.biasY
        }
    
    def save_weights_biases(self, filename):
        
        weights_biases = self.get_weights_biases()
        
        with open(filename, 'wb') as file:
            pickle.dump(weights_biases, file)
    
    def load_weights_biases(self, filename):
        with open(filename, 'rb') as file:
            weights_biases = pickle.load(file)
            self.XY = weights_biases['XY']
            self.biasY = weights_biases['biasY']
            
    def relu(self,resultadosPuros):
        linhas,colunas = resultadosPuros.shape
        
        resultados = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultadosPuros[i][j] >= 0):
                    resultados[i,j] = resultadosPuros[i][j]
                else:
                    resultados[i,j] = 0
        
        return resultados
    
    def derivadaRelu(self,resultado):
        linhas, colunas = resultado.shape
        resultadoDerivada = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultado[i][j]>=0):
                    resultadoDerivada[i][j] = 1
                else:
                    resultadoDerivada[i][j] = 0
                    
        return resultadoDerivada
    
    def adaptive_learning_rate(self, erro):
        
        deltaErro = erro - self.erro_anterior
        
        if deltaErro < 0.5:
            self.alpha = self.alpha * 1.5  # Reduza a taxa de aprendizagem
        else:
            self.alpha = self.alpha * 0.5  # Aumente a taxa de aprendizagem

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def derivadaSigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def get_accuracy(self,entrada, target):

        camadaY = np.dot(entrada, self.XY) + self.biasY
        Y = np.tanh(camadaY)
                    
        y_pred = np.where(Y > 0, 1, -1)
        

        y_pred = np.where(Y > 0, 1, -1)
        
        accuracy = accuracy_score(target, y_pred)

        # accuracy = np.mean(y_pred == target)
        
        # print(accuracy)
        return accuracy
    
    def accuracy(self):
        accuracy_train = self.get_accuracy(self.entrada, self.target)
        accuracy_test = self.get_accuracy(self.entrada_test, self.target_test)

        return accuracy_train, accuracy_test
    
    def reset(self):
        print('================================')
        print('reinicando os pesos!!!')
        print('================================')
         
        for i in range(self.entradas):
            for j in range(self.saidas):
                self.XY[i][j] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for j in range(self.saidas):
            self.biasY[0][j] = rd.uniform(-self.aleatorio,self.aleatorio)
               
        self.listaCiclo = []
        self.erros = []
        
        self.listaCiclo_test = []
        self.erros_test = []
        
        self.ciclo = 0
        
        self.erroTotal = np.Infinity
        
        self.trained = 0
        
        self.firstErro = 0
    
    def __init__(self, entrada,target,entradas_test,targets_test):
        self.entrada = entrada
        self.target = target
        self.entrada_test = entradas_test
        self.target_test = targets_test
        self.entradas = entrada.shape[1]
        self.saidas = target.shape[1]
        self.XY = np.zeros((self.entradas,self.saidas))
        self.biasY = np.zeros((1,self.saidas))
        self.camadaY = np.zeros((1,self.saidas))
        self.listaCiclo = []
        self.erros = []
        self.listaCiclo_test = []
        self.erros_test = []
        self.firstErro = 0
    
    def train(self):
        if(self.trained==0):
            while self.ciclo<5000 and self.erroTotal>50 and self.accuracy_train<0.95:
                     
                self.camadaY = np.dot(self.entrada, self.XY) + self.biasY
                
                self.Y = np.tanh(self.camadaY)

                erro = np.sqrt(2*np.sum(0.5*(self.target-self.Y)**2))
                erro_test_ = self.test(entradas_test, targets_test)
                y_pred = np.where(self.Y > 0, 1, -1)
                self.accuracy_train = accuracy_score(self.target, y_pred) 
                deltinhaXY = ((self.target-self.Y)/np.sqrt(np.sum((self.target-self.Y)**2)))
                
                deltaXY = self.alpha * np.dot(deltinhaXY.transpose(),self.entrada)
                deltaBiasY = self.alpha * np.sum(deltinhaXY)   
                    
                self.XY = self.XY + deltaXY.transpose()
                self.biasY = self.biasY + deltaBiasY               
                
                self.accuracy_test = self.get_accuracy(self.entrada_test, self.target_test)
                
                self.list_accuracy_train.append(self.accuracy_train)
                self.list_accuracy_test.append(self.accuracy_test)
                
                print('================================')
                print('Ciclo: ' + str(self.ciclo))
                print('Alpha: ' + str(self.alpha))
                print('Treinamento Acuracia: ' + str(self.accuracy_train))
                print('Treinamento LMSE: ' + str(erro))
                print('Teste Acuracia: ' + str(self.accuracy_test))
                print('Teste LMSE: ' + str(erro_test))
                print('================================')
                
                self.listaCiclo.append(self.ciclo)

                self.erros.append(erro)
                    
                self.erros_test.append(erro_test)
                             
                data = {
                    'Epoch': self.listaCiclo,
                    'Error': self.erros,
                    'Test Error': self.erros_test,
                    'Accuracy': self.list_accuracy_train,
                    'Test Accuracy': self.list_accuracy_test
                }
                
                df = pd.DataFrame(data)

                # Plotar os gráficos
                plt.figure(figsize=(12, 6))
            
                plt.subplot(1, 2, 1)
                sns.lineplot(x='Epoch', y='Error', data=df, label='Training Error')
                sns.lineplot(x='Epoch', y='Test Error', data=df, label='Test Error')
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title('Training and Test Error over Epochs')
            
                plt.subplot(1, 2, 2)
                sns.lineplot(x='Epoch', y='Accuracy', data=df, label='Training Accuracy')
                sns.lineplot(x='Epoch', y='Test Accuracy', data=df, label='Test Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Training and Test Accuracy over Epochs')
            
                plt.tight_layout()
                plt.show()
                
                self.ciclo = self.ciclo + 1
                
            self.trained = 1
        
        else:
            print('================================')
            print('Modelo ja treinado')
            print('================================')   
        
    
    def test(self, entrada,target):
              
        camadaY = np.dot(entrada, self.XY) + self.biasY
        
        Y = np.tanh(camadaY)
            
        erro = np.sqrt(2*np.sum(0.5*(target-Y)**2))
        
        Y = np.where(Y > 0, 1, -1)
             
        return erro,Y

#%%
                     
model = Perceptron(entradas_train, targets_train, entradas_test, targets_test)
model.reset()
model.train()

#%%
model.save_weights_biases('pesosMadaline/pesos.pkl')

#%%

model2 = Perceptron(entradas_train, targets_train, entradas_test, targets_test)

model2.load_weights_biases('pesosMadaline/pesos.pkl')

precisao_train = model2.get_accuracy(entradas_train, targets_train)
erro_train = model2.test(entradas_train, targets_train)

print('==========================================================')
print('Acuracia Treinamento: ' + str(precisao_train))
print('Erro Treinamento: ' + str(erro_train))
print('==========================================================')

precisao_test = model2.get_accuracy(entradas_test, targets_test)
erro_test = model2.test(entradas_test, targets_test)

print('==========================================================')
print('Acuracia Teste: ' + str(precisao_test))
print('Erro Teste: ' + str(erro_test))
print('==========================================================')

# precisao_validation = model2.get_accuracy(entradas_validation, targets_validation)
# erro_validation = model2.test(entradas_validation, targets_validation)

# print('==========================================================')
# print('Acuracia Validacao: ' + str(precisao_validation))
# print('Erro Validacao: ' + str(erro_validation))
# print('==========================================================')





