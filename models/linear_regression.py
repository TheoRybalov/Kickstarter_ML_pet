import pandas as pd
import numpy as np

class NormalEquation():

    def __init__(self, samples = pd.DataFrame, targets = pd.DataFrame, fit_intercept: bool = True, copy: bool = True):
        self.X = samples.copy() if copy else samples
        self.Y = targets

        if fit_intercept:
                ones = np.ones((self.X.shape[0], 1))
                self.X = np.hstack((self.X, ones))
    
    def solve(self):
        XTX = np.dot(self.X.T, self.X)
        XTX_inv = np.linalg.inv(XTX)
        XTX_inv_XT = np.dot(XTX_inv, self.X.T)
        final_betas = np.dot(XTX_inv_XT, self.Y)

        return final_betas


class GradientDescentMse:

    def __init__(self, samples = pd.DataFrame, targets = pd.DataFrame,
                 learning_rate: float = 1e-3, threshold: float = 1e-5, copy: bool = True):
        
        """
        self.samples - матрица признаков
        self.targets - вектор таргетов
        self.beta - вектор из изначальными весами модели == коэффициентами бета (состоит из единиц)
        self.learning_rate - параметр *learning_rate* для корректировки нормы градиента
        self.threshold - величина, меньше которой изменение в loss-функции означает остановку градиентного спуска
        iteration_loss_dict - словарь, который будет хранить номер итерации и соответствующую MSE
        copy: копирование матрицы признаков или создание изменения in-place
        """

        self.samples = samples.copy() if copy else samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.betas = np.ones(self.samples.shape[1])
        self.iteration_loss_dict = {}
    
    def add_constant_feature(self):

        """
        Метод для создания константной фичи в матрице объектов samples
        Метод создает колонку с константным признаком (interсept) в матрице признаков.
        Hint: так как количество признаков увеличилось на одну, не забудьте дополнить вектор с изначальными весами модели!
        """
        self.samples['intercept'] = 1
        self.betas = np.append(self.betas, 1)
    
    def calculate_mse_loss(self):

        predictions = np.dot(self.samples, self.betas)
        residuals = predictions - self.targets.values
        MSE = ((residuals ** 2)).mean()

        return MSE
    
    def calculate_gradient(self):

        gradient = np.zeros(self.samples.shape[1])

        predictions = np.dot(self.samples, self.betas)
        residuals = (predictions - self.targets).values

        for j in range(self.samples.shape[1]):
            d_ij = self.samples.iloc[:, j]
            gradient[j] = 2 * (residuals * d_ij).mean()
        
        return gradient
    

    def iteration(self):
        """
        Обновляем веса модели в соответствии с текущим вектором-градиентом
        """
        gradient = self.calculate_gradient()
        self.betas = self.betas - self.learning_rate * gradient
    
    def learn(self):

        """
        Итеративное обучение весов модели до срабатывания критерия останова
        Запись mse и номера итерации в iteration_loss_dict
        Описание алгоритма работы для изменения функции потерь:
            Фиксируем текущие mse -> previous_mse
            Делаем шаг градиентного спуска
            Записываем новые mse -> next_mse
            Пока |(previous_mse) - (next_mse)| > threshold:
                Повторяем первые 3 шага
        """

        iteration = 0
        previous_MSE = self.calculate_mse_loss()

        while True:
            self.iteration()
            current_MSE = self.calculate_mse_loss()

            self.iteration_loss_dict[iteration] = current_MSE

            if abs(previous_MSE - current_MSE) < self.threshold:
                break

            previous_MSE = current_MSE
            iteration += 1

    
