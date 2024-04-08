import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
cross_validate = False
ES_option = False # Option to use early stopping
import optuna
import torch.optim as optim

torch.manual_seed(0)
def save_model(model, name):
    """
    Saves the state dictionary using torch
    :param name: name of the file
    :param model: the model for which we want to save the state dictionary
    """
    torch.save(model.state_dict(), '../models/' + str(name))


class Training:

    def __init__(
            self,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            optimizer_name,
            trial=None,
            batch_size=32,
            learning_rate=0.001,
            criterion=torch.nn.MSELoss()
            ):
        """
        The training class for the pytorch model
        :param model: The model that we train
        :param X_train: the tensor with training values for X
        :param y_train: the tensor with training values for y
        :param X_test: the tensor with test values for X
        :param y_test: the tensor with test values for y
        :param epochs: the number of epochs that we wish to train for
        :param batch_size: the batch size before going through backpropagation
        :param learning_rate: the learning rate
        :param criterion: the criterion by which to evaluate the performance (i.e. the loss function)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))

        days = y_train.shape[0] + y_test.shape[0]
        self.months = round(days / 30.5)
        self.total_data = ConcatDataset([train_data, test_data])
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        self.trial = trial
        self.model = model
        self.criterion = criterion
        self.optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        

    def fit(self):
        """
        The training loop itself
        :return: state_dict_list: the state dictionary for each of the epochs, argmin_test: the best epoch
        """
        avg_train_error = []
        avg_test_error = []
        state_dict_list = []
            
        early_stopper = EarlyStopper(patience=5, min_delta=0.005)
        for epoch in range(self.epochs):
            num_train_batches = 0
            num_test_batches = 0
            total_loss = 0
            total_loss_test = 0
            batches = iter(self.train_loader)
            self.model.train()

            for input, output in batches:
                prediction = self.model(input)
                output = output.squeeze()
                loss = self.criterion(prediction, output)
                total_loss += float(loss)
                num_train_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()

            with torch.inference_mode():
                test_batches = iter(self.test_loader)

                for input, output in test_batches:
                    prediction = self.model(input)
                    output = output.squeeze()
                    test_loss = self.criterion(prediction, output)

                    total_loss_test += float(test_loss)
                    num_test_batches += 1

            avg_train_error.append(total_loss / num_train_batches)
            avg_test_error.append(total_loss_test / num_test_batches)
            if early_stopper.early_stop(avg_test_error[-1]) and ES_option:
                break
            
            state_dict_list.append(self.model.state_dict())
            if self.trial is not None:
                self.trial.report(avg_train_error[-1], epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            if epoch % 5 == 0:
                print('Step {}: Average train loss: {:.4f} | Average test loss: {:.4f}'.format(epoch, avg_train_error[epoch],
                                                                                            avg_test_error[epoch]))

        argmin_test = avg_test_error.index(min(avg_test_error))

        print('Best Epoch: ' + str(argmin_test))

        plt.plot(avg_train_error, label='train error ' + str(self.months) + ' months')
        plt.plot(avg_test_error, label='test error ' + str(self.months) + ' months')
        plt.legend()

        return min(avg_test_error), state_dict_list[argmin_test]

    def fit_cv(self):
        fold_avg_test_loss = []
        best_loss = np.inf
        kfold = KFold()
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.total_data)):
            avg_train_error = []
            avg_test_error = []
            state_dict_list = []
            print(f'FOLD {fold}')
            print('-----------------------')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(self.total_data, 
                                                        batch_size=self.batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self.total_data, 
                                                        batch_size=self.batch_size, sampler=test_subsampler)
            for epoch in range(self.epochs):
                num_train_batches = 0
                num_test_batches = 0
                total_loss = 0
                total_loss_test = 0
                batches = iter(trainloader)
                self.model.train()

                for input, output in batches:
                    prediction = self.model(input)
                    output = output.squeeze()
                    loss = self.criterion(prediction, output)
                    total_loss += float(loss)
                    num_train_batches += 1

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()

                with torch.inference_mode():

                    test_batches = iter(testloader)

                    for input, output in test_batches:
                        prediction = self.model(input)
                        output = output.squeeze()
                        test_loss = self.criterion(prediction, output)

                        total_loss_test += float(test_loss)
                        num_test_batches += 1

                    avg_train_error.append(total_loss / num_train_batches)
                    avg_test_error.append(total_loss_test / num_test_batches)
                    state_dict_list.append(self.model.state_dict())
                    if self.trial is not None:
                        self.trial.report(avg_test_error[-1], epoch)   
                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    if epoch % 5 == 0:
                        print('Step {}: Average train loss: {:.4f} | Average test loss: {:.4f}'.format(epoch, avg_train_error[epoch],
                                                                                                    avg_test_error[epoch]))
            
            fold_argmin_test = avg_test_error.index(min(avg_test_error))
            fold_avg_test_loss.append(avg_test_error[fold_argmin_test])

            if fold_avg_test_loss[-1] < best_loss:
                best_state_dict = state_dict_list[fold_argmin_test]
                best_loss = fold_avg_test_loss[-1]
            

            ### RESET WEIGHTS
            for name, module in self.model.named_children():
                print('resetting ', name)
                module.reset_parameters()

            print(f'Best Epoch for fold{fold}: ' + str(fold_argmin_test))

            plt.plot(avg_train_error, label='train error ' + str(self.months) + ' months')
            plt.plot(avg_test_error, label='test error ' + str(self.months) + ' months')
            plt.legend()
        
        
        avg_error = np.mean(fold_avg_test_loss)
        print(f"Average error:{avg_error}")
        return avg_error, best_state_dict


class EarlyStopper:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
