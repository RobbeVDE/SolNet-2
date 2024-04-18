import torch


def _moving_window(tensor, timesteps, prediction_length, length):
    """
    Create a moving window based on the steps that we want to forecast and the length of the forecast period
    :param tensor: the flat tensor
    :param timesteps: the N timesteps that we want to move forward before forecasting a new period
    :param prediction_length: the length of the period to forecast
    :return: a tensor with dimensions taking the moving window into account
    """
    #length = int((len(tensor) - timesteps) / prediction_length) + 1
    moving_window = torch.zeros(length, timesteps, 1)

    for i in range(length):
        moving_window[i, :, 0] = tensor[i * prediction_length:timesteps + i * prediction_length].flatten()

    return moving_window


def _scale(train, test=None, domain_min=None, domain_max=None):
    """
    MinMax scaling, fitting and transforming the train set, transforming the test set (with the train set fit)
    :param train: the train tensor
    :param test: the test tensor
    :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
    :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
    :return: returns the scaled train and test sets
    """
    minimum = domain_min if domain_min is not None else train.min()
    maximum = domain_max if domain_max is not None else train.max()

    denominator = maximum - minimum or 1e-8

    train = (train - minimum) / denominator
    test = (test - minimum) / denominator if test is not None else None

    return train, test


class Tensorisation:

    def __init__(
            self,
            data,
            target: str,
            features: list,
            lags: int,
            forecast_period: int,
            train_test_split=0.8,
            domain_min=None,
            domain_max=None):
        """
        create tensors for use in pytorch, based on the data list we get from the datafetcher.py script.
        :param data: the data (list of dataframes)
        :param target: the target variable name
        :param features: A list of feature names
        :param lags: The number of lags to include (the input length)
        :param forecast_period: the number of hours to forecast
        :param train_test_split: the train test split as a float (0.8 means 80% train data and 20% test data)
        :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
        :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
        """
        self.data = data
        self.target = target
        self.features = features
        self.lags = lags
        self.forecast_period = forecast_period
        self.train_test_split = train_test_split
        self.domain_min = domain_min
        self.domain_max = domain_max

    def tensor_creation(self, WFE=False, window_days=30):
        """
        The method doing the tensor creation
        :return: tensors, split in a train and test set, with features (X) and targets (y)
        """
        prediction_len = len(self.data) - self.lags  # See how much data is used for predictions

        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows = int(
            prediction_len / self.forecast_period)  # Get the number of predictions we can make.

        train_len = round(windows * self.train_test_split)  # Split the features into a train set...
        test_len = windows - train_len  # ... and a test set

        X_train = torch.zeros(train_len, self.lags,
                              len(self.features))  # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags,
                             len(self.features))  # Create the empty testing tensor for the features

        flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):
            X_tensor = torch.tensor(self.data[feature]).type(torch.float32)
                  
            X_train_feature = X_tensor[self.lags:flat_train_len+self.lags]  # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len+self.lags:]  # ... and a test set

            # Use the scaling method to get everything between 0 and 1     
            train, test = _scale(X_train_feature,
                                X_test_feature,
                                domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)

            # Use the moving window to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:, :, i] = _moving_window(train,
                                            self.lags,
                                            self.forecast_period, train_len).squeeze(-1)
            X_test[:, :, i] = _moving_window(test,
                                            self.lags,
                                            self.forecast_period, test_len).squeeze(-1)
            
            

            # Make the target vector if the feature is our target
        y_tensor = torch.tensor(self.data['P']).type(torch.float32)
        y_tensor = y_tensor[self.lags:]
        y_train, y_test = _scale(y_tensor[:flat_train_len],
                                    y_tensor[flat_train_len:],
                                     domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)

        y_train = y_train.view(train_len, self.forecast_period, 1)
        y_test = y_test.view(test_len, self.forecast_period, 1)


        if WFE:
            X = torch.concat([X_train, X_test])
            y = torch.concat([y_train, y_test])
            iterations = int(X.size(0)/window_days)
            X_train =[]
            X_test = []
            y_train = []
            y_test = []
            for i in range(iterations):
                y_test.append(y[i*window_days:(i+1)*window_days, :,:])
                X_test.append(X[i*window_days:(i+1)*window_days, :,:])
                if iterations <= 12:
                    X_train.append(X[:(i+1)*window_days, :,:])
                    y_train.append(y[:(i+1)*window_days, :,:])
                else:
                    # After 12 iterations, expanding window goes to rolling window to avoid having over exessive data
                    X_train.append(X[(i-12)*window_days:(i+1)*window_days, :,:])
                    y_train.append(y[(i-12)*window_days:(i+1)*window_days, :,:])
            
            #Pad with zeroes last test month to have consistent tensor sizes
            
            y_final = torch.zeros(window_days, y.size(1), y.size(2))
            n_days_left = X.size(0)%window_days
            y_final[:n_days_left,:,:] = y[iterations*window_days:,:,:]
            X_final = torch.zeros(window_days, X.size(1), X.size(2))
            X_final[:n_days_left,:,:] = X[iterations*window_days:,:,:]
            y_test.append(y_final)
            X_test.append(X_final)

        return X_train, X_test, y_train, y_test

    def evaluation_tensor_creation(self):
        """
        A similar method that takes into account a separate evaluation set for the purpose of transfer learning
        :param evaluation_length: the length of the evaluation set
        :return: tensors, split in a train, test and evaluation set, with features (X) and targets (y)
        """
        prediction_len = len(self.data) - self.lags  # See how much data is used for predictions

        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows = int(prediction_len / self.forecast_period)  # Get the number of predictions we can make.

        try:
            self.features.remove(self.target)
        except:
            pass

        X_eval = torch.zeros(windows, self.lags, len(self.features))

        flat_train_len = (windows * self.forecast_period) + self.lags - self.forecast_period
        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):
            X_tensor = torch.tensor(self.data[feature]).type(torch.float32)
            X_tensor = X_tensor[self.lags:]
            # Use the scaling method to get everything between 0 and 1     
            eval, _ = _scale(X_tensor,
                                 domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)

            # Use the moving window to go from the flat tensor to the correct dimensions (window, lags per window)
            X_eval[:, :, i] = _moving_window(eval,
                                            self.lags,
                                            self.forecast_period, windows).squeeze(-1)
            
            

        y_tensor = torch.tensor(self.data['P']).type(torch.float32)
        y_tensor = y_tensor[self.lags:]   
        y_eval, _ = _scale(y_tensor,
                                    domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)    
        y_eval = y_eval.view(windows, self.forecast_period, 1)
        # Make the target vector if the feature is our target
       
        return X_eval, y_eval
    
class NWP_tensorisation:
    def __init__(
        self,
        input_data,
        target_data,
        features: list,
        lags: int,
        forecast_period: int,
        train_test_split=0.8,
        domain_min=None,
        domain_max=None):
        """
        create tensors for use in pytorch, based on the data list we get from the datafetcher.py script.
        :param data: the data (list of dataframes)
        :param target: the target variable name
        :param features: A list of feature names
        :param lags: The number of lags to include (the input length)
        :param forecast_period: the number of hours to forecast
        :param train_test_split: the train test split as a float (0.8 means 80% train data and 20% test data)
        :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
        :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
        """
        self.input_data = input_data
        self.target_data = target_data
        self.features = features
        self.lags = lags
        self.forecast_period = forecast_period
        self.train_test_split = train_test_split
        self.domain_min = domain_min
        self.domain_max = domain_max

    def tensor_creation(self):
        """
        The method doing the tensor creation
        :return: tensors, split in a train and test set, with features (X) and targets (y)
        """
        prediction_len = len(self.input_data) - self.lags  # See how much data is used for predictions

        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows = int(
            prediction_len / self.forecast_period)  # Get the number of predictions we can make.

        train_len = round(windows * self.train_test_split)  # Split the features into a train set...
        test_len = windows - train_len  # ... and a test set

        X_train = torch.zeros(train_len, self.lags,
                                len(self.features))  # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags,
                                len(self.features))  # Create the empty testing tensor for the features
        
        Y_train = torch.zeros(train_len, self.lags,
                                len(self.features))  # Create the empty training tensor for the features
        Y_test = torch.zeros(test_len, self.lags,
                                len(self.features))  # Create the empty testing tensor for the features

        flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):
            X_tensor = torch.tensor(self.input_data.loc[:,feature].values).type(torch.float32)
            Y_tensor = torch.tensor(self.target_data.loc[:,feature].values).type(torch.float32)
                    
            X_train_feature = X_tensor[self.lags:flat_train_len+self.lags]  # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len+self.lags:]  # ... and a test set

            Y_train_feature = Y_tensor[self.lags:flat_train_len+self.lags]  # Split the flattened dataframe in a train set...
            Y_test_feature = Y_tensor[flat_train_len+self.lags:]  # ... and a test set

            # Use the scaling method to get everything between 0 and 1     
            train_input, test_input = _scale(X_train_feature,
                                X_test_feature,
                                domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)
            
            train_target, test_target = _scale(Y_train_feature,
                                Y_test_feature,
                                domain_min=self.domain_min[feature] if isinstance(self.domain_max, dict) else None,
                                domain_max=self.domain_max[feature] if isinstance(self.domain_max, dict) else None)

            # Use the moving window to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:, :, i] = _moving_window(train_input,
                                            self.lags,
                                            self.forecast_period, train_len).squeeze(-1)
            X_test[:, :, i] = _moving_window(test_input,
                                            self.lags,
                                            self.forecast_period, test_len).squeeze(-1)
            
            Y_train[:, :, i] = _moving_window(train_target,
                                            self.lags,
                                            self.forecast_period, train_len).squeeze(-1)
            Y_test[:, :, i] = _moving_window(test_target,
                                            self.lags,
                                            self.forecast_period, test_len).squeeze(-1)
            
            

        return X_train, X_test, Y_train, Y_test