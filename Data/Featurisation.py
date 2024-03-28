import numpy as np
import pickle
import pandas
from pvlib import location
from pvlib import irradiance

def _load_data(file_path):
    """
    load data from a given file path
    :param file_path: the file path name
    :return: return the data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: {} doesn't exist.".format(file_path))

    return data


class Featurisation:

    def __init__(self, data):
        """
        Include features for the training of the models
        :param data: the original dataframe (in list format as provided by datafetcher.py or as a file name)
        """
        if data is None:
            raise ValueError("Data cannot be None. Please provide a (list of) pandas dataframe(s) or a file path.")
        elif isinstance(data, list):
            self.data = data
        elif isinstance(data, str):
            self.data = _load_data(data)
        else:
            raise ValueError("Invalid data type provided. Must be a (list of) pandas dataframe(s) or a file path.")

    def base_features(self, featurelist):
        """
        Features to include that are already provided by PVGIS
        :param featurelist: list of features that we want to include in the model
        :return: the data list but with only the features provided in the featurelist included
        """
        for i in range(len(self.data)):
            self.data[i] = self.data[i][featurelist]

        return self.data

    def cyclic_features(self, yearly=True, daily=True):
        """
        Cyclical features to include in the model
        :param yearly: yearly cyclical features, transforming the months of the year in sin and cos features
        :param daily: daily cyclical features, transforming the hours of the day in sin and cos features
        :return: the data list but with chosen cyclic features included
        """
        for i in range(len(self.data)):
            if daily is True:
                self.data[i]['hour_sin'] = np.sin(2 * np.pi * self.data[i].index.hour / 24)
                self.data[i]['hour_cos'] = np.cos(2 * np.pi * self.data[i].index.hour / 24)
            if yearly is True:
                self.data[i]['month_sin'] = np.sin(2 * np.pi * self.data[i].index.month / 12)
                self.data[i]['month_cos'] = np.cos(2 * np.pi * self.data[i].index.month / 12)

        return self.data
    
    def add_shift(self, feature, period=24, fill_value=0.13):
        # shift of solar power to get these lags, we use 0.13 as fill value bcs we will first normalise these values and this is sort of cf
        for i in range(len(self.data)):        
            self.data[i][f'{feature}_24h_shift'] = self.data[i][feature].shift(periods=period,fill_value=fill_value)  # Use yearmy average as fill value
            
        return self.data
    
    def cyclic_angle(self, feature):
        """
        Take the cosine and sine of an angle to better represent cyclic behaviour
        """
        for i in range(len(self.data)):
            radians = np.deg2rad(self.data[i][feature])
            self.data[i][f'{feature}_cos'] = np.cos(radians)
            self.data[i][f'{feature}_sin'] = np.sin(radians)
        return self.data
    

    def PoA(self, latitude, longitude, tilt, azimuth,
            GHI_name = "downward_surface_SW_flux",
            DNI_name = "direct_surface_SW_flux",
            DHI_name = "diffuse_surface_SW_flux"):
        
        azimuth = azimuth+180 #PVGIS works in [-180,180] and pvlib in [0,360]

        site = location.Location(latitude, longitude,  tz='UTC')
        for i in range(len(self.data)):

            times = self.data[i].index
            solar_position = site.get_solarposition(times=times)
            ghi = self.data[i][GHI_name]
            dhi = self.data[i][DHI_name]
            dni = self.data[i][DNI_name]
            dni_extra = irradiance.get_extra_radiation(times)
            POA_irrad = irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth, 
                                                        dni=dni, ghi=ghi, dhi=dhi, solar_zenith=solar_position['apparent_zenith'],
                                                        dni_extra=dni_extra, model='perez',solar_azimuth=solar_position['azimuth'])
            self.data[i]['PoA'] = POA_irrad['poa_global'].fillna(0)

        return self.data
    
    def remove_outliers(self, GHI_name = 'downward_surface_SW_flux', tolerance = 10): 
        """"
        Remove data entries where the power of PV is 0 but GHI is higher than a specified tolerance
        """
        for i in range(len(self.data)):

            mask = (self.data[i]['P'] == 0) * (self.data[i][GHI_name] > 0)
            self.data[i] = self.data[i][~mask]

        return self.data
