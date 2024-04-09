import numpy as np
import pickle
import pandas as pd
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

def data_slicer(data, date_range):
    """
    Take a slice of the data which belongs to the desired date_range
    """
    data = data[data.index.isin(date_range)]

    return data
def merge_slice(range, *args):

    merged = pd.DataFrame(index = range)
    for arg in args:
        merged = pd.merge(merged, data_slicer(arg, range), right_index=True, left_index=True, how='inner')
    if ~isinstance(merged.index, pd.DatetimeIndex):
        merged.index = pd.to_datetime(merged.index, utc=True)
    return merged
    
def target_renamer(dataset, original_name):
    #Rename column with target to 'P' to simplify rest of code
    dataset = dataset.rename(columns ={original_name:'P'})
    return dataset            

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
    
    def remove_outliers(self, GHI_name = 'downward_surface_SW_flux', tolerance = 50, outlier_list = None): 
        """"
        Remove data entries where the power of PV is 0 but GHI is higher than a specified tolerance
        Take a 50 default tolerance, bcs there seem to be no power produced at beginning of day when irrad really low
        but want machine to learn this bcs this is not an outlier
        """
        if outlier_list is None:
            outlier_list = [True] * len(self.data)
        for i in range(len(self.data)):
            if outlier_list[i]:
                dataset = self.data[i]
                mask = (dataset['P'] == 0) * (dataset[GHI_name] > tolerance)
                indices = dataset[mask].index
                dates = list(indices.date)
                dataset['date'] = dataset.index.date
                dataset = dataset[~dataset.date.isin(dates)]
                self.data[i] = dataset.drop('date', axis=1)

        return self.data
    
    def inverter_limit(self, inverter_rating, inv_list):
        """
        Add inverter rating which is not possible for PVGIS
        """
        for i in range(len(self.data)):
            if inv_list[i]:
                dataset = self.data[i]['P']
                dataset[dataset>inverter_rating] = inverter_rating
                self.data[i]['P'] = dataset
        return self.data
    

    
def data_handeler(source=None, target=None, eval=None, transform = True, month_source=False):
    """
    Add explation, maybe more general power function if we use more test setups
    RETURNS source data, target_data, eval_data
    """
    #Month ranges, maybe option to specify this with function
    if month_source:   
        source_range = pd.date_range("2019-08-01", "2019-08-31 23:00", freq='h', tz="UTC")
    else:
        source_range = pd.date_range("2016-05-01","2020-07-31 23:00", freq='h', tz="UTC")

    target_range = pd.date_range("2020-08-01", "2020-08-31 23:00", freq='h', tz="UTC")
    eval_range = pd.date_range("2020-09-01", "2021-07-31 23:00", tz="UTC", freq='h')

    #Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        print("In Google Colab environment: Using .csv files")
    except:
        IN_COLAB = False
        print("Not in Colab environment: Using .pickle files")
    
    
    #Data intakeer
    if IN_COLAB:
        openmeteo = pd.read_csv("Data/openmeteo.csv", parse_dates=True)
        # openmeteo.index = pd.to_datetime(openmeteo.index)
        pvgis = pd.read_csv('Data/PVGIS.csv', parse_dates=True)
        # pvgis.index = pd.to_datetime(pvgis.index, parse_dates=True)
        ceda = pd.read_csv("CEDA_dataNL.csv", parse_dates=True)
        # ceda.index = pd.to_datetime(ceda.index, parse_dates=True)
        is_day = pd.read_csv("Data/is_day.csv", parse_dates=True)
        # is_day.index = pd.to_datetime(is_day.index, parse_dates=True)
    else:
        openmeteo = pd.read_pickle("Data/openmeteo.pickle")

        pvgis = pd.read_pickle('Data/PVGIS.pickle')

        ceda = pd.read_pickle("CEDA_dataNL.pickle")
        is_day = pd.read_pickle("Data/is_day.pickle")

    meteo2CEDA = {'temperature_2m' :'temperature_1_5m', 
                "relative_humidity_2m":"relative_humidity_1_5m", 
                "pressure_msl": "pressure_MSL",
                "cloud_cover":"total_cloud_amount",
                "shortwave_radiation": "downward_surface_SW_flux",
                "diffuse_radiation":"diffuse_surface_SW_flux",
                "direct_normal_irradiance":"direct_surface_SW_flux",
                "wind_speed_10m": "wind_speed_10m",
                "wind_direction_10m": "wind_direction_10m"
                }
    openmeteo = openmeteo.rename(columns=meteo2CEDA)

    #NL production data
    installation_id = "3437BD60"
    prodNL = pd.read_parquet('Data/production.parquet', engine='pyarrow')
    metadata = pd.read_csv("Data/installations Netherlands.csv", sep=';')
    metadata = metadata.set_index('id')
    metadata_id = metadata.loc[installation_id]
    tilt = metadata_id["Tilt"]
    peakPower = metadata_id["Watt Peak"]
    azimuth = metadata_id["Orientation"]
    latitude = metadata_id["Latitude"]
    longitude = metadata_id["Longitude"]
    power = prodNL.loc[installation_id]
    power = target_renamer(power, 'watt')
    power = power.resample('h').sum()/4
    power = power.tz_localize('UTC')

    #Merge right data to have source, target and eval dataset
    list = [eval, target, source]
    power_list = [power, power, pvgis]
    range_list = [eval_range, target_range,source_range]
    data = []
    for i, entry in enumerate(list):
        if entry == "nwp":
            covariates = ceda
        elif entry == "era5":
            covariates = openmeteo
        elif entry is None:
            break
        else:
            raise KeyError
        
        if transform:
            data.append(merge_slice(range_list[i], power_list[i], covariates, is_day))
        else:
            data.append(merge_slice(range_list[i], power_list[i], covariates))

    #Add extra variates
    data = Featurisation(data)
    data.data = data.cyclic_features()
    data.data = data.cyclic_angle('wind_direction_10m')
    if transform:
        data.data = data.PoA(latitude, longitude, tilt, azimuth)
        outlier_list = [False, True, True] #No outliers removed for evaluation as this is not 
        inv_list = [False, False, True] #Pre-processing only allowed for 
        data.data = data.remove_outliers(tolerance=50, outlier_list=outlier_list)
        data.data = data.inverter_limit(2500, inv_list)
    data.data = data.add_shift('P') #Shift after inv_limit, otherwise discrepancy


    if source is not None:
        source_dataset = data.data[2]
    if target is not None:
        target_dataset = data.data[1]
    if eval is not None:
        eval_dataset= data.data[0]
    
    return source_dataset, target_dataset, eval_dataset 


      

