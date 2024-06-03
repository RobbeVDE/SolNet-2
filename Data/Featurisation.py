import numpy as np
import pickle
import pandas as pd
from pvlib import location, irradiance, temperature, pvsystem
import matplotlib.pyplot as plt

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
        merged = pd.merge(merged, data_slicer(arg, range), right_index=True, left_index=True, how='outer')
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
            # self.data[i].drop(columns = feature, inplace=True)
        return self.data
    

    def PoA(self, latitude, longitude, tilt, azimuth,
            GHI_name = "downward_surface_SW_flux",
            DNI_name = "direct_surface_SW_flux",
            DHI_name = "diffuse_surface_SW_flux"):
        
        azimuth = azimuth+180 #PVGIS works in [-180,180] and pvlib in [0,360]

        site = location.Location(latitude, longitude, altitude=location.lookup_altitude(latitude, longitude),  tz='UTC')
        for i in range(len(self.data)):

            times = self.data[i].index
            solar_position = site.get_solarposition(times=times)
            ghi = self.data[i][GHI_name]
            pres = self.data[i]['pressure_MSL']
            dni_extra = irradiance.get_extra_radiation(times)
            try:
                dhi = self.data[i][DHI_name]
                dni = self.data[i][DNI_name]
            except KeyError: #Meaning DNI and DHI are not yet present, ie UKV:Global add them also to dataframe
                out_erbs = irradiance.erbs(ghi, solar_position.zenith, times)
                dhi = out_erbs['dhi']
                dni = out_erbs['dni']
                self.data[i][DHI_name] = dhi
                self.data[i][DNI_name] = dni
            
            POA_irrad = irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth, 
                                                        dni=dni, ghi=ghi, dhi=dhi, solar_zenith=solar_position['apparent_zenith'],
                                                        dni_extra=dni_extra, model='perez',solar_azimuth=solar_position['azimuth'])
            self.data[i]['PoA'] = POA_irrad['poa_global'].fillna(0)
            


        return self.data

    def decomposition(self, lat, lon,
                      DNI_name = "direct_surface_SW_flux", 
                      DHI_name = "diffuse_surface_SW_flux"):
        site = location.Location(lat, lon, altitude=location.lookup_altitude(lat, lon),  tz='UTC')
        for i in range(len(self.data)):        
            times = self.data[i].index
            ghi = self.data[i]["downward_surface_SW_flux"]
            pres = self.data[i]["pressure_MSL"] *100
            solar_position = site.get_solarposition(times=times)
            dni_dirint = irradiance.disc(ghi, solar_position.zenith, times, pres)
            df_dirint = irradiance.complete_irradiance(solar_position.apparent_zenith, ghi, dni=dni_dirint.dni, dhi=None)
            self.data[i][DHI_name] = df_dirint.dhi
            self.data[i][DNI_name] = df_dirint.dni
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

    def clearsky_power(self, lat, lon, peakPower, tilt, azimuth):
        #small praemters to have conservative maximum
        azimuth = azimuth +180
        loss_inv = 0.99
        temp_coeff = -0.002
        site = location.Location(lat, lon, altitude=location.lookup_altitude(lat, lon),tz='UTC')

        for i in range(len(self.data)):

            #1. Get clear sky irradiance for location
            times = self.data[i].index
            cs = site.get_clearsky(times)
            

            temp = self.data[i]['temperature_1_5m'] -275.13
            wind_speed = self.data[i]['wind_speed_10m']
            wind_height = 10
            solar_position = site.get_solarposition(times=times)
            ghi = cs["ghi"]
            dhi = cs["dhi"]
            dni = cs["dni"]
            dni_extra = irradiance.get_extra_radiation(times)
            POA_irrad = irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth, 
                                                        dni=dni, ghi=ghi, dhi=dhi, solar_zenith=solar_position['apparent_zenith'],
                                                        dni_extra=dni_extra, model='perez',solar_azimuth=solar_position['azimuth'])
            
            poa = POA_irrad['poa_global'].fillna(0)
            csi = irradiance.clearsky_index(self.data[i]["downward_surface_SW_flux"], ghi)    

            temp_cell = temperature.fuentes(poa, temp, wind_speed, 49, wind_height=wind_height,
                                                surface_tilt=tilt)

            inv_params = {'pdc0': peakPower, 'eta_inv_nom': loss_inv}
            module_params = {'pdc0': peakPower, 'gamma_pdc': temp_coeff}
            mount = pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
            array = pvsystem.Array(mount=mount, module_parameters = module_params)
            pvsys = pvsystem.PVSystem(arrays=[array], inverter_parameters=inv_params)
            dc_power = pvsys.pvwatts_dc(poa, temp_cell)

            cs_power =  pvsys.get_ac('pvwatts', dc_power)
            self.data[i]["CS_power"] = cs_power
            self.data[i]["T_PV"] = temp_cell
            self.data[i]["csi"] = csi

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
    
    def deseasonalise(self, lat, lon):
        for i in range(len(self.data)):
            power = self.data[i]['P']
            power_24h = self.data[i]['P_24h_shift']
            cs_power = self.data[i]['CS_power']

            times = self.data[i].index
            site= location.Location(lat, lon, altitude=location.lookup_altitude(lat,lon), tz='UTC')
            sol_pos = site.get_solarposition(times)     
            mask = sol_pos['zenith'] < 80

            power[~mask] = 0.0
            power[mask] = power[mask]/cs_power[mask]
            power_24h[~mask] = 0.0
            power_24h[mask] = power_24h[mask]/cs_power[mask]

        return self.data
    
      
    
def data_handeler(installation_int = 0, source=None, target=None, eval=None, transform = True, month_source=False, HP_tuning = True, start_month=None):
    """
    Add explation, maybe more general power function if we use more test setups
    RETURNS source data, target_data, eval_data
    """
    #Metadata
    metadata = pd.read_pickle("Data/Sites/metadata.pkl")
    metadata = metadata.loc[installation_int]
    peakPower = metadata['Installed Power']
    tilt = metadata['Tilt']
    azimuth = metadata['Azimuth']
    lat = metadata['Latitude']
    lon = metadata['Longitude']
    inv_limit = metadata['Inverter Power']
    start = metadata['Start']
    end = metadata['End']

    #Month ranges, maybe option to specify this with function
    if month_source:   
        source_range = pd.date_range("2018-08-01", "2018-08-31 23:00", freq='h', tz="UTC")
    else:
        if start_month is None:
            if HP_tuning:
                source_range = pd.date_range("2016-05-01","2019-04-30 23:00", freq='h', tz="UTC")
                target_range = pd.date_range("2019-05-01", "2020-04-30 23:00", freq='h', tz="UTC")
            else:
                if installation_int == 3:
                    source_range = pd.date_range("2016-05-01", start.tz_localize(None) - pd.Timedelta('1h'), freq='h', tz="UTC")
                    target_range = pd.date_range("2019-05-01", "2020-04-30 23:00", freq='h', tz="UTC") #Not used for source training so dont care
                else:
                    source_range = pd.date_range("2017-05-01", start.tz_localize(None) - pd.Timedelta('1h'), freq='h', tz="UTC")
                    target_range = pd.date_range("2019-05-01", "2020-04-30 23:00", freq='h', tz="UTC") #Not used for source training so dont care
            eval_range = pd.date_range(start, end, tz="UTC", freq='h')
        else:
            if start_month <= 2:
                start_year=2020
            else:
                start_year=2019
            
        
            start_eval = pd.to_datetime(f"{start_year}-{start_month}-01")
            end_eval = pd.to_datetime(f"{start_year+1}-{start_month}-01") - pd.Timedelta("1h")

           
            end_source = start_eval.tz_localize(None) - pd.Timedelta('1h')
            end_year = end_source.year
            if start_month == 1:
                start_year = end_year-2
            else:
                start_year = end_year-3
            start_source = pd.to_datetime(f'{start_year}-{start_month}-01')
            print(start_source, end_source)
            source_range = pd.date_range(start_source, end_source, freq='h', tz="UTC")
            eval_range = pd.date_range(start_eval, end_eval, tz="UTC", freq='h')
            target_range = pd.date_range("2019-05-01", "2020-04-30 23:00", freq='h', tz="UTC") #Not used for source training so dont care

    
    

    #Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        print("In Google Colab environment: Using .csv files")
    except:
        IN_COLAB = False
        print("Not in Colab environment: Using .pkl files")
    
    
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
        openmeteo = pd.read_pickle(f"Data/Sites/Reanalysis_{installation_int}.pkl")

        pvgis = pd.read_pickle(f"Data/Sites/PVGIS_{installation_int}.pkl")

        ceda = pd.read_pickle(f"Data/Sites/NWP_{installation_int}.pkl")
        is_day = pd.read_pickle(f"Data/Sites/is_day_{installation_int}.pkl")
        power = pd.read_pickle(f"Data/Sites/PV_{installation_int}.pkl")

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
    

    #Merge right data to have source, target and eval dataset
    looplist = [eval, target, source]
    power_list = [power, power, pvgis]
    range_list = [eval_range, target_range,source_range]
    data = []
    for i, entry in enumerate(looplist):
        if entry == "nwp":
            covariates = ceda
        elif entry == "era5":
            covariates = openmeteo
        elif entry == "no_weather":
            covariates = pd.DataFrame(index=openmeteo.index)
        elif entry is None:
            break
        else:
            raise KeyError
        
        if transform:
            data.append(merge_slice(range_list[i], power_list[i], covariates, is_day))
        else:
            data.append(merge_slice(range_list[i], power_list[i], covariates))
    site = location.Location(lat,lon, altitude = location.lookup_altitude(lat,lon), tz="UTC")
    
    for i, df in enumerate(data):
        times = df.index
        sol_pos = site.get_solarposition(times)
        mask = sol_pos["zenith"] > 85
        if source == "no_weather":
            filter_list = ['P']
        elif installation_int == 2:
            filter_list = ['P', "downward_surface_SW_flux"]
        else:
            filter_list = ['P', 'downward_surface_SW_flux', 'direct_surface_SW_flux', 'diffuse_surface_SW_flux']
        df.loc[mask,  filter_list] = 0
        # mask = (df['downward_surface_SW_flux'] - df['diffuse_surface_SW_flux'] - df["direct_surface_SW_flux"]*np.cos(sol_pos["zenith"])).abs().mean()
        # print(mask)
        data[i] = df
    # Add extra variates
    print(data[0]['P'].head(20))
    data = Featurisation(data)
    data.data = data.cyclic_features()
    


    if source != 'no_weather':
        if (transform):
            inv_list = [False, False, True] #Pre-processing only allowed for source_data 
            data.data = data.inverter_limit(inv_limit, inv_list)

        data.data = data.cyclic_angle('wind_direction_10m')
        data.data = data.add_shift('P') #Shift after inv_limit, otherwise discrepancy
        outlier_list = [False, True, True] #No outliers removed for evaluation as this is not

    for i in range(len(data.data)): #We added this bcs now we merge on outer of indices but lot of missing day data, now we want to have accurate regression so we do like this       
        data.data[i] = data.data[i].dropna()

    if source != "no_weather":
        data.data = data.remove_outliers(tolerance=100, outlier_list=outlier_list)
        if (installation_int == 2): #NWP Global only had GHI
            data.data = data.decomposition(lat, lon)
    else:
        data.data = data.add_shift('P')
        

    
    if transform:
        data.data = data.PoA(lat, lon, tilt, azimuth)
        data.data = data.clearsky_power(lat, lon, peakPower, tilt, azimuth)
        #data.data = data.deseasonalise(lat, lon)

    if source is not None:
        source_dataset = data.data[2]
    if target is not None:
        target_dataset = data.data[1]
    if eval is not None:
        eval_dataset= data.data[0]
    return source_dataset, target_dataset, eval_dataset 


      

