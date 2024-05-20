import pickle

ft_no_phys = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'P_24h_shift', "wind_speed_10m"]
ft_phys = ['PoA', "P_24h_shift", "is_day", 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'T_PV'] # 'P_24h_shift', "wind_speed_10m", "total_cloud_amount", "month_sin", "month_cos", "hour_sin", "hour_cos"
ft_no_w = ["P_24h_shift", "month_sin", "month_cos", "hour_sin", "hour_cos"]
ft_no_phys_sa = ['downward_surface_SW_flux','P_24h_shift', 'direct_surface_SW_flux', 'hour_cos', 'hour_sin', 'diffuse_surface_SW_flux', 'relative_humidity_1_5m', 'temperature_1_5m',  "wind_speed_10m",  "pressure_MSL", "month_cos", "month_sin", "total_cloud_amount"]
ft_phys_sa = ["is_day", 'PoA', "P_24h_shift", 'downward_surface_SW_flux', 'diffuse_surface_SW_flux',  'T_PV', "wind_speed_10m", 'relative_humidity_1_5m', 'direct_surface_SW_flux',     "month_sin", "month_cos", "hour_sin", "hour_cos", "temperature_1_5m","pressure_MSL", "total_cloud_amount"]
with open("features/ft_no_phys.pkl", 'wb') as f:
    pickle.dump(ft_no_phys, f)

with open("features/ft_phys.pkl", 'wb') as f:
    pickle.dump(ft_phys, f)

with open("features/ft_no_weather_no_phys.pkl", 'wb') as f:
    pickle.dump(ft_no_w, f)    

with open("features/ft_no_phys_sa.pkl", 'wb') as f:
    pickle.dump(ft_no_phys_sa, f)   
    
with open("features/ft_phys_sa.pkl", 'wb') as f:
    pickle.dump(ft_phys_sa, f)  
