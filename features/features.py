import pickle

ft_no_phys = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'P_24h_shift', "wind_speed_10m"]
ft_phys = ['temperature_1_5m', 'PoA', "is_day", "cs_power", 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'P_24h_shift', "wind_speed_10m", "total_cloud_amount", "month_sin", "month_cos", "hour_sin", "hour_cos"]
ft_no_w = ["P_24h_shift", "month_sin", "month_cos", "hour_sin", "hour_cos"]
with open("features/ft_no_phys.pkl", 'wb') as f:
    pickle.dump(ft_no_phys, f)

with open("features/ft_phys.pkl", 'wb') as f:
    pickle.dump(ft_phys, f)

with open("features/ft_no_weather_no_phys.pkl", 'wb') as f:
    pickle.dump(ft_no_w, f)    