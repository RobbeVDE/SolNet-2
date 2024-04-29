from pvlib import location

site = location.Location(9.93676, -84.04388)

print(location.lookup_altitude(9.93676, -84.04388))

