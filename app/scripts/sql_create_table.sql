CREATE TABLE elec_actuals (
    time TIMESTAMP PRIMARY KEY,
    load_kw DECIMAL(10,2) NOT NULL,
    pressure_kpa DECIMAL(7,2) NOT NULL,
    cloud_cover_pct DECIMAL(5,2) NOT NULL,
    humidity_pct DECIMAL(5,2) NOT NULL,
    temperature_c DECIMAL(5,2) NOT NULL,
    wind_direction_deg DECIMAL(5,2) NOT NULL,
    wind_speed_kmh DECIMAL(5,2) NOT NULL
);

CREATE TABLE elec_forecasts (
    time TIMESTAMP PRIMARY KEY,
    pressure_kpa DECIMAL(7,2) NOT NULL,
    cloud_cover_pct DECIMAL(5,2) NOT NULL,
    temperature_c DECIMAL(5,2) NOT NULL,
    wind_direction_deg DECIMAL(5,2) NOT NULL,
    wind_speed_kmh DECIMAL(5,2) NOT NULL,
    forecast_1 DECIMAL(10,2),
    forecast_2 DECIMAL(10,2)
);
