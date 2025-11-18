import pandas as pd
import numpy as np
import os

# --- Constants for Specific Humidity Calculation ---
# Ratio of molecular weight of water vapor to dry air
EPSILON = 0.622
# Standard atmospheric pressure (1013.25 hPa -> 101325 Pa). Assumed since pressure data is unavailable.
STANDARD_PRESSURE_PA = 101325.0 

def calculate_saturation_vapor_pressure(T_celsius):
    """
    Calculates saturation vapor pressure (Pa) using the enhanced Magnus-Tetens formula.
    T_celsius: Temperature in Celsius (°C)
    """
    # Uses different coefficients for T >= 0 and T < 0
    if T_celsius >= 0:
        return 611.2 * np.exp( (17.67 * T_celsius) / (T_celsius + 243.5) )
    else:
        # Approximation for saturation vapor pressure over ice (T < 0)
        return 611.2 * np.exp( (22.5 * T_celsius) / (T_celsius + 273.15) )


def calculate_specific_humidity(T_celsius, RH_percent):
    """
    Calculates Specific Humidity (q in kg/kg) from temperature and relative humidity.
    The result is returned in g/kg.
    """
    # 1. Calculate saturation vapor pressure (e_s in Pa)
    e_s = calculate_saturation_vapor_pressure(T_celsius)
    
    # 2. Calculate actual vapor pressure (e in Pa)
    e = e_s * (RH_percent / 100.0)
    
    # 3. Calculate specific humidity (q in kg/kg)
    # q = (EPSILON * e) / (P - (1 - EPSILON) * e)
    q = (EPSILON * e) / (STANDARD_PRESSURE_PA - (1 - EPSILON) * e)
    
    # 4. Convert result to g/kg (the typical unit for Specific Humidity in modeling)
    return q * 1000.0

def main():
    """
    Main execution logic to load the file, calculate specific humidity, and save the new file.
    """
    file_name = 'Korea-ASOS_Temp-Hum.csv'

    try:
        # Load the CSV file
        df = pd.read_csv(file_name, encoding='cp949')
        
        # Define necessary column names
        T_col = 'Average Temperature(°C)'
        RH_col = 'Average Relative Humidity(%)'
        
        # Data type conversion (handling potential strings)
        df[T_col] = pd.to_numeric(df[T_col], errors='coerce')
        df[RH_col] = pd.to_numeric(df[RH_col], errors='coerce')
        
        # Drop rows where calculation is impossible
        df.dropna(subset=[T_col, RH_col], inplace=True)

        # Calculate Specific Humidity and add the new column
        # Round the result to 3 decimal places (g/kg)
        df['Specific Humidity (g/kg)'] = df.apply(
            lambda row: calculate_specific_humidity(row[T_col], row[RH_col]), axis=1
        ).round(3)

        # --- Output Results ---
        print(f"Success: Specific humidity calculated in '{file_name}'.")
        print("\n--- Calculated Specific Humidity Data (Top 5 Rows) ---")
        print(df[['Date', T_col, RH_col, 'Specific Humidity (g/kg)']].head())
        
        # --- Save the new file ---
        output_file_name = 'Korea-ASOS_Temp-Hum_with_SH.csv'
        df.to_csv(output_file_name, index=False, encoding='utf-8')
        print(f"\n>>> File with specific humidity added: '{output_file_name}' saved. <<<")
        
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Please verify the file path.")
    except Exception as e:
        print(f"Error during calculation: {e}")

# Standard Python execution entry point
if __name__ == '__main__':
    main()