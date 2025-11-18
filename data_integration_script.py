import pandas as pd
import numpy as np
from datetime import timedelta

# --- Constants for Specific Humidity Calculation ---
EPSILON = 0.622
STANDARD_PRESSURE_PA = 101325.0 

# --- Specific Humidity Calculation Functions ---
def calculate_saturation_vapor_pressure(T_celsius):
    if T_celsius >= 0:
        return 611.2 * np.exp( (17.67 * T_celsius) / (T_celsius + 243.5) )
    else:
        return 611.2 * np.exp( (22.5 * T_celsius) / (T_celsius + 273.15) )

def calculate_specific_humidity(T_celsius, RH_percent):
    e_s = calculate_saturation_vapor_pressure(T_celsius)
    e = e_s * (RH_percent / 100.0)
    q = (EPSILON * e) / (STANDARD_PRESSURE_PA - (1 - EPSILON) * e)
    return q * 1000.0

# --- Core Data Loading Functions ---

def load_korea_weather(sh_file_name, data_index):
    """Loads Korea weather data with Specific Humidity (assuming calculation is done)."""
    try:
        df = pd.read_csv(sh_file_name, encoding='utf-8')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        weekly = df[['Average Temperature(°C)', 'Specific Humidity (g/kg)']].resample('W-SUN').mean().rename(columns={'Average Temperature(°C)': 'ROK_Temp_Avg', 'Specific Humidity (g/kg)': 'ROK_SH_Avg'})
        print("성공: 한국 기상 파일 (비습 포함) 로드 완료.")
        return weekly
    except Exception as e:
        print(f"경고: 한국 기상 파일 로드 실패 ({e}). 임시 NaN 데이터로 대체.")
        return pd.DataFrame(index=data_index, data={'ROK_Temp_Avg': np.nan, 'ROK_SH_Avg': np.nan})

def load_naver_trends(file_name, data_index):
    """Loads and preprocesses the specific Naver Trends file."""
    try:
        # 이 파일은 7줄의 메타데이터와 '건강' 키워드가 포함되어 있어, 로드 로직을 하드코딩합니다.
        # 정확히 7줄 건너뛰고, 8번째 줄부터 데이터 로드
        df = pd.read_csv(file_name, skiprows=6, encoding='cp949') 
        df.columns = ['Date', 'ROK_Search_Naver']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').replace('-', np.nan).astype(float) 
        weekly = df['ROK_Search_Naver'].resample('W-SUN').mean().interpolate(method='linear').to_frame(name='ROK_Search_Naver')
        print(f"성공: {file_name} 파일 로드 및 주간 변환 완료.")
        return weekly
    except Exception as e:
        print(f"경고: Naver Trends 파일 로드 실패 ({e}). 임시 데이터로 대체.")
        return pd.DataFrame(index=data_index, data={'ROK_Search_Naver': np.nan})

def load_google_trends(file_name, col_name, skip_rows, data_index):
    """Loads and preprocesses Google Trends files."""
    try:
        df = pd.read_csv(file_name, skiprows=skip_rows, encoding='utf-8')
        # Google Trends 파일은 보통 3번째 컬럼이 데이터 값
        df.columns = ['Date', 'Category_Placeholder', col_name]
        df = df.drop(columns=['Category_Placeholder'], errors='ignore')
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').replace('<1', 0).astype(float)
        weekly = df[col_name].resample('W-SUN').mean().interpolate(method='linear').to_frame(name=col_name)
        print(f"성공: {file_name} 파일 로드 및 주간 변환 완료.")
        return weekly
    except Exception as e:
        print(f"경고: {file_name} 파일 로드 실패 ({e}). 임시 데이터로 대체.")
        return pd.DataFrame(index=data_index, data={col_name: np.nan})


# --- Main Execution ---
if __name__ == '__main__':
    # 기준이 되는 주간 인덱스 생성
    data_index = pd.date_range(start='2020-01-05', end='2025-11-16', freq='W-SUN')

    # 1. 한국 기상 데이터 로드 (가장 최근에 생성된 파일 사용)
    korea_weather_weekly = load_korea_weather(
        sh_file_name='Korea-ASOS_Temp-Hum_with_SH.csv', 
        data_index=data_index
    )

    # 2. 검색 데이터 로드
    kr_naver_search_weekly = load_naver_trends('NaverTrend_SouthKorea_Timeline.xlsx - 개요.csv', data_index)
    us_google_search_weekly = load_google_trends('GoogleTrend_US_multiTimeline.csv', 'US_Search_Avg', skip_rows=2, data_index=data_index)
    jp_google_search_weekly = load_google_trends('GoogleTrend_Japan_multiTimeline.csv', 'JP_Search_Avg', skip_rows=3, data_index=data_index)

    # 3. ILI/해외 기상 (가상 데이터)
    korea_ili_weekly = pd.DataFrame(index=data_index, data={'ROK_ILI': np.random.rand(len(data_index)) * 50})
    
    us_other_data_weekly = pd.DataFrame(index=data_index, data={
        'US_Hosp_Rate': np.random.rand(len(data_index)) * 5,  # CDC FluSurv-NET (가상)
        'US_Temp_Avg': np.random.rand(len(data_index)) * 25, # NOAA 기온 (가상)
    })
    us_data_weekly = us_other_data_weekly.join(us_google_search_weekly)

    jp_other_data_weekly = pd.DataFrame(index=data_index, data={
        'JP_ILI_Sentinel': np.random.rand(len(data_index)) * 15, # NIID ILI (가상)
        'JP_Temp_Avg': np.random.rand(len(data_index)) * 20,  # JMA 기온 (가상)
    })
    jp_data_weekly = jp_other_data_weekly.join(jp_google_search_weekly)

    
    # 4. 최종 데이터셋 통합
    integrated_df = korea_ili_weekly.join([
        korea_weather_weekly,
        kr_naver_search_weekly,
        us_data_weekly,
        jp_data_weekly,
    ])

    # 5. 데이터 정규화 및 저장
    integrated_df.index.name = 'Week_Start_Date'
    integrated_df = integrated_df.reset_index()

    for col in ['ROK_ILI', 'US_Hosp_Rate', 'JP_ILI_Sentinel']:
        # NaN 값이 있을 경우 Z-Score 계산을 위해 임시로 0 처리 (실제 모델링 시는 다른 방식 사용)
        temp_col = integrated_df[col].fillna(0)
        integrated_df[f'{col}_ZScore'] = (temp_col - temp_col.mean()) / temp_col.std()

    print("--- 통합 데이터셋 결측치 확인 ---")
    print(integrated_df.isnull().sum())

    integrated_df.to_csv('integrated_multi_national_data.csv', index=False)
    print("\n>>> 최종 통합 데이터셋 'integrated_multi_national_data.csv' 저장 완료. <<<")
    print(f"데이터 기간: {integrated_df['Week_Start_Date'].min()} to {integrated_df['Week_Start_Date'].max()}")
    print(f"총 주간 데이터 포인트: {len(integrated_df)}")

# End of code