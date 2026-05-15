import os
import numpy as np
import pandas as pd

# --- Configuration & Paths ---
file_pop_data = 'data/eq_pop04_page_linear.csv'
file_sdg_data = 'data/sdg_08_10_page_linear.csv'
file_ches_data = 'data/1999-2024_CHES_dataset_meansV2 (1).csv'
file_dhl_data = 'data/DHL_GCS_EU_filtered.csv'
file_equal_dex_data = 'data/equaldex_equality_index-2024-dec.csv'
file_ess_data = 'data/ESS11e04_1.csv'
file_ess_country_codes = 'data/ess_country_codes.csv'
output_csv_path = 'output/combined_data.csv'
output_ess_summary = 'output/ess_country_level_means.csv'

url_pop_data_for_log = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan$defaultview/1.0"
    "?compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=name"
)
POP_LOG_YEAR = 2024

country_mapping = {
    1: 'Belgium', 2: 'Denmark', 3: 'Germany', 4: 'Greece', 5: 'Spain',
    6: 'France', 7: 'Ireland', 8: 'Italy', 10: 'Netherlands',
    12: 'Portugal', 13: 'Austria', 14: 'Finland', 16: 'Sweden', 20: 'Bulgaria',
    21: 'Czech Republic', 22: 'Estonia', 23: 'Hungary', 24: 'Latvia', 25: 'Lithuania',
    26: 'Poland', 27: 'Romania', 28: 'Slovakia', 29: 'Slovenia', 31: 'Croatia',
    37: 'Malta', 40: 'Cyprus'
}

name_to_id = {name: id for id, name in country_mapping.items()}

name_corrections = {
    "Czechia": "Czech Republic",
    "Slovak Republic": "Slovakia",
    "Great Britain": "United Kingdom",
    "UK": "United Kingdom"
}

def ensure_directory_for(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_equal_dex_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def process_equal_dex_scores(df: pd.DataFrame) -> pd.DataFrame:
    processed = df[['Name', 'EI Legal']].copy()
    processed['Name'] = processed['Name'].astype(str).str.strip()
    processed['Name'] = processed['Name'].replace(name_corrections)
    processed['country_id'] = processed['Name'].map(name_to_id)
    processed['ei_legal'] = pd.to_numeric(processed['EI Legal'], errors='coerce')
    processed = processed.dropna(subset=['country_id']).copy()
    processed['country_id'] = processed['country_id'].astype(int)
    return processed[['country_id', 'ei_legal']]

def load_ess_country_codes(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Value': 'Value',
        'Category': 'Category'
    })
    df['Category'] = df['Category'].astype(str).str.strip()
    df['Value'] = df['Value'].astype(str).str.strip()
    df = df.dropna(subset=['Value']).copy()
    return df

def build_country_mapping(ess_codes: pd.DataFrame) -> dict[str, str]:
    return {row['Value']: row['Category'] for _, row in ess_codes.sort_values('Value').iterrows()}

def process_ess_survey_data() -> pd.DataFrame:
    ess_df = pd.read_csv(file_ess_data, low_memory=False)
    codes_df = load_ess_country_codes(file_ess_country_codes)
    country_map = build_country_mapping(codes_df)

    ess_df['cntry'] = ess_df['cntry'].astype(str).str.strip()
    ess_df['cntryname'] = ess_df['cntry'].map(country_map)
    ess_df['country_id'] = ess_df['cntryname'].map(name_to_id)

    weighted_columns = ['sclmeet', 'rlgblg', 'rlgdgr']
    ess_df['anweight'] = pd.to_numeric(ess_df['anweight'], errors='coerce')
    for col in weighted_columns:
        ess_df[col] = pd.to_numeric(ess_df[col], errors='coerce')
    
    # Drop missing countries to speed up processing
    df_valid = ess_df.dropna(subset=['country_id']).copy()

    for col in weighted_columns:
        # weigthing score using ESS anweight
        df_valid[f'{col}_prod'] = df_valid[col] * df_valid['anweight']
        
        # columns on anweights not null 
        df_valid[f'{col}_valid_wt'] = df_valid['anweight'].where(df_valid[col].notna())

   # summing rows by country
    agg_cols = [f'{col}_prod' for col in weighted_columns] + [f'{col}_valid_wt' for col in weighted_columns]
    summary = df_valid.groupby('country_id', as_index=False)[agg_cols].sum()

    # final weighted mean (weigthed value/sum of weights)
    for col in weighted_columns:
        summary[f'{col}_wg'] = summary[f'{col}_prod'] / summary[f'{col}_valid_wt']

# final extraction
    final_cols = ['country_id'] + [f'{col}_wg' for col in weighted_columns]
    summary = summary[final_cols]
    summary['country_name'] = summary['country_id'].map(country_mapping)
    return summary

def calculate_country_weighted_galtan(country_group: pd.DataFrame) -> float:
    valid_group = country_group.dropna(subset=['galtan', 'seat'])
    total_seats = valid_group['seat'].sum()
    if total_seats == 0:
        return np.nan
    return (valid_group['galtan'] * valid_group['seat']).sum() / total_seats

def process_data() -> None:
    try:
        ensure_directory_for(output_csv_path)
        ensure_directory_for(output_ess_summary)

        print("Loading EqualDex equality index data…")
        equal_dex_df = load_equal_dex_data(file_equal_dex_data)
        equal_dex_processed = process_equal_dex_scores(equal_dex_df)

        print("Processing ESS survey data…")
        ess_summary_df = process_ess_survey_data()
        ess_summary_df.to_csv(output_ess_summary, index=False)
        ess_merge_df = ess_summary_df[['country_id', 'sclmeet_wg', 'rlgblg_wg', 'rlgdgr_wg']].copy()

        print("Loading avg_age data…")
        df_pop_avg_age = pd.read_csv(file_pop_data)
        df_pop_avg_age['geo'] = df_pop_avg_age['geo'].replace(name_corrections)
        df_pop_avg_age['country_id'] = df_pop_avg_age['geo'].map(name_to_id)
        df_pop_processed = df_pop_avg_age.dropna(subset=['country_id'])[['country_id', 'OBS_VALUE']]
        df_pop_processed.rename(columns={'OBS_VALUE': 'avg_age'}, inplace=True)

        print("Loading SDG (gdp per capita) data…")
        df_sdg = pd.read_csv(file_sdg_data)
        df_sdg['geo'] = df_sdg['geo'].replace(name_corrections)
        df_sdg['country_id'] = df_sdg['geo'].map(name_to_id)
        df_sdg_processed = df_sdg.dropna(subset=['country_id'])[['country_id', 'OBS_VALUE']]
        df_sdg_processed.rename(columns={'OBS_VALUE': 'gdp_pc'}, inplace=True)

        print("Loading CHES data…")
        parl_gov = pd.read_csv(file_ches_data)
        parl_gov_2024 = parl_gov[parl_gov['year'] == POP_LOG_YEAR].copy()
        ches_scores = parl_gov_2024.groupby('country').apply(
            calculate_country_weighted_galtan, include_groups=False
        )
        ches_processed = ches_scores.reset_index(name='weighted_galtan').rename(columns={'country': 'country_id'})

        print("Loading DHL index data…")
        df_dhl = pd.read_csv(file_dhl_data, sep=';')
        df_dhl['Country'] = df_dhl['Country'].replace(name_corrections)
        df_dhl['country_id'] = df_dhl['Country'].map(name_to_id)
        dhl_processed = df_dhl.dropna(subset=['country_id'])[['country_id', '2024']]
        dhl_processed.rename(columns={'2024': 'dhl_index'}, inplace=True)

        print("Fetching log_population data from Eurostat…")
        df_pop_url = pd.read_csv(url_pop_data_for_log, compression="gzip")
        df_pop_url.columns = df_pop_url.columns.str.strip()
        df_pop_url_filtered = df_pop_url[
            (df_pop_url["age"] == "TOTAL") &
            (df_pop_url["sex"] == "T") &
            (df_pop_url["TIME_PERIOD"] == POP_LOG_YEAR)
        ].copy()
        df_pop_url_filtered["country_clean"] = (
            df_pop_url_filtered["Geopolitical entity (reporting)"]
            .replace(name_corrections)
        )
        df_pop_url_filtered['country_id'] = df_pop_url_filtered['country_clean'].map(name_to_id)
        df_log_pop_processed = df_pop_url_filtered.dropna(subset=['country_id']).copy()
        df_log_pop_processed["population"] = pd.to_numeric(df_log_pop_processed["OBS_VALUE"], errors="coerce")
        df_log_pop_processed["log_population"] = np.log(df_log_pop_processed["population"])
        df_log_pop_processed = df_log_pop_processed[['country_id', 'population', 'log_population']]

        print("Merging all datasets…")
        base_df = pd.DataFrame(list(country_mapping.items()), columns=['country_id', 'country_name'])
        dataframes = [
            df_pop_processed,
            df_sdg_processed,
            ches_processed,
            dhl_processed,
            df_log_pop_processed,
            ess_merge_df,
            equal_dex_processed,
        ]
        for df in dataframes:
            df['country_id'] = df['country_id'].astype(int)

        merged_df = base_df.copy()
        for next_df in dataframes:
            merged_df = pd.merge(merged_df, next_df, on='country_id', how='left')

        final_columns = [
            'country_id', 'country_name', 'avg_age', 'population',
            'log_population', 'gdp_pc', 'weighted_galtan', 'dhl_index',
            'sclmeet_wg', 'rlgblg_wg', 'rlgdgr_wg', 'ei_legal'
        ]
        merged_df = merged_df[final_columns]

        merged_df.to_csv(output_csv_path, index=False)
        print(f"Successfully created: {output_csv_path}")
        print(f"Total rows in combined data: {len(merged_df)}")
        print(merged_df.head().to_markdown(index=False))

    except Exception as e:
        print(f"An error occurred while processing data: {e}")

if __name__ == "__main__":
    process_data()