import pandas as pd
import numpy as np
import os

# --- Configuration & Paths ---
file_pop_data = 'data/eq_pop04_page_linear.csv'
file_sdg_data = 'data/sdg_08_10_page_linear.csv'
file_ches_data = 'data/1999-2024_CHES_dataset_meansV2 (1).csv'
file_dhl_data = 'data/DHL_GCS_EU_filtered.csv' 
output_csv_path = 'output/combined_data.csv'

# Eurostat URL for total population (log calculation)
url_pop_data_for_log = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan$defaultview/1.0?compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=name"
POP_LOG_YEAR = 2024 

# Country mapping provided by the user
country_mapping = {
    1: 'Belgium', 2: 'Denmark', 3: 'Germany', 4: 'Greece', 5: 'Spain',
    6: 'France', 7: 'Ireland', 8: 'Italy', 10: 'Netherlands', 11: 'United Kingdom',
    12: 'Portugal', 13: 'Austria', 14: 'Finland', 16: 'Sweden', 20: 'Bulgaria',
    21: 'Czech Republic', 22: 'Estonia', 23: 'Hungary', 24: 'Latvia', 25: 'Lithuania',
    26: 'Poland', 27: 'Romania', 28: 'Slovakia', 29: 'Slovenia', 31: 'Croatia',
    37: 'Malta', 38: 'Luxembourg', 40: 'Cyprus'
}

# Inverted mapping for searching by name
name_to_id = {name: id for id, name in country_mapping.items()}

def calculate_country_weighted_galtan(country_group):
    """Logic from gov.py: Weighted GAL-TAN based on seat percentage."""
    valid_group = country_group.dropna(subset=['galtan', 'seat'])
    total_seats = valid_group['seat'].sum()
    if total_seats == 0:
        return np.nan
    return (valid_group['galtan'] * valid_group['seat']).sum() / total_seats

def process_data():
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # --- 1. Process Average Age Data ---
        print("Loading avg_age data...")
        df_pop_avg_age = pd.read_csv(file_pop_data)
        df_pop_avg_age['country_id'] = df_pop_avg_age['geo'].map(name_to_id)
        df_pop_processed = df_pop_avg_age.dropna(subset=['country_id'])[['country_id', 'OBS_VALUE']]
        df_pop_processed.rename(columns={'OBS_VALUE': 'avg_age'}, inplace=True)

        # --- 2. Process SDG (GDP) Data ---
        print("Loading SDG data...")
        df_sdg = pd.read_csv(file_sdg_data)
        df_sdg['country_id'] = df_sdg['geo'].map(name_to_id)
        df_sdg_processed = df_sdg.dropna(subset=['country_id'])[['country_id', 'OBS_VALUE']]
        df_sdg_processed.rename(columns={'OBS_VALUE': 'gdp_pc'}, inplace=True)

        # --- 3. Process CHES (GAL-TAN) Data ---
        print("Loading CHES data...")
        parl_gov = pd.read_csv(file_ches_data)
        parl_gov_2024 = parl_gov[parl_gov['year'] == 2024].copy()
        ches_scores = parl_gov_2024.groupby('country').apply(calculate_country_weighted_galtan, include_groups=False)
        ches_processed = ches_scores.reset_index(name='weighted_galtan').rename(columns={'country': 'country_id'})

        # --- 4. Process DHL Data ---
        print("Loading DHL index data...")
        df_dhl = pd.read_csv(file_dhl_data, sep=';')
        # Standardize names for mapping (e.g., Czechia -> Czech Republic)
        df_dhl['Country'] = df_dhl['Country'].replace({"Czechia": "Czech Republic"})
        df_dhl['country_id'] = df_dhl['Country'].map(name_to_id)
        dhl_processed = df_dhl.dropna(subset=['country_id'])[['country_id', '2024']]
        dhl_processed.rename(columns={'2024': 'dhl_index'}, inplace=True)

        # --- 5. Process Total Population & Log Population from URL ---
        print("Fetching log_population data from Eurostat...")
        df_pop_url = pd.read_csv(url_pop_data_for_log, compression="gzip")
        df_pop_url.columns = df_pop_url.columns.str.strip()
        df_pop_url_filtered = df_pop_url[
            (df_pop_url["age"] == "TOTAL") & (df_pop_url["sex"] == "T") & (df_pop_url["TIME_PERIOD"] == POP_LOG_YEAR)
        ].copy()
        df_pop_url_filtered["country_clean"] = df_pop_url_filtered["Geopolitical entity (reporting)"].replace({"Czechia": "Czech Republic"})
        df_pop_url_filtered['country_id'] = df_pop_url_filtered['country_clean'].map(name_to_id)
        
        df_log_pop_processed = df_pop_url_filtered.dropna(subset=['country_id']).copy()
        df_log_pop_processed["population"] = pd.to_numeric(df_log_pop_processed["OBS_VALUE"], errors="coerce")
        df_log_pop_processed["log_population"] = np.log(df_log_pop_processed["population"])
        df_log_pop_processed = df_log_pop_processed[['country_id', 'population', 'log_population']]

        # --- 6. Merge All Data ---
        print("Merging all datasets...")
        # Ensure ID columns are integers for matching
        dataframes = [df_pop_processed, df_sdg_processed, ches_processed, dhl_processed, df_log_pop_processed]
        for df in dataframes:
            df['country_id'] = df['country_id'].astype(int)

        merged_df = dataframes[0]
        for next_df in dataframes[1:]:
            merged_df = pd.merge(merged_df, next_df, on='country_id', how='inner')

        # --- 7. Final Formatting ---
        # Add the country name column based on the mapping
        merged_df['country_name'] = merged_df['country_id'].map(country_mapping)

        # Reorder columns as requested
        final_columns = [
            'country_id', 'country_name', 'avg_age', 'population', 
            'log_population', 'gdp_pc', 'weighted_galtan', 'dhl_index'
        ]
        merged_df = merged_df[final_columns]

        # Save to CSV
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Successfully created: {output_csv_path}")
        print(merged_df.head().to_markdown(index=False))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_data()