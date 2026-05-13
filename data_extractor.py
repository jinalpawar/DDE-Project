
import pandas as pd
import numpy as np # Import numpy for log calculation

# Define the file paths for the input CSVs and the KOFGI Excel file, and the output file.
file_pop_data = 'data/eq_pop04_page_linear.csv'
file_sdg_data = 'data/sdg_08_10_page_linear.csv'
file_kofgi_data = 'data/KOFGI_2025_public.xlsx'
output_csv_path = 'output/combined_data.csv'

# New constants for total population data from URL
url_pop_data_for_log = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan$defaultview/1.0?compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=name"
POP_LOG_YEAR = 2023 # User requested 2023 for log population

# The provided country mapping.
country_mapping = {
    1: 'Belgium', 2: 'Denmark', 3: 'Germany', 4: 'Greece', 5: 'Spain',
    6: 'France', 7: 'Ireland', 8: 'Italy', 10: 'Netherlands', 11: 'United Kingdom',
    12: 'Portugal', 13: 'Austria', 14: 'Finland', 16: 'Sweden', 20: 'Bulgaria',
    21: 'Czech Republic', 22: 'Estonia', 23: 'Hungary', 24: 'Latvia', 25: 'Lithuania',
    26: 'Poland', 27: 'Romania', 28: 'Slovakia', 29: 'Slovenia', 31: 'Croatia',
    37: 'Malta', 38: 'Luxembourg', 40: 'Cyprus'
}

# Invert the country mapping to map country names to IDs
country_name_to_id_map = {name: id for id, name in country_mapping.items()}

# Create a DataFrame from country_mapping for easier merging
mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['country_id', 'country'])

def process_data(pop_avg_age_filepath: str, sdg_filepath: str, kofgi_filepath: str, url_pop_filepath: str, year_pop_log: int, output_filepath: str):
    """
    Loads population (avg_age), SDG, KOFGI, and total population data,
    merges them, calculates log population, and saves the result.

    Args:
        pop_avg_age_filepath (str): Path to the population CSV file for avg_age.
        sdg_filepath (str): Path to the SDG CSV file for gdp_pc.
        kofgi_filepath (str): Path to the KOFGI Excel file.
        url_pop_filepath (str): URL for total population data.
        year_pop_log (int): The year to filter total population data for log calculation.
        output_filepath (str): Path to save the merged CSV file.
    """
    try:
        # --- Step 1: Load and process population data for avg_age ---
        # This part remains the same as the original script, using pop_avg_age_filepath
        print(f"Loading population data for avg_age from: {pop_avg_age_filepath}")
        df_pop_avg_age = pd.read_csv(pop_avg_age_filepath)
        df_pop_avg_age_filtered = df_pop_avg_age[df_pop_avg_age['geo'].isin(country_name_to_id_map.keys())].copy()
        df_pop_avg_age_filtered['country_id'] = df_pop_avg_age_filtered['geo'].map(country_name_to_id_map)
        df_pop_processed = df_pop_avg_age_filtered[['country_id', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'avg_age'})
        print(f"Population data (avg_age) processed. Found {len(df_pop_processed)} records for mapped countries.")

        # --- Step 2: Load and process SDG data for gdp_pc ---
        print(f"Loading SDG data from: {sdg_filepath}")
        df_sdg = pd.read_csv(sdg_filepath)
        df_sdg_filtered = df_sdg[df_sdg['geo'].isin(country_name_to_id_map.keys())].copy()
        df_sdg_filtered['country_id'] = df_sdg_filtered['geo'].map(country_name_to_id_map)
        df_sdg_processed = df_sdg_filtered[['country_id', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'gdp_pc'})
        print(f"SDG data processed. Found {len(df_sdg_processed)} records for mapped countries.")

        # --- Step 3: Load and process KOFGI data for KOF ---
        print(f"Loading KOFGI data from: {kofgi_filepath}")
        df_kofgi = pd.read_excel(kofgi_filepath)
        # Filter for the year 2023 as per KOFGI requirement
        df_kofgi_2023 = df_kofgi[df_kofgi['year'] == 2023].copy()
        # Filter for countries present in our mapping
        df_kofgi_2023_filtered = df_kofgi_2023[df_kofgi_2023['country'].isin(country_name_to_id_map.keys())].copy()
        df_kofgi_2023_filtered['country_id'] = df_kofgi_2023_filtered['country'].map(country_name_to_id_map)
        df_kofgi_processed = df_kofgi_2023_filtered[['country_id', 'KOFGI']].rename(columns={'KOFGI': 'KOF'})
        print(f"KOFGI data processed. Found {len(df_kofgi_processed)} records for 2023 in mapped countries.")

        # --- Step 4: Load and process TOTAL POPULATION data from URL for log_population ---
        print(f"Loading total population data for log calculation from: {url_pop_filepath} for year {year_pop_log}")
        df_pop_url = pd.read_csv(url_pop_filepath, compression="gzip")
        df_pop_url.columns = df_pop_url.columns.str.strip()

        # Filter for total population, male/female total, and the specified year
        df_pop_url_filtered = df_pop_url[
            (df_pop_url["age"] == "TOTAL") &
            (df_pop_url["sex"] == "T") &
            (df_pop_url["TIME_PERIOD"] == year_pop_log)
        ].copy()

        # Rename columns to be consistent
        df_pop_url_renamed = df_pop_url_filtered.rename(columns={
            "Geopolitical entity (reporting)": "country",
            "OBS_VALUE": "population", # This OBS_VALUE is total population
            "TIME_PERIOD": "year"
        })

        # Handle 'Czechia' mapping to 'Czech Republic'
        df_pop_url_renamed["country"] = df_pop_url_renamed["country"].replace({"Czechia": "Czech Republic"})

        # Merge with mapping_df to ensure we only get mapped countries and their IDs
        df_pop_url_merged_with_map = pd.merge(
            mapping_df, # Use mapping_df to get country_id for mapped countries
            df_pop_url_renamed[["country", "year", "population"]], # Select relevant columns
            on="country",
            how="inner" # Inner join ensures we only keep countries present in mapping_df
        )

        # Data cleaning and type conversion for population and year
        df_pop_url_merged_with_map["population"] = pd.to_numeric(df_pop_url_merged_with_map["population"], errors="coerce")
        df_pop_url_merged_with_map = df_pop_url_merged_with_map.dropna(subset=["population"]).copy() # Remove rows with invalid population
        df_pop_url_merged_with_map["year"] = df_pop_url_merged_with_map["year"].astype(int)
        df_pop_url_merged_with_map["population"] = df_pop_url_merged_with_map["population"].astype(int) # Keep population as integer

        # Calculate log population
        df_pop_url_merged_with_map["log_population"] = np.log(df_pop_url_merged_with_map["population"])

        # Select columns for the new processed dataframe that will be merged later
        df_log_pop_processed = df_pop_url_merged_with_map[['country_id', 'year', 'population', 'log_population']]
        print(f"Total population data processed. Found {len(df_log_pop_processed)} records for {year_pop_log} in mapped countries.")

        # --- Step 5: Merge all processed dataframes ---
        # Start with population data for avg_age
        merged_df = df_pop_processed
        print(f"Initial merge_df shape (avg_age): {merged_df.shape}")

        # Merge with SDG data
        print("Merging avg_age and SDG data...")
        merged_df = pd.merge(merged_df, df_sdg_processed, on='country_id', how='inner')
        print(f"Shape after merging SDG: {merged_df.shape}")

        # Merge with KOFGI data
        print("Merging with KOFGI data...")
        merged_df = pd.merge(merged_df, df_kofgi_processed, on='country_id', how='inner')
        print(f"Shape after merging KOFGI: {merged_df.shape}")

        # Merge with the new total population & log_population data
        print("Merging with total population (log_population) data...")
        merged_df = pd.merge(merged_df, df_log_pop_processed, on='country_id', how='inner')
        print(f"Shape after merging log_population data: {merged_df.shape}")

        # --- Step 6: Ensure the final DataFrame has only the required columns in order ---
        # Updated final_columns to include new columns and year
        final_columns = ['country_id', 'avg_age', 'population', 'log_population', 'gdp_pc', 'KOF', 'year']
        
        # Ensure all columns exist and reorder them
        for col in final_columns:
            if col not in merged_df.columns:
                print(f"Warning: Column '{col}' not found after merge. This indicates a potential issue.")
                merged_df[col] = pd.NA # Add missing column with NA values

        merged_df = merged_df[final_columns]
        print(f"Final DataFrame columns ordered: {final_columns}")

        # --- Step 7: Save the merged dataframe to a new CSV file ---
        print(f"Saving merged data to: {output_filepath}")
        merged_df.to_csv(output_filepath, index=False)
        print(f"Successfully created '{output_filepath}'")

        # Display the first 5 rows of the merged data as a markdown table for verification
        print("First 5 rows of the combined data:")
        print(merged_df.head().to_markdown(index=False))

    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. Please check the paths. Details: {e}")
    except KeyError as e:
        print(f"Error: Expected column not found - {e}.")
        print("Please ensure input files have expected columns (e.g., 'geo', 'OBS_VALUE', 'country', 'year', 'KOFGI').")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # Call the processing function with all defined file paths and parameters
    process_data(file_pop_data, file_sdg_data, file_kofgi_data, url_pop_data_for_log, POP_LOG_YEAR, output_csv_path)
