
import pandas as pd

# Define the file paths for the input CSVs and the KOFGI Excel file, and the output file.
file_pop_data = 'data/eq_pop04_page_linear.csv'
file_sdg_data = 'data/sdg_08_10_page_linear.csv'
file_kofgi_data = 'data/KOFGI_2025_public.xlsx'
output_csv_path = 'output/combined_data.csv'

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

def process_data(pop_filepath: str, sdg_filepath: str, kofgi_filepath: str, output_filepath: str):
    """
    Loads population, SDG, and KOFGI data, merges them using country names to IDs,
    filters for valid countries and year 2023 (for KOFGI), and saves the result.

    Args:
        pop_filepath (str): Path to the population CSV file.
        sdg_filepath (str): Path to the SDG CSV file.
        kofgi_filepath (str): Path to the KOFGI Excel file.
        output_filepath (str): Path to save the merged CSV file.
    """
    try:
        # --- Step 1: Load and process population data ---
        print(f"Loading population data from: {pop_filepath}")
        df_pop = pd.read_csv(pop_filepath)

        # Filter for countries present in our mapping and create 'country_id'
        # Assuming 'geo' column contains country names like 'Belgium', 'Denmark', etc.
        df_pop_filtered = df_pop[df_pop['geo'].isin(country_name_to_id_map.keys())].copy()
        df_pop_filtered['country_id'] = df_pop_filtered['geo'].map(country_name_to_id_map)

        # Select and rename columns. 'OBS_VALUE' from population data becomes 'avg_age'.
        df_pop_processed = df_pop_filtered[['country_id', 'OBS_VALUE']].rename(
            columns={'OBS_VALUE': 'avg_age'}
        )
        print(f"Population data processed. Found {len(df_pop_processed)} records for mapped countries.")

        # --- Step 2: Load and process SDG data ---
        print(f"Loading SDG data from: {sdg_filepath}")
        df_sdg = pd.read_csv(sdg_filepath)

        # Filter for countries present in our mapping and create 'country_id'
        df_sdg_filtered = df_sdg[df_sdg['geo'].isin(country_name_to_id_map.keys())].copy()
        df_sdg_filtered['country_id'] = df_sdg_filtered['geo'].map(country_name_to_id_map)

        # Select and rename columns. 'OBS_VALUE' from SDG data becomes 'gdp_pc'.
        df_sdg_processed = df_sdg_filtered[['country_id', 'OBS_VALUE']].rename(
            columns={'OBS_VALUE': 'gdp_pc'}
        )
        print(f"SDG data processed. Found {len(df_sdg_processed)} records for mapped countries.")

        # --- Step 3: Load and process KOFGI data ---
        print(f"Loading KOFGI data from: {kofgi_filepath}")
        df_kofgi = pd.read_excel(kofgi_filepath)

        # Filter for the year 2023
        df_kofgi_2023 = df_kofgi[df_kofgi['year'] == 2023].copy()

        # Filter for countries present in our mapping and create 'country_id'
        # Using 'country' column from KOFGI Excel as per user info
        df_kofgi_2023_filtered = df_kofgi_2023[df_kofgi_2023['country'].isin(country_name_to_id_map.keys())].copy()
        df_kofgi_2023_filtered['country_id'] = df_kofgi_2023_filtered['country'].map(country_name_to_id_map)

        # Select and rename columns. 'KOFGI' from KOFGI data becomes 'kofgi_form'.
        df_kofgi_processed = df_kofgi_2023_filtered[['country_id', 'KOFGI']].rename(
            columns={'KOFGI': 'kofgi_form'}
        )
        print(f"KOFGI data processed. Found {len(df_kofgi_processed)} records for 2023 in mapped countries.")


        # --- Step 4: Merge the datasets ---
        # Start with population data
        merged_df = df_pop_processed

        # Merge with SDG data
        print("Merging population and SDG datasets using an inner join on country_id...")
        merged_df = pd.merge(
            merged_df,
            df_sdg_processed,
            on='country_id',
            how='inner'
        )
        print(f"Population and SDG data merged. Current shape: {merged_df.shape}")

        # Merge with KOFGI data
        print("Merging with KOFGI data using an inner join on country_id...")
        merged_df = pd.merge(
            merged_df,
            df_kofgi_processed,
            on='country_id',
            how='inner'
        )
        print(f"All datasets merged. Resulting shape: {merged_df.shape}")


        # --- Step 5: Ensure the final DataFrame has only the required columns in order ---
        # Updated final_columns to include 'kofgi_form'
        final_columns = ['country_id', 'avg_age', 'gdp_pc', 'kofgi_form']
        
        # Ensure all columns exist and reorder them
        for col in final_columns:
            if col not in merged_df.columns:
                print(f"Warning: Column '{col}' not found after merge. This indicates an issue.")
                merged_df[col] = pd.NA 

        merged_df = merged_df[final_columns]
        print(f"Final DataFrame columns ordered: {final_columns}")

        # --- Step 6: Save the merged dataframe to a new CSV file ---
        print(f"Saving merged data to: {output_filepath}")
        merged_df.to_csv(output_filepath, index=False)
        print(f"Successfully created '{output_filepath}'")

        # Display the first 5 rows of the merged data as a markdown table for verification
        print("First 5 rows of the combined data:")
        # Use .to_markdown() for nice rendering in the output
        print(merged_df.head().to_markdown(index=False))

    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. Please check the paths. Details: {e}")
    except KeyError as e:
        print(f"Error: Expected column not found - {e}.")
        print("Please ensure the CSV files have 'geo' and 'OBS_VALUE' columns, and the Excel file has 'country', 'year', and 'KOFGI' columns.")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # Call the processing function with the defined file paths
    process_data(file_pop_data, file_sdg_data, file_kofgi_data, output_csv_path)
