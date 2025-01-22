import os
import sys
import json
import pandas as pd
import re
from typing import List


CONFIG_FILE_PATH = "auxiliary_files/skills_occupation_config.json"


# Set DEBUG_MODE to True to enable debug prints, False to disable
DEBUG_MODE = True

def debug_print(message: str, data=None):
    """
    Prints debug messages along with data samples if DEBUG_MODE is enabled.
    
    Args:
        message (str): Description of the debug message.
        data: The data to be printed (optional).
    """
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
        if data is not None:
            if isinstance(data, pd.DataFrame):
                print(data.head())  # Show first 5 rows for DataFrames
            elif isinstance(data, list):
                for item in data[:5]:  # Show first 5 items for lists
                    print(item)
            elif isinstance(data, dict):
                # Show first 5 key-value pairs for dictionaries
                for i, (k, v) in enumerate(data.items()):
                    print(f"{k}: {v}")
                    if i >= 4:
                        break
            else:
                print(data)
            print("-" * 50)

def load_config(config_path: str) -> dict:
    """
    Load the JSON configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    debug_print("Loaded configuration:", config)
    return config

def find_csv_files(local_folder: str, task: str) -> List[str]:
    """
    Find all CSV files in the local folder that correspond to the given task.

    Args:
        local_folder (str): Path to the local directory containing CSV files.
        task (str): Task name ('skills' or 'occupation').

    Returns:
        List[str]: List of file paths matching the task.
    """
    csv_files = []
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            if file.lower().endswith('.csv') and task in file.lower():
                csv_files.append(os.path.join(root, file))
    debug_print(f"Found CSV files for task '{task}':", csv_files)
    return csv_files

def concatenate_csv_files(csv_files: List[str], task: str) -> pd.DataFrame:
    """
    Concatenate multiple CSV files into a single DataFrame and drop duplicates based on 'job_guid'.

    Args:
        csv_files (List[str]): List of CSV file paths.
        task (str): Task name ('skills' or 'occupation').

    Returns:
        pd.DataFrame: Concatenated DataFrame with duplicates removed.
    """
    if not csv_files:
        print(f"[WARNING] No CSV files found for task '{task}'.")
        return pd.DataFrame()

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            initial_shape = df.shape
            df = df.drop_duplicates(subset=['job_guid'])
            duplicates_dropped = initial_shape[0] - df.shape[0]
            if duplicates_dropped > 0:
                print(f"[INFO] Dropped {duplicates_dropped} duplicate records from file: {file}.")
            dataframes.append(df)
            print(f"[INFO] Loaded CSV file: {file} with shape {df.shape}.")
            debug_print(f"Sample data from file '{file}':", df)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV file {file}: {e}")

    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        pre_drop_shape = concatenated_df.shape
        concatenated_df = concatenated_df.drop_duplicates(subset=['job_guid'])
        duplicates_dropped = pre_drop_shape[0] - concatenated_df.shape[0]
        if duplicates_dropped > 0:
            print(f"[INFO] Dropped {duplicates_dropped} duplicate records after concatenating CSV files for task '{task}'.")
        print(f"[INFO] Concatenated {len(dataframes)} CSV files for task '{task}' into a DataFrame with shape {concatenated_df.shape}.")
        debug_print(f"Sample data after concatenating CSV files for task '{task}':", concatenated_df)
        return concatenated_df
    else:
        print(f"[WARNING] No valid DataFrames to concatenate for task '{task}'.")
        return pd.DataFrame()

def strip_code_fences(value: str) -> str:
    """
    Removes any ``` or ```json fences (and whitespace) from the string.
    """
    # Remove occurrences of ```json, ``` or ```
    stripped = re.sub(r'```(?:json)?\s*', '', value)
    return stripped.strip()

def post_process_occupation_results(raw_results: str):
    """
    Processes the raw_results string by removing code fences and extracting the 'occupation' data.

    Args:
        raw_results (str): The raw string containing JSON data, possibly wrapped in code fences.

    Returns:
        list: A list of occupations extracted from the JSON data.
    """
    processed_results = {}
    try:
        # 1. Remove code fences
        clean_value = strip_code_fences(raw_results)
        debug_print("Cleaned value after stripping code fences:", clean_value)
        
        # 2. Parse the cleaned JSON
        parsed_value = json.loads(clean_value)
        debug_print("Parsed JSON object:", parsed_value)

        if isinstance(parsed_value, dict) and "occupation" in parsed_value:
            processed_results = parsed_value["occupation"]
            debug_print("Extracted 'occupation' field:", processed_results)
        else:
            print("No 'occupation' field found in the parsed JSON.")
    except json.JSONDecodeError as e:
        print(f"Failed to parse raw_results: {e}")
    
    return processed_results

def main():
    # Use the fixed session_id as per the original script
    session_id = '20250122-171108-81fb0'

    # Load configuration
    try:
        config = load_config(CONFIG_FILE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    # Extract necessary paths from config
    try:
        paths_config = config["paths"]
        local_output_folder = paths_config.get("local_output_folder", "downloaded_outputs/{session_number}")
        local_output_folder = local_output_folder.format(session_number=session_id)

        # Corrected the format string by specifying the placeholder name
        final_parquet_file = paths_config["final_parquet_file"].format(session_number=session_id)
        existing_parquet_file = paths_config["extracted_data_path"]
    except KeyError as e:
        print(f"[ERROR] Missing configuration parameter: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error in formatting final_parquet_file: {e}")
        sys.exit(1)

    debug_print("Configuration paths extracted:", {
        "local_output_folder": local_output_folder,
        "final_parquet_file": final_parquet_file,
        "existing_parquet_file": existing_parquet_file
    })

    # Verify local_output_folder exists
    if not os.path.exists(local_output_folder):
        print(f"[ERROR] Local output folder does not exist: {local_output_folder}")
        sys.exit(1)
    debug_print("Verified existence of local_output_folder:", local_output_folder)

    # Find and concatenate CSV files for 'occupation'
    occupation_csv_files = find_csv_files(local_output_folder, 'occupation')
    occupation_df = concatenate_csv_files(occupation_csv_files, 'occupation')
    if not occupation_df.empty:
        print(f"[INFO] Occupation DataFrame shape after removing duplicates: {occupation_df.shape}")
        debug_print("Occupation DataFrame sample:", occupation_df)
    else:
        print("[INFO] Occupation DataFrame is empty.")

    # Find and concatenate CSV files for 'skills'
    skills_csv_files = find_csv_files(local_output_folder, 'skills')
    skills_df = concatenate_csv_files(skills_csv_files, 'skills')
    if not skills_df.empty:
        print(f"[INFO] Skills DataFrame shape after removing duplicates: {skills_df.shape}")
        debug_print("Skills DataFrame sample:", skills_df)
    else:
        print("[INFO] Skills DataFrame is empty.")

    # Check if both DataFrames are non-empty
    if occupation_df.empty:
        print("[WARNING] Occupation DataFrame is empty. Proceeding without occupation data.")
    if skills_df.empty:
        print("[WARNING] Skills DataFrame is empty. Proceeding without skills data.")

    if occupation_df.empty and skills_df.empty:
        print("[ERROR] Both Occupation and Skills DataFrames are empty. Exiting.")
        sys.exit(1)

    # Merge 'occupation' and 'skills' DataFrames on 'job_guid' using inner join
    try:
        if not occupation_df.empty and not skills_df.empty:
            merged_tasks_df = pd.merge(occupation_df, skills_df, on='job_guid', how='inner')
            pre_drop_shape = merged_tasks_df.shape
            merged_tasks_df = merged_tasks_df.drop_duplicates(subset=['job_guid'])
            duplicates_dropped = pre_drop_shape[0] - merged_tasks_df.shape[0]
            if duplicates_dropped > 0:
                print(f"[INFO] Dropped {duplicates_dropped} duplicate records after merging 'occupation' and 'skills' DataFrames.")
            print(f"[INFO] Merged 'occupation' and 'skills' DataFrames into a single DataFrame with shape {merged_tasks_df.shape}.")
            debug_print("Merged 'occupation' and 'skills' DataFrame sample:", merged_tasks_df)
        elif not occupation_df.empty:
            merged_tasks_df = occupation_df
            print(f"[INFO] Only 'occupation' DataFrame is available with shape {merged_tasks_df.shape}.")
            debug_print("Only 'occupation' DataFrame sample:", merged_tasks_df)
        else:
            merged_tasks_df = skills_df
            print(f"[INFO] Only 'skills' DataFrame is available with shape {merged_tasks_df.shape}.")
            debug_print("Only 'skills' DataFrame sample:", merged_tasks_df)
    except Exception as e:
        print(f"[ERROR] Failed to merge 'occupation' and 'skills' DataFrames: {e}")
        sys.exit(1)

    # Check if the existing Parquet file exists
    if not os.path.exists(existing_parquet_file):
        print(f"[ERROR] Existing Parquet file does not exist: {existing_parquet_file}")
        sys.exit(1)
    debug_print("Verified existence of existing Parquet file:", existing_parquet_file)

    # Load existing Parquet file
    try:
        existing_parquet_df = pd.read_parquet(existing_parquet_file)
        print(f"[INFO] Loaded existing Parquet file: {existing_parquet_file} with shape {existing_parquet_df.shape}.")
        debug_print("Existing Parquet DataFrame sample:", existing_parquet_df)

        # Drop duplicates in existing_parquet_df
        existing_parquet_before = existing_parquet_df.shape[0]
        existing_parquet_df = existing_parquet_df.drop_duplicates(subset=['job_guid'])
        duplicates_dropped = existing_parquet_before - existing_parquet_df.shape[0]
        if duplicates_dropped > 0:
            print(f"[INFO] Dropped {duplicates_dropped} duplicate records from existing Parquet DataFrame.")
        print(f"[INFO] Existing Parquet DataFrame shape after removing duplicates: {existing_parquet_df.shape}.")
        debug_print("Existing Parquet DataFrame after dropping duplicates:", existing_parquet_df)
    except Exception as e:
        print(f"[ERROR] Failed to read existing Parquet file {existing_parquet_file}: {e}")
        sys.exit(1)

    # Merge the combined DataFrame with the existing Parquet DataFrame on 'job_guid' using inner join
    try:
        final_df = pd.merge(existing_parquet_df, merged_tasks_df, on='job_guid', how='inner')
        pre_drop_final_shape = final_df.shape[0]
        final_df = final_df.drop_duplicates(subset=['job_guid'])
        duplicates_dropped = pre_drop_final_shape - final_df.shape[0]
        if duplicates_dropped > 0:
            print(f"[INFO] Dropped {duplicates_dropped} duplicate records in the final DataFrame.")
        print(f"[INFO] Merged existing Parquet DataFrame with tasks DataFrame into final DataFrame with shape {final_df.shape}.")
        debug_print("Final merged DataFrame sample:", final_df)
    except Exception as e:
        print(f"[ERROR] Failed to merge existing Parquet DataFrame with tasks DataFrame: {e}")
        sys.exit(1)

    # Save the final DataFrame to a new Parquet file
    try:
        # Ensure the directory for the final Parquet file exists
        final_parquet_dir = os.path.dirname(final_parquet_file)
        if final_parquet_dir and not os.path.exists(final_parquet_dir):
            os.makedirs(final_parquet_dir, exist_ok=True)
            print(f"[INFO] Created directory for final Parquet file: {final_parquet_dir}")
            debug_print("Created directory for final Parquet file:", final_parquet_dir)

        # Save the final DataFrame
        final_df.to_parquet(final_parquet_file, index=False)
        print(f"[INFO] Saved final merged DataFrame to Parquet file: {final_parquet_file} with shape {final_df.shape}.")
        debug_print("Final DataFrame saved to Parquet:", final_df)
    except Exception as e:
        print(f"[ERROR] Failed to save final DataFrame to Parquet file {final_parquet_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
