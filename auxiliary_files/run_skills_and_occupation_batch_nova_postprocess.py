import sys
import os
import json
import boto3
from tqdm import tqdm
from typing import Optional
import re
import pandas as pd  # Import pandas for DataFrame operations


CONFIG_FILE_PATH = "auxiliary_files/skills_occupation_config.json"
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

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
    return config

def get_boto3_client(service_name: str, profile_name: Optional[str] = None, region_name: str = 'us-east-1'):
    """
    Initialize a boto3 client for the specified AWS service.

    Args:
        service_name (str): Name of the AWS service (e.g., 's3').
        profile_name (Optional[str]): AWS profile name. If None, default profile is used.
        region_name (str): AWS region name.

    Returns:
        boto3.client: Boto3 client for the specified service.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client(service_name)

def download_s3_folder(s3_client, bucket_name: str, s3_folder: str, local_folder: str):
    """
    Download all objects from a specified S3 folder to a local directory.

    Args:
        s3_client (boto3.client): Initialized boto3 S3 client.
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): S3 folder path (prefix) to download from.
        local_folder (str): Local directory to download the files to.
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

    # Collect all object keys
    object_keys = []
    for page in pages:
        contents = page.get('Contents', [])
        for obj in contents:
            object_keys.append(obj['Key'])

    if not object_keys:
        print(f"No objects found in s3://{bucket_name}/{s3_folder}")
        return

    print(f"Found {len(object_keys)} objects in s3://{bucket_name}/{s3_folder}. Starting download...")

    os.makedirs(local_folder, exist_ok=True)

    for key in tqdm(object_keys, desc="Downloading files"):
        # Determine the relative path to maintain folder structure
        relative_path = os.path.relpath(key, s3_folder)
        local_path = os.path.join(local_folder, relative_path)
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        try:
            s3_client.download_file(bucket_name, key, local_path)
        except Exception as e:
            print(f"Error downloading {key}: {e}")

    print(f"All files downloaded to {local_folder}")

def strip_code_fences(value: str) -> str:
    """
    Removes any ``` or ```json fences (and whitespace) from the string.

    Args:
        value (str): The raw string containing code fences.

    Returns:
        str: The cleaned string without code fences.
    """
    # Remove occurrences of ```json, ``` or ```
    stripped = re.sub(r'```(?:json)?\s*', '', value)
    return stripped.strip()

def post_process_occupation_results(raw_results: str) -> list:
    """
    Processes the raw_results string by removing code fences and extracting the 'occupation' data.

    Args:
        raw_results (str): The raw string containing JSON data, possibly wrapped in code fences.

    Returns:
        list: A list of occupations extracted from the JSON data.
    """
    try:
        # 1. Remove code fences
        clean_value = strip_code_fences(raw_results)
        # 2. Parse the cleaned JSON
        parsed_value = json.loads(clean_value)

        # 3. Extract the 'occupation' field if it exists
        if isinstance(parsed_value, dict) and "occupation" in parsed_value:
            return parsed_value["occupation"]
        else:
            print("No 'occupation' field found in the parsed JSON.")
            return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse raw_results: {e}")
        return []

def process_output_files(local_folder: str):
    """
    Process downloaded .jsonl.out files to extract task-specific data and save as CSV.

    Args:
        local_folder (str): Path to the local directory containing downloaded files.
    """
    # Compile a regex pattern to extract JSON content within triple backticks
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)

    # Traverse the local_folder to find all .jsonl.out files
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            if file.endswith('.jsonl.out'):
                file_path = os.path.join(root, file)
                print(f"[INFO] Processing file: {file_path}")

                records = []
                try:
                    # Determine the task based on the file name
                    # Assuming the task is the first part of the file name before '_'
                    if 'skills' in file.lower():
                        task_name = 'skills'
                        print(f"[INFO] Detected task: {task_name}")
                    elif 'occupation' in file.lower():
                        task_name = 'occupation'
                        print(f"[INFO] Detected task: {task_name}")
                    else:
                        print(f"[WARNING] Unknown task for file {file}. Skipping file.")
                        continue

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_number, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue  # Skip empty lines
                            try:
                                data = json.loads(line)
                                record_id = data.get('recordId', '')
                                model_output = data.get('modelOutput', {})
                                output = model_output.get('output', {})
                                message = output.get('message', {})
                                content_list = message.get('content', [])

                                # Assuming content_list is a list with one dict containing 'text'
                                if content_list and isinstance(content_list, list):
                                    content_text = content_list[0].get('text', '')

                                    if task_name == 'occupation':
                                        # Extract JSON within triple backticks and process
                                        match = json_pattern.search(content_text)
                                        if match:
                                            json_str = match.group(1)
                                            # Parse the extracted JSON string
                                            json_data = json.loads(json_str)
                                            # Extract the 'occupation' data
                                            occupation = post_process_occupation_results(content_text)
                                            records.append({
                                                'job_guid': record_id,
                                                'occupation': occupation
                                            })
                                        else:
                                            print(f"[WARNING] No JSON content found in line {line_number} of {file_path}")
                                    elif task_name == 'skills':
                                        # Directly map recordId to text
                                        skills_text = content_text
                                        records.append({
                                            'job_guid': record_id,
                                            'skills': skills_text
                                        })
                                else:
                                    print(f"[WARNING] Unexpected content format in line {line_number} of {file_path}")
                            except json.JSONDecodeError as jde:
                                print(f"[ERROR] JSON decode error in file {file_path} on line {line_number}: {jde}")
                            except Exception as e:
                                print(f"[ERROR] Unexpected error processing line {line_number} in {file_path}: {e}")

                    if records:
                        # Create DataFrame
                        df = pd.DataFrame(records)
                        # Define CSV file path
                        base_name = os.path.splitext(os.path.splitext(file)[0])[0]  # Remove both .jsonl and .out
                        csv_file_name = f"{base_name}.csv"
                        csv_file_path = os.path.join(root, csv_file_name)
                        # Save DataFrame to CSV
                        df.to_csv(csv_file_path, index=False)
                        print(f"[INFO] Saved CSV file: {csv_file_path}")
                    else:
                        print(f"[INFO] No valid records found in {file_path}. Skipping CSV creation.")

                except Exception as e:
                    print(f"[ERROR] Failed to process file {file_path}: {e}")

# ----------------------- Main Execution -----------------------

def main():
    # Use the fixed session_id as per the original script
    session_id = '20250122-171108-81fb0'

    # Extract necessary parameters from config
    try:
        aws_config = config["aws"]
        s3_bucket = aws_config["bucket_name"]
        aws_profile = aws_config.get("profile_name")
        aws_region = aws_config.get("region_name", "us-east-1")

        # Construct S3 output folder path based on session ID
        s3_folder_output_files = config["paths"]["s3_folder_output_files"].format(session_number=session_id)

        # Define local folder path for downloads
        local_output_folder = config["paths"].get("local_output_folder", "downloaded_outputs/{session_number}")
        local_output_folder = local_output_folder.format(session_number=session_id)
    except KeyError as e:
        print(f"[ERROR] Missing configuration parameter: {e}")
        sys.exit(1)

    # Initialize S3 client
    try:
        s3_client = get_boto3_client("s3", profile_name=aws_profile, region_name=aws_region)
    except Exception as e:
        print(f"[ERROR] Error initializing S3 client: {e}")
        sys.exit(1)

    # Start downloading
    download_s3_folder(s3_client, s3_bucket, s3_folder_output_files, local_output_folder)

    # Process downloaded files
    process_output_files(local_output_folder)

if __name__ == "__main__":
    main()

