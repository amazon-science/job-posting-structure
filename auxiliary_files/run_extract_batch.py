import sys
import os
import json
import time
import boto3
import pandas as pd
import re
import uuid
from urllib.parse import urlparse
from datetime import datetime

from typing import List, Optional
from tqdm import tqdm  

CONFIG_FILE_PATH = "extract_config.json"
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

# If needed, add jobstruct_path to sys.path
jobstruct_path = config.get("jobstruct_path", "")
if jobstruct_path and jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

# Now import your prompts
from jobstruct import Prompts

# Execution-related values from config
CREATE_AND_UPLOAD_INPUT_FILES = config["execution"]["create_and_upload_input_files"]
RUN_BEDROCK_JOBS_END_TO_END   = config["execution"]["run_bedrock_jobs_end_to_end"]
RECORD_INDEX                  = tuple(config["execution"]["record_index"])

# The 'task' object in config: we expect "extract" to be in this list.
TASK_LIST      = config["task"]["task_list"]      
TAXONOMY_FILES = config["task"].get("taxonomy_files", [])  
MAX_TOKENS     = config["task"]["max_tokens"]   
batch_sizes    = config["task"]["batch_sizes"]  

# AWS details
region_name  = config["aws"]["region_name"]
profile_name = config["aws"]["profile_name"]
role_name    = config["aws"]["role_name"]
bucket_name  = config["aws"]["bucket_name"]

# Models
model_id = config["model_id"]  # e.g., {"extract": "<your-model-id>"}

# Input parquet location
data_path = config["paths"]["data_path"]

def generate_session_number() -> str:
    current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short_ubid = uuid.uuid4().hex[:5]
    return f"{current_time}-{short_ubid}"

def generate_unique_job_name(prefix: str = "bedrock-job", session_number: str = None) -> str:
    current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    if session_number:
        return f"{prefix}-{session_number}-{current_time}-{unique_id}"
    else:
        return f"{prefix}-{current_time}-{unique_id}"

def clean_text(text: str) -> str:
    if not text:
        return text
    # Remove extra newlines, spaces, etc.
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'(?<!\s)\s(?!\s)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_boto3_client(service_name, profile_name=None, region_name='us-east-1'):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client(service_name)

def get_iam_role_arn(role_name):
    iam_client = boto3.client('iam')
    return iam_client.get_role(RoleName=role_name)["Role"]["Arn"]

def upload_file_to_s3(file_path, bucket_name, s3_key, s3_client):
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"[INFO] File uploaded successfully to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise

def write_and_upload_jsonl(
    records, 
    file_prefix, 
    bucket_name, 
    s3_folder, 
    s3_client, 
    local_folder, 
    max_records_per_file
):
    """
    Creates JSONL files from `records` (a list of dicts) and uploads them to S3 in chunks of `max_records_per_file`.
    """
    num_records = len(records)
    file_paths = []

    print(f"[INFO] Starting JSONL creation for '{file_prefix}'. "
          f"Total records: {num_records}, max_records_per_file: {max_records_per_file}")
    os.makedirs(local_folder, exist_ok=True)

    for i in tqdm(range(0, num_records, max_records_per_file), desc=f"Processing {file_prefix}"):
        chunk = records[i:i + max_records_per_file]
        file_name = f"{file_prefix}_{(i // max_records_per_file) + 1}.jsonl"
        local_file_path = os.path.join(local_folder, file_name)

        with open(local_file_path, 'w', encoding='utf-8') as f:
            for record in chunk:
                f.write(json.dumps(record) + '\n')

        file_paths.append(local_file_path)
        print(f"    [INFO] Created JSONL file: {file_name} with {len(chunk)} records.")

        file_size = os.path.getsize(local_file_path)
        print(f"    [VERBOSE] File size: {file_size / (1024 ** 3):.2f} GB. Path: {local_file_path}")

        # Upload to S3
        input_s3_key = f"{s3_folder}{file_name}"
        upload_file_to_s3(local_file_path, bucket_name, input_s3_key, s3_client)

    print("[INFO] JSONL file creation and upload completed.\n")
    return file_paths

def create_bedrock_job(bedrock_client, role_arn, model_id, input_s3_url, output_s3_url, prefix="bedrock-job", session_number=None):
    input_data_config = {"s3InputDataConfig": {"s3Uri": input_s3_url}}
    output_data_config = {"s3OutputDataConfig": {"s3Uri": output_s3_url}}
    return bedrock_client.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model_id,
        jobName=generate_unique_job_name(prefix, session_number=session_number),
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config,
    )


if __name__ == "__main__":
    session_number = generate_session_number()
    print(f"Session Number: {session_number}")

    # Format folder paths that include {session_number} in config
    local_folder           = config["paths"]["local_folder"].format(session_number=session_number)
    s3_folder_input_files  = config["paths"]["s3_folder_input_files"].format(session_number=session_number)
    s3_folder_output_files = config["paths"]["s3_folder_output_files"].format(session_number=session_number)

    print("[INFO] Setting up AWS clients...")
    s3_client      = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn       = get_iam_role_arn(role_name)
    print("[INFO] AWS clients initialized.\n")


    if CREATE_AND_UPLOAD_INPUT_FILES:
        print(f"[INFO] Loading Parquet data from {data_path}...")
        df = pd.read_parquet(data_path)
        df = df.iloc[RECORD_INDEX[0]:RECORD_INDEX[1]]
        print(f"[INFO] Data loaded, total records after slicing: {len(df)}.\n")

        for task_name, _taxonomy_file in zip(TASK_LIST, TAXONOMY_FILES):
            # Because we only have one column of unstructured text, no need for taxonomy
            print(f"[INFO] Starting preparation for {task_name.upper()} Task...")

            input_prefix = f"{task_name}_inputs_{session_number}"
            records = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing {task_name.upper()} Records"):
                job_guid = row.get('job_guid', str(uuid.uuid4()))
                unstructured_text = row.get('job_description', '')

                # Build the model input object
                model_input = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS[task_name],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": getattr(Prompts, task_name).format(text=unstructured_text)
                                }
                            ]
                        }
                    ]
                }

                records.append({
                    "recordId": job_guid,
                    "modelInput": model_input
                })

            print(f"[INFO] Finished building input records for {task_name.upper()} (count={len(records)}).")
            print(f"[INFO] Creating & uploading JSONL files for {task_name.upper()}...")

            write_and_upload_jsonl(
                records=records,
                file_prefix=input_prefix,
                bucket_name=bucket_name,
                s3_folder=s3_folder_input_files,
                s3_client=s3_client,
                local_folder=local_folder,
                max_records_per_file=batch_sizes[task_name]
            )


    if RUN_BEDROCK_JOBS_END_TO_END:
        print("[INFO] Submitting Bedrock jobs for all tasks...\n")

        for task_name in TASK_LIST:
            input_prefix  = f"{task_name}_inputs_{session_number}"
            output_folder = f"{s3_folder_output_files}{task_name}/"

            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_input_files)
            if 'Contents' not in response:
                print(f"[WARNING] No objects found in s3://{bucket_name}/{s3_folder_input_files}. Skipping.")
                continue

            jsonl_keys = [
                obj['Key']
                for obj in response['Contents']
                if obj['Key'].startswith(f"{s3_folder_input_files}{input_prefix}") and obj['Key'].endswith(".jsonl")
            ]
            print(f"[INFO] Found {len(jsonl_keys)} JSONL files for {task_name.upper()} Task.")

            for jsonl_s3_key in tqdm(jsonl_keys, desc=f"Starting Bedrock Jobs for {task_name.upper()}"):
                jsonl_s3_url  = f"s3://{bucket_name}/{jsonl_s3_key}"
                output_s3_url = f"s3://{bucket_name}/{output_folder}"
                
                print(f"[INFO] Submitting job for file: {jsonl_s3_key}")
                bedrock_response = create_bedrock_job(
                    bedrock_client=bedrock_client,
                    role_arn=role_arn,
                    model_id=model_id[task_name],
                    input_s3_url=jsonl_s3_url,
                    output_s3_url=output_s3_url,
                    prefix=f"{task_name}-job",
                    session_number=session_number
                )
                job_arn = bedrock_response.get('jobArn')
                print(f"       [INFO] Job ARN: {job_arn}")

            print(f"[INFO] All jobs for {task_name.upper()} Task have been submitted.\n")

        print("[INFO] All Bedrock jobs submitted. End of script.")
