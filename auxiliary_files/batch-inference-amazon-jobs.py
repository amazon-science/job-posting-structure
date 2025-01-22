import sys

jobstruct_path = "/home/fkkarami/workspace/amazon/old_ebsvolume/home/fkkarami/workspace/NASWA/job-posting-structure/src"
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

import os
import sys
import json
import time
import boto3
import pandas as pd
import re
import uuid
from urllib.parse import urlparse
from datetime import datetime
from jobstruct import Prompts
from typing import List, Optional
from tqdm import tqdm  # For progress bars
from pydantic import BaseModel

# ---------------------
# Configuration Flags
# ---------------------
CREATE_AND_UPLOAD_INPUT_FILES = True  # If True, will create JSONL inputs and upload them to S3
RUN_BEDROCK_JOBS_END_TO_END = True    # If True, will create Bedrock jobs for each input JSONL, monitor status, and download/save results

# ----------------------------------------------------------------
# Helper for Generating a Session Number
# ----------------------------------------------------------------
def generate_session_number() -> str:
    """
    Generates a session number based on the current UTC time and a short 5-character UUID portion.
    Example: 20250108-103030-abc12
    """
    current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short_ubid = uuid.uuid4().hex[:5]
    return f"{current_time}-{short_ubid}"

# ----------------------------------------------------------------
# Unique Job Name Generation
# ----------------------------------------------------------------
def generate_unique_job_name(prefix: str = "bedrock-job", session_number: str = None) -> str:
    """
    Generates a unique job name by including:
      - A custom prefix
      - The session number
      - A UTC timestamp
      - A random short UUID
    """
    current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:8]  # Generate a short random identifier
    if session_number:
        return f"{prefix}-{session_number}-{current_time}-{unique_id}"
    else:
        return f"{prefix}-{current_time}-{unique_id}"

# ----------------------------------------------------------------
# Text Cleaning Utility
# ----------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Removing extra spaces between characters.
    - Removing excessive newline characters.
    - Reassembling words if characters are split.
    - Removing leading/trailing whitespace.
    """
    if not text:
        return text
    
    # Replace multiple newline characters with a single space
    text = re.sub(r'\n+', ' ', text)
    
    # Reassemble words by removing spaces between characters but keeping proper word boundaries
    text = re.sub(r'(?<!\s)\s(?!\s)', '', text)  # Removes spaces between letters in words
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    return text.strip()

# ----------------------------------------------------------------
# AWS and Boto3 Utilities
# ----------------------------------------------------------------
def get_boto3_client(service_name, profile_name=None, region_name='us-east-1'):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client(service_name)


def get_iam_role_arn(role_name):
    iam_client = boto3.client('iam')
    return iam_client.get_role(RoleName=role_name)["Role"]["Arn"]


def upload_file_to_s3(file_path, bucket_name, s3_key, s3_client):
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"File uploaded successfully to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise

# ----------------------------------------------------------------
# Writing and Uploading JSONL
# ----------------------------------------------------------------
def write_and_upload_jsonl(
    records, 
    file_prefix, 
    bucket_name, 
    s3_folder, 
    s3_client, 
    local_folder, 
    max_records_per_file=49000
):
    """
    Writes records into chunked JSONL files, keeps local copies, and uploads them to S3.
    """
    num_records = len(records)
    file_paths = []
    
    print(f"Starting JSONL file creation and upload. Total records: {num_records}, "
          f"max records per file: {max_records_per_file}")
    os.makedirs(local_folder, exist_ok=True)  # Ensure local folder exists
    
    for i in tqdm(range(0, num_records, max_records_per_file), desc=f"Processing {file_prefix}"):
        chunk = records[i:i + max_records_per_file]
        file_name = f"{file_prefix}_{(i // max_records_per_file) + 1}.jsonl"
        local_file_path = os.path.join(local_folder, file_name)
        
        # Write JSONL file locally
        with open(local_file_path, 'w') as f:
            for record in chunk:
                f.write(json.dumps(record) + '\n')
        file_paths.append(local_file_path)
        print(f"Created JSONL file: {file_name} with {len(chunk)} records.")
        
        # Upload to S3
        input_s3_key = f"{s3_folder}{file_name}"
        upload_file_to_s3(local_file_path, bucket_name, input_s3_key, s3_client)
    
    print("JSONL file creation and upload completed.")
    return file_paths

# ----------------------------------------------------------------
# Bedrock Job Creation and Monitoring
# ----------------------------------------------------------------
def create_bedrock_job(bedrock_client, role_arn, model_id, input_s3_url, output_s3_url, prefix="bedrock-job", session_number=None):
    """
    Create a Bedrock model invocation job with a unique name that includes session_number.
    """
    input_data_config = {"s3InputDataConfig": {"s3Uri": input_s3_url}}
    output_data_config = {"s3OutputDataConfig": {"s3Uri": output_s3_url}}
    
    return bedrock_client.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model_id,
        jobName=generate_unique_job_name(prefix, session_number=session_number),
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config,
    )


def check_job_status(job_arn, bedrock_client, task=""):
    """
    Blocks until the job is completed, fails, or is stopped.
    """
    while True:
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = response['status']
        print(f"{task} Job Status: {status}")
        if status in ['Completed', 'Failed', 'Stopped']:
            return response
        time.sleep(60)


def download_file_from_s3(s3_url, local_folder, s3_client):
    """
    Download a file from S3 to the specified local folder.
    """
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    file_name = os.path.basename(key)
    local_path = os.path.join(local_folder, file_name)
    os.makedirs(local_folder, exist_ok=True)
    
    s3_client.download_file(bucket_name, key, local_path)
    return local_path


def save_results_to_json(results, output_file):
    """
    Save the given results to a JSON file with indentation.
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


# ----------------------------------------------------------------
# Post-Processing of Completed Jobs
# ----------------------------------------------------------------
def post_process_bedrock_jobs(task_names, bucket_name, s3_folder_output_files, local_folder, s3_client, bedrock_client):
    """
    Post-process completed Bedrock jobs by listing the .out files in S3, 
    downloading them, and saving the combined results as JSON.
    """
    print("Starting post-processing of Bedrock jobs...")

    for task_name in task_names:
        print(f"Processing completed jobs for {task_name.capitalize()} Task...")
        output_folder = f"{s3_folder_output_files}{task_name}/"

        # List JSONL output files in the S3 output folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=output_folder)
        contents = response.get('Contents', [])
        output_files = [obj['Key'] for obj in contents if obj['Key'].endswith(".out")]
        
        print(f"Found {len(output_files)} output files for {task_name.capitalize()} Task.")

        for output_s3_key in tqdm(output_files, desc=f"Downloading results for {task_name.capitalize()}"):
            output_s3_url = f"s3://{bucket_name}/{output_s3_key}"

            # Download the result file from S3
            local_output_file = download_file_from_s3(output_s3_url, local_folder, s3_client)

            # Process and save the results locally
            print(f"Processing results from {output_s3_url}...")
            with open(local_output_file, 'r') as f:
                results = [json.loads(line) for line in f]

            # Save the processed results to a consolidated JSON file
            local_results_file = os.path.join(local_folder, f"{task_name}_results.json")
            save_results_to_json(results, local_results_file)

        print(f"Post-processing completed for {task_name.capitalize()} Task.")

    print("All post-processing steps completed.")


# ----------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Generate a session number to tag all resources
    session_number = generate_session_number()
    print(f"Session Number: {session_number}")
    
    # AWS Configurations
    region_name = 'us-east-1'
    profile_name = 'pssl-bedrock'
    role_name = "BedrockPermissionsRole"
    bucket_name = "fkkarami-projects"
    
    # Incorporate session_number into the S3 folders and local folder
    s3_folder_input_files = f'bedrock-batch-inference/{session_number}/input-amz-jobs/'
    s3_folder_output_files = f'bedrock-batch-inference/{session_number}/output-amz-jobs/'
    local_folder = f"tests/amazon-jobs-data/{session_number}/"

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    batch_size = 8000

    # Preloaded extraction data file path
    extracted_data_path = "tests/data_deduplicated.parquet"  # Replace with actual file path

    # Initialize AWS Clients
    print("Setting up AWS clients...")
    s3_client = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn = get_iam_role_arn(role_name)
    print("AWS clients initialized.")

    # Load extraction data
    print(f"Loading extraction data from {extracted_data_path}...")
    amazon_job_data = pd.read_parquet(extracted_data_path)
    amazon_job_data = amazon_job_data.iloc[100000:200000]  # Optional: Limit records for testing
    print(f"Extraction data loaded. Total records: {len(amazon_job_data)}")

    # ------------------------------------------------
    # Prepare and Upload JSONL for Skills & Occupation
    # ------------------------------------------------
    for task_name, taxonomy_file in [
        ("skills",      "tests/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json"),
        # ("occupation",  None)
    ]:
        print(f"Starting preparation for {task_name.capitalize()} Task...")
        
        # Incorporate session_number into the JSONL file prefix
        input_prefix = f"{task_name}_inputs_{session_number}"
        
        # Load taxonomy if applicable
        taxonomy = None
        if taxonomy_file:
            with open(taxonomy_file) as f:
                taxonomy = json.load(f)
        
        # Prepare input records for the task
        records = []
        for _, record in tqdm(amazon_job_data.iterrows(), total=len(amazon_job_data), desc=f"Preparing {task_name.capitalize()} Records"):
            model_input = {}
            job_title   = clean_text(record['external_title'])
            details     = clean_text("\n".join(record['external_qualifications']))
            required    = clean_text("\n".join(record['basic_qualifications']))
            preferred   = clean_text("\n".join(record['preferred_qualifications']))
            
            input_text = "\n\n".join([
                job_title,
                details,
                required,
                preferred,
            ])
            
            model_input['messages'] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": getattr(Prompts, task_name).format(
                                text=input_text, 
                                skills=taxonomy if taxonomy else ""
                            )
                        }
                    ]
                }
            ]
            records.append({"recordId": record['job_guid'], "modelInput": model_input})
        
        print(f"Input preparation for {task_name.capitalize()} Task completed.")
        
        # Write, upload, and keep local copies of JSONL files
        if CREATE_AND_UPLOAD_INPUT_FILES:
            print(f"Creating and uploading JSONL files for {task_name.capitalize()} Task...")
            write_and_upload_jsonl(
                records, 
                file_prefix=input_prefix, 
                bucket_name=bucket_name,
                s3_folder=s3_folder_input_files,
                s3_client=s3_client,
                local_folder=local_folder,
                max_records_per_file=batch_size
            )

    # ------------------------------------------------
    # Submit Bedrock Jobs (without waiting for completion)
    # ------------------------------------------------
    if RUN_BEDROCK_JOBS_END_TO_END:
        print("Running Bedrock jobs for all tasks...")

        # Process JSONL files from S3 for both Skills and Occupation tasks
        for task_name in ["skills", 
        # "occupation"
        ]:
            # Match the prefix used above: e.g. "skills_inputs_<session_number>"
            input_prefix = f"{task_name}_inputs_{session_number}"
            output_folder = f"{s3_folder_output_files}{task_name}/"

            # List JSONL files in the S3 input folder
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_input_files)
            jsonl_keys = [
                obj['Key'] 
                for obj in response.get('Contents', []) 
                if obj['Key'].startswith(f"{s3_folder_input_files}{input_prefix}") and obj['Key'].endswith(".jsonl")
            ]
            print(f"Found {len(jsonl_keys)} JSONL files for {task_name.capitalize()} Task.")

            # Run Bedrock jobs for each JSONL file
            for jsonl_s3_key in tqdm(jsonl_keys, desc=f"Starting Bedrock Jobs for {task_name.capitalize()}"):
                jsonl_s3_url = f"s3://{bucket_name}/{jsonl_s3_key}"
                output_s3_url = f"s3://{bucket_name}/{output_folder}"

                # Create Bedrock job using session_number in the name
                print(f"Submitting Bedrock job for file: {jsonl_s3_key}")
                bedrock_response = create_bedrock_job(
                    bedrock_client=bedrock_client, 
                    role_arn=role_arn, 
                    model_id=model_id, 
                    input_s3_url=jsonl_s3_url, 
                    output_s3_url=output_s3_url,
                    prefix=f"{task_name}-job",    # e.g., "skills-job"
                    session_number=session_number # ensures uniqueness
                )
                job_arn = bedrock_response.get('jobArn')
                print(f"Job ARN: {job_arn}")

            print(f"All jobs for {task_name.capitalize()} Task have been submitted.")

        print("All Bedrock jobs have been submitted.")



        # print("Starting post-processing...")
        # task_names = ["skills", "occupation"]
        # post_process_bedrock_jobs(task_names, bucket_name, s3_folder_output_files, local_folder, s3_client, bedrock_client)

