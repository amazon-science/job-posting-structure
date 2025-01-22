import sys

jobstruct_path = "/home/fkkarami/workspace/amazon/old_ebsvolume/home/fkkarami/workspace/NASWA/job-posting-structure/src"
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

import os
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


S3_PRECOMPUTED_FOLDER = False # If True, will process precomputed JSONL files from S3
CREATE_AND_UPLOAD_INPUT_FILES = True  # If True, will create JSONL inputs and upload them to S3
RUN_BEDROCK_JOBS_END_TO_END = True   # If True, will create Bedrock jobs for each input JSONL, monitor status, and download/save results
SESSION_NUMBER = '20250110-201626-1bcf2'
RECORD_INDEX = (160000, 200000)
TASK_LIST = [
    'skills', 
    'occupation'
    ]
TAXONOMY_FILES = [
    "auxiliary_files/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json", 
    None
    ]

MAX_TOKENS = {"skills": 1024, "occupation": 128}

# Batch size per task
batch_sizes = {
    "skills": 20000,
    "occupation": 40000
}


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

# ----------------------------------------------------------------
# Text Cleaning Utility
# ----------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return text
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
    num_records = len(records)
    file_paths = []

    print(f"[INFO] Starting JSONL file creation/upload for prefix '{file_prefix}'. "
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

        input_s3_key = f"{s3_folder}{file_name}"
        upload_file_to_s3(local_file_path, bucket_name, input_s3_key, s3_client)

    print("[INFO] JSONL file creation and upload completed.")
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


def build_compact_hierarchy(node: dict) -> dict:
    children = node.get("children", [])
    if not children:
        return {}
    result = {}
    for child in children:
        child_name = child["name"]
        result[child_name] = build_compact_hierarchy(child)
    return result


if __name__ == "__main__":
    session_number = generate_session_number()
    print(f"Session Number: {session_number}")
    
    region_name = 'us-east-1'
    profile_name = 'pssl-bedrock'
    role_name = "BedrockPermissionsRole"
    bucket_name = "fkkarami-projects"

    s3_folder_input_files = f'bedrock-batch-inference/{session_number}/input-amz-jobs/'
    s3_folder_output_files = f'bedrock-batch-inference/{session_number}/output-amz-jobs/'
    local_folder = f"auxiliary_files/amazon-jobs-data/{session_number}/"



    print("[INFO] Setting up AWS clients...")
    s3_client = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn = get_iam_role_arn(role_name)
    print("[INFO] AWS clients initialized.")

    if CREATE_AND_UPLOAD_INPUT_FILES:


        model_id = {
            "skills": "anthropic.claude-3-sonnet-20240229-v1:0",
            "occupation": "anthropic.claude-3-sonnet-20240229-v1:0"
        }

        extracted_data_path = "auxiliary_files/data_deduplicated.parquet"
        print(f"[INFO] Loading extraction data from {extracted_data_path}...")
        amazon_job_data = pd.read_parquet(extracted_data_path)
        amazon_job_data = amazon_job_data.iloc[RECORD_INDEX[0]:RECORD_INDEX[1]]
        print(f"[INFO] Extraction data loaded. Total records: {len(amazon_job_data)}")

        for task_name, taxonomy_file in zip(TASK_LIST, TAXONOMY_FILES):
            print(f"[INFO] Starting preparation for {task_name.capitalize()} Task...")
            
            input_prefix = f"{task_name}_inputs_{session_number}"
            
            taxonomy = None
            if taxonomy_file:
                with open(taxonomy_file) as f:
                    taxonomy = json.load(f)
                taxonomy = build_compact_hierarchy(taxonomy)

            records = []
            for _, record in tqdm(amazon_job_data.iterrows(), total=len(amazon_job_data), desc=f"Preparing {task_name.capitalize()} Records"):
                model_input = {}
                model_input["anthropic_version"] = "bedrock-2023-05-31"
                model_input["max_tokens"] = MAX_TOKENS[task_name]
                job_title = clean_text(record['external_title'])
                details = clean_text("\n".join(record['external_qualifications']))
                required = clean_text("\n".join(record['basic_qualifications']))
                preferred = clean_text("\n".join(record['preferred_qualifications']))
                
                input_text = "\n\n".join([job_title, details, required, preferred])
                model_input['messages'] = [
                    {"role": "user", "content": [
                        {"type": "text", "text": getattr(Prompts, task_name).format(
                            text=input_text, 
                            skills=taxonomy if taxonomy else ""
                        )}
                    ]}
                ]
                records.append({"recordId": record['job_guid'], "modelInput": model_input})

            print(f"[INFO] Input preparation for {task_name.capitalize()} Task completed.")
            
            # if CREATE_AND_UPLOAD_INPUT_FILES:
            print(f"[INFO] Creating and uploading JSONL files for {task_name.capitalize()} Task...")
            write_and_upload_jsonl(
                records, 
                file_prefix=input_prefix, 
                bucket_name=bucket_name,
                s3_folder=s3_folder_input_files,
                s3_client=s3_client,
                local_folder=local_folder,
                max_records_per_file=batch_sizes[task_name]
            )


    if RUN_BEDROCK_JOBS_END_TO_END:
        print("[INFO] Running Bedrock jobs for all tasks...")

        for task_name in TASK_LIST:
            if S3_PRECOMPUTED_FOLDER:
                session_number = SESSION_NUMBER
            else:
                # e.g. "skills_inputs_<session_number>"
                input_prefix = f"{task_name}_inputs_{session_number}"
            output_folder = f"{s3_folder_output_files}{task_name}/"

            # List JSONL files in the S3 input folder
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_input_files)
            jsonl_keys = [
                obj['Key'] 
                for obj in response.get('Contents', []) 
                if obj['Key'].startswith(f"{s3_folder_input_files}{input_prefix}") and obj['Key'].endswith(".jsonl")
            ]
            print(f"[INFO] Found {len(jsonl_keys)} JSONL files for {task_name.capitalize()} Task.")

            # Run Bedrock jobs for each JSONL file
            for jsonl_s3_key in tqdm(jsonl_keys, desc=f"Starting Bedrock Jobs for {task_name.capitalize()}"):
                jsonl_s3_url = f"s3://{bucket_name}/{jsonl_s3_key}"
                output_s3_url = f"s3://{bucket_name}/{output_folder}"

                print(f"[INFO] Submitting Bedrock job for file: {jsonl_s3_key}")
                bedrock_response = create_bedrock_job(
                    bedrock_client=bedrock_client, 
                    role_arn=role_arn, 
                    model_id=model_id[task_name], 
                    input_s3_url=jsonl_s3_url, 
                    output_s3_url=output_s3_url,
                    prefix=f"{task_name}-job",    # e.g., "skills-job"
                    session_number=session_number
                )
                job_arn = bedrock_response.get('jobArn')
                print(f"[INFO] Job ARN: {job_arn}")

            print(f"[INFO] All jobs for {task_name.capitalize()} Task have been submitted.")

        print("[INFO] All Bedrock jobs have been submitted.")

        # Example of how to do post-processing once jobs are complete:
        # tasks_to_postprocess = ["skills"]  # or ["skills","occupation"]
        # post_process_bedrock_jobs(
        #     task_names=tasks_to_postprocess,
        #     bucket_name=bucket_name,
        #     s3_folder_output_files=s3_folder_output_files,
        #     local_folder=local_folder,
        #     s3_client=s3_client,
        #     bedrock_client=bedrock_client
        # )
