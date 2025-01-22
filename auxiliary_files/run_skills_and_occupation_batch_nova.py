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

CONFIG_FILE_PATH = "auxiliary_files/skills_occupation_config.json"
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

params = {
    "paths": {
        "jobstruct_path":         config["jobstruct_path"],
        "extracted_data_path":    config["paths"]["extracted_data_path"],
        "local_folder":           config["paths"]["local_folder"],
        "s3_folder_input_files":  config["paths"]["s3_folder_input_files"],
        "s3_folder_output_files": config["paths"]["s3_folder_output_files"],
    },
    "execution": {
        "CREATE_AND_UPLOAD_INPUT_FILES": config["execution"]["create_and_upload_input_files"],
        "RUN_BEDROCK_JOBS_END_TO_END":   config["execution"]["run_bedrock_jobs_end_to_end"],
        "RECORD_INDEX":                  tuple(config["execution"]["record_index"]),
    },
    "task": {
        "TASK_LIST":       config["task"]["task_list"],
        "TAXONOMY_FILES":  config["task"]["taxonomy_files"],
        "MAX_TOKENS":      config["task"]["max_tokens"],
        "BATCH_SIZES":     config["task"]["batch_sizes"],
    },
    "aws": {
        "region_name":  config["aws"]["region_name"],
        "profile_name": config["aws"]["profile_name"],
        "role_name":    config["aws"]["role_name"],
        "bucket_name":  config["aws"]["bucket_name"],
    },
    "model_id": config["model_id"]
}

jobstruct_path = params["paths"]["jobstruct_path"]
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

from jobstruct import Prompts

class MultiStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

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

def write_and_upload_jsonl(records, file_prefix, bucket_name, s3_folder, s3_client, local_folder, max_records_per_file):
    num_records = len(records)
    file_paths = []
    print(f"[INFO] Starting JSONL file creation/upload for prefix '{file_prefix}'. Total records: {num_records}, max_records_per_file: {max_records_per_file}")
    os.makedirs(local_folder, exist_ok=True)
    for i in tqdm(range(0, num_records, max_records_per_file), desc=f"Processing {file_prefix}"):
        chunk = records[i:i + max_records_per_file]
        start_index = i
        end_index = min(i + max_records_per_file - 1, num_records - 1)
        file_name = f"{file_prefix}_{start_index}-{end_index}.jsonl"
        local_file_path = os.path.join(local_folder, file_name)
        with open(local_file_path, 'w', encoding='utf-8') as f:
            for record in chunk:
                f.write(json.dumps(record) + '\n')
        file_paths.append(local_file_path)
        print(f"    [INFO] Created JSONL file: {file_name} with {len(chunk)} records (indexes {start_index} to {end_index}).")
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
    local_folder = params["paths"]["local_folder"].format(session_number=session_number)
    s3_folder_input_files = params["paths"]["s3_folder_input_files"].format(session_number=session_number)
    s3_folder_output_files = params["paths"]["s3_folder_output_files"].format(session_number=session_number)
    create_and_upload_input_files = params["execution"]["CREATE_AND_UPLOAD_INPUT_FILES"]
    run_bedrock_jobs_end_to_end = params["execution"]["RUN_BEDROCK_JOBS_END_TO_END"]
    record_index = params["execution"]["RECORD_INDEX"]
    task_list = params["task"]["TASK_LIST"]
    taxonomy_files = params["task"]["TAXONOMY_FILES"]
    max_tokens = params["task"]["MAX_TOKENS"]
    batch_sizes = params["task"]["BATCH_SIZES"]
    region_name = params["aws"]["region_name"]
    profile_name = params["aws"]["profile_name"]
    role_name = params["aws"]["role_name"]
    bucket_name = params["aws"]["bucket_name"]
    model_id = params["model_id"]
    extracted_data_path = params["paths"]["extracted_data_path"]
    os.makedirs(local_folder, exist_ok=True)
    log_file_path = os.path.join(local_folder, f"log_{session_number}.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_file_path, "w")
    multi_stream = MultiStream(original_stdout, log_file)
    sys.stdout = multi_stream
    sys.stderr = multi_stream
    try:
        print(f"Session Number: {session_number}")
        print("[INFO] Setting up AWS clients...")
        s3_client = get_boto3_client("s3", profile_name, region_name)
        bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
        role_arn = get_iam_role_arn(role_name)
        print("[INFO] AWS clients initialized.")
        if create_and_upload_input_files:
            print(f"[INFO] Loading extraction data from {extracted_data_path}...")
            amazon_job_data = pd.read_parquet(extracted_data_path)
            amazon_job_data.sort_values(by='metadata_event_utc_timestamp', ascending=False, inplace=True)
            amazon_job_data = amazon_job_data.iloc[record_index[0] : record_index[1]]
            print(f"[INFO] Extraction data loaded. Total records: {len(amazon_job_data)}")
            for task_name, file in zip(task_list, taxonomy_files):
                print(f"[INFO] Starting preparation for {task_name.capitalize()} Task...")
                input_prefix = f"{task_name}_inputs_{session_number}"
                ref_data = None
                if file:
                    with open(file) as f:
                        ref_data = json.load(f)
                    if task_name == "skills":
                        ref_data = build_compact_hierarchy(ref_data)
                records = []
                for index, row in tqdm(amazon_job_data.iterrows(), total=len(amazon_job_data), desc=f"Preparing {task_name.capitalize()} Records"):
                    job_title = clean_text(row['external_title'])
                    details = clean_text("\n".join(row['external_qualifications']))
                    required = clean_text("\n".join(row['basic_qualifications']))
                    preferred = clean_text("\n".join(row['preferred_qualifications']))
                    input_text = "\n\n".join([job_title, details, required, preferred])
                    record = {
                        "recordId": row['job_guid'],
                        "modelInput": {
                            "inferenceConfig": {
                                "max_new_tokens": 512,
                                "temperature": 0,
                                "top_p": 0.95,
                                "top_k": 50
                            },
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "text": getattr(Prompts, task_name).format(
                                                text=input_text,
                                                skills=ref_data if task_name=='skills' else "",
                                                soc_codes=ref_data if task_name=='occupation' else ""
                                            )
                                        }
                                    ]
                                }
                            ],
                            "system": [
                                {
                                    "text": "You are an helpful agent who can select all correct answers and output using given format."
                                }
                            ],
                        }
                    }
                    records.append(record)
                print(f"[INFO] Input preparation for {task_name.capitalize()} Task completed.")
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
        if run_bedrock_jobs_end_to_end:
            print("[INFO] Running Bedrock jobs for all tasks...")
            for task_name in task_list:
                input_prefix = f"{task_name}_inputs_{session_number}"
                output_folder = f"{s3_folder_output_files}{task_name}/"
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_input_files)
                jsonl_keys = [
                    obj['Key']
                    for obj in response.get('Contents', [])
                    if obj['Key'].startswith(f"{s3_folder_input_files}{input_prefix}") and obj['Key'].endswith(".jsonl")
                ]
                print(f"[INFO] Found {len(jsonl_keys)} JSONL files for {task_name.capitalize()} Task.")
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
                        prefix=f"{task_name}-job",
                        session_number=session_number
                    )
                    job_arn = bedrock_response.get('jobArn')
                    print(f"[INFO] Job ARN: {job_arn}")
                print(f"[INFO] All jobs for {task_name.capitalize()} Task have been submitted.")
            print("[INFO] All Bedrock jobs have been submitted.")
    finally:
        log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
