import sys
import os
import json
import random
import ast
import time
import boto3
import pandas as pd
import hashlib
from urllib.parse import urlparse
from datetime import datetime
from jobstruct import Prompts, JobStructAI
from importlib import resources
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any, Union

def get_boto3_client(service_name, profile_name=None, region_name='us-east-1'):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client(service_name)


def generate_unique_job_name(prefix="bedrock-job"):
    current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{current_time}"


def get_iam_role_arn(role_name):
    iam_client = boto3.client('iam')
    return iam_client.get_role(RoleName=role_name)["Role"]["Arn"]


def expand_parquet_file_with_id(parquet_path, target_records):
    df = pd.read_parquet(parquet_path)
    repeat_times = (target_records // len(df)) + 1
    expanded_df = pd.concat([df] * repeat_times, ignore_index=True)
    expanded_df = expanded_df.iloc[:target_records]
    expanded_df['new_id'] = [random.randint(100000000, 999999999) for _ in range(target_records)]
    return expanded_df


def create_jsonl_for_extract(df, output_file, task_name, prompt_config, skills="", start_idx=0):
    """
    Create a JSONL file for the extract step.
    """
    with open(output_file, 'w') as file:
        for idx, row in df.iterrows():
            model_input = prompt_config.copy()
            text = row['description']
            id_ = row['new_id']

            model_input['messages'] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": getattr(Prompts, task_name).format(text=text, skills=skills)
                        }
                    ]
                }
            ]
            record = {
                "recordId": id_,
                "modelInput": model_input
            }
            file.write(json.dumps(record) + '\n')


def create_jsonl_for_skills_or_occupation(records, output_file, task_name, prompt_config, skills=""):
    with open(output_file, 'w') as file:
        for idx, record in records.items():
            model_input = prompt_config.copy()
            job_title = record.get("job_title", "")
            details = record.get("details", [])
            required = record.get("required", {})
            preferred = record.get("preferred", {})

            input_text = "\n\n".join([
                job_title or "",
                "\n".join(details or []),
                "\n".join(required.get("qualifications", []) or []),
                "\n".join(preferred.get("qualifications", []) or []),
            ])

            model_input['messages'] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": getattr(Prompts, task_name).format(text=input_text, skills=skills)
                        }
                    ]
                }
            ]
            record = {
                "recordId": idx,
                "modelInput": model_input
            }
            file.write(json.dumps(record) + '\n')


def upload_file_to_s3(file_path, bucket_name, s3_key, s3_client):
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"File uploaded successfully to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise


def create_bedrock_job(bedrock_client, role_arn, model_id, input_s3_url, output_s3_url, prefix="bedrock-job"):
    input_data_config = {"s3InputDataConfig": {"s3Uri": input_s3_url}}
    output_data_config = {"s3OutputDataConfig": {"s3Uri": output_s3_url}}
    return bedrock_client.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model_id,
        jobName=generate_unique_job_name(prefix),
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config,
    )


def check_job_status(job_arn, bedrock_client, task = ""):
    while True:
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = response['status']
        print(f"{task} Job Status: {status}")
        if status in ['Completed', 'Failed', 'Stopped']:
            return response
        time.sleep(60)


def download_file_from_s3(s3_url, local_folder, s3_client):
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    file_name = os.path.basename(key)
    local_path = os.path.join(local_folder, file_name)
    os.makedirs(local_folder, exist_ok=True)
    s3_client.download_file(bucket_name, key, local_path)
    return local_path


def wrap_json_file_with_array(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    wrapped_content = f"[{content.strip()}]".replace('}\n{', '},{')
    return json.loads(wrapped_content)


def save_results_to_json(results, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def extract_occupation(data):
    try:
        data_json = json.loads(data)
    except json.JSONDecodeError:
        data_json = ast.literal_eval(data)
    return data_json['occupation']


def extract_skills(data_str):
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        data = ast.literal_eval(data_str)

    skills = []
    if isinstance(data, dict):
        # Dictionary format: keys map to list of skills
        for v in data.values():
            if isinstance(v, list):
                skills.extend(v)
    elif isinstance(data, list):
        if all(isinstance(item, dict) and 'skills' in item for item in data):
            # Format: list of dicts, each with 'skills' key
            for item in data:
                skills.extend(item.get('skills', []))
        elif all(isinstance(item, str) for item in data):
            # Format: list of strings
            skills.extend(data)
    # Remove duplicates
    return list(set(skills))


class ExtractQualifications(BaseModel):
    education: Optional[str] = None
    major: Optional[List[str]] = []
    experience: Optional[int] = None
    qualifications: Optional[List[str]] = []

class ExtractOutputModel(BaseModel):
    job_title: Optional[str] = None
    details: Optional[List[str]] = []
    required: ExtractQualifications = ExtractQualifications()
    preferred: ExtractQualifications = ExtractQualifications()
    benefits: Optional[List[str]] = []
    salary: Optional[List[float]] = []
    wage: Optional[List[float]] = []
    entry_level: Optional[bool] = False
    college_degree: Optional[bool] = False
    full_time: Optional[bool] = False
    remote: Optional[bool] = False

class OccupationOutputModel(BaseModel):
    occupation: List[str]

class SkillsOutputModel(BaseModel):
    skills: List[str]


def load_batch_output(output_records, task_name):
    results = {}
    invalid_records = []
    for record in output_records:
        body = None
        try:
            body = record["modelOutput"]["content"][0]["text"]
            if isinstance(body, str):
                body = body.strip()

            if task_name == "extract":
                parsed_data = Prompts.safe_json(body, {})
                validated = ExtractOutputModel(**parsed_data)
                results[record["recordId"]] = validated.dict()

            elif task_name == "occupation":
                occupation_data = extract_occupation(body)
                validated = OccupationOutputModel(occupation=occupation_data)
                results[record["recordId"]] = validated.dict()

            elif task_name == "skills":
                skill_list = extract_skills(body)
                validated = SkillsOutputModel(skills=skill_list)
                results[record["recordId"]] = validated.dict()
            else:
                return

        except Exception as e:
            print(body)
            print(f"Error processing record: {e}")
            invalid_records.append({
                "recordId": record.get("recordId"),
                "body": body,
                "error": str(e)
            })

    if invalid_records:
        invalid_file = f"{task_name}_invalid_records.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_records, f, indent=4)
        print(f"Invalid records saved to {invalid_file}")

    return results


# -------------------------
# Cache Handling and Batching for Extract Step
# -------------------------

# Example cache: For demonstration, start empty. In a real scenario, load from DB or file.
cache = {}  # key: job_id, value: job_description_hash

def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def filter_and_prepare_extract_batch(df, cache, batch_size=1000):
    """
    Filters the DataFrame based on the cache and prepares batches for processing.

    Args:
        df (pd.DataFrame): Input DataFrame containing job data.
        cache (dict): Cache with job_id as keys and job_description_hash as values.
        batch_size (int): Size of each batch.

    Returns:
        List[pd.DataFrame]: List of filtered and batched DataFrames.
    """
    # Total number of jobs before filtering
    total_jobs = len(df)
    print(f"Total jobs before filtering: {total_jobs}")

    # 1. Filter out jobs whose job_id is in cache
    df = df[~df['job_id'].isin(cache.keys())]
    jobs_after_id_filter = len(df)
    print(f"Jobs after filtering job_id in cache: {jobs_after_id_filter} (Filtered out: {total_jobs - jobs_after_id_filter})")

    # 2. Compute job_description_hash for remaining
    df['job_description_hash'] = df['description'].apply(compute_sha256)

    # 3. Filter out records whose hash already exists in cache values
    existing_hashes = set(cache.values())
    df = df[~df['job_description_hash'].isin(existing_hashes)]
    jobs_after_hash_filter = len(df)
    print(f"Jobs after filtering job_description_hash in cache: {jobs_after_hash_filter} (Filtered out: {jobs_after_id_filter - jobs_after_hash_filter})")

    # 4. Split into batches
    total_batches = (jobs_after_hash_filter + batch_size - 1) // batch_size
    print(f"Total batches created: {total_batches} with batch size {batch_size}")

    batches = []
    for i in range(0, jobs_after_hash_filter, batch_size):
        batch_df = df.iloc[i:i+batch_size].copy()
        batches.append(batch_df)
        print(f"Batch {len(batches)}: {len(batch_df)} jobs")

    return batches



def batch_callback(batch_df, results, cache):
    # Add processed records to cache
    # Key: job_id, Value: job_description_hash
    # We must have these in batch_df and in results. Here we assume 'recordId' = 'new_id' matches a unique job_id in df.
    # Suppose the original job_id column is still in batch_df. We map new_id -> job_id from batch_df.
    df_map = batch_df.set_index('new_id')
    for record_id, result in results.items():
        if record_id in df_map.index:
            job_id = df_map.loc[record_id, 'job_id']
            job_hash = df_map.loc[record_id, 'job_description_hash']
            cache[job_id] = job_hash


def batch_dict(d, batch_size=1500):
    items = list(d.items())
    for i in range(0, len(items), batch_size):
        yield dict(items[i:i + batch_size])


if __name__ == "__main__":
    region_name = 'us-east-1'
    profile_name = 'pssl-bedrock'
    role_name = "BedrockPermissionsRole"
    bucket_name = "fkkarami-projects"
    input_parquet = "tests/jobstruct/job_descriptions_test_extraction_2024_09_30.parquet"
    local_folder = "tests/cost-estimation/data/"
    target_records = 3000
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    offline_extract_mode = False
    offline_skills_mode = False
    offline_occupation_mode = False

    # Generate a session ID based on current datetime
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    s3_client = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn = get_iam_role_arn(role_name)

    with resources.open_text("jobstruct.data", "prompt_configs.json") as f:
        prompt_configs = json.load(f)

    # Prepare logs
    failed_batches = []
    successful_batches = []
    failed_log_file = os.path.join(local_folder, f"failed_batches_{session_id}.log")
    successful_log_file = os.path.join(local_folder, f"successful_batches_{session_id}.log")

    ################################
    # Batch 1: Extraction
    ################################
    task_name = "extract"
    extract_input_file_offline = "tests/cost-estimation/data/extract_inputs.jsonl.out"
    extract_output_file_offline = f"{local_folder}{task_name}_results_offline_{session_id}.json"
    extract_output_file = f"{local_folder}{task_name}_results_{session_id}.json"

    extraction_failed = False  # Track if any extraction batch failed
    extract_results = {}

    if offline_extract_mode:
        print("Offline mode enabled for extraction. Reading local results.")
        extract_records = wrap_json_file_with_array(extract_input_file_offline)
        extract_results = load_batch_output(extract_records, "extract")
        save_results_to_json(extract_results, extract_output_file_offline)
    else:
        expanded_df = expand_parquet_file_with_id(input_parquet, target_records)
        # Filter and batch for extraction
        batches = filter_and_prepare_extract_batch(expanded_df, cache, batch_size=1500)

        prompt_config = prompt_configs[task_name].copy()
        prompt_config.pop('modelId', None)

        output_s3_url = f"s3://{bucket_name}/bedrock-batch-inference/output-test/"

        for i, batch_df in enumerate(batches):
            if extraction_failed:
                # If a previous batch failed, we may decide to stop processing further extraction batches.
                break

            jsonl_file = f"{task_name}_inputs_batch_{i}_{session_id}.jsonl"
            create_jsonl_for_extract(batch_df, jsonl_file, task_name, prompt_config)
            input_s3_key = f"bedrock-batch-inference/input-test/{jsonl_file}"
            upload_file_to_s3(jsonl_file, bucket_name, input_s3_key, s3_client)
            input_s3_url = f"s3://{bucket_name}/{input_s3_key}"

            response = create_bedrock_job(bedrock_client, role_arn, model_id, input_s3_url, output_s3_url)
            job_arn = response.get('jobArn')
            final_status = check_job_status(job_arn, bedrock_client, f"Extraction Batch {i}")

            if final_status['status'] == "Completed":
                extract_s3_url = f"{output_s3_url}{job_arn.split('/')[-1]}/{jsonl_file}.out"
                extract_local_file = download_file_from_s3(extract_s3_url, local_folder, s3_client)
                extract_records = wrap_json_file_with_array(extract_local_file)
                batch_results = load_batch_output(extract_records, "extract")

                # Update global extract_results
                extract_results.update(batch_results)

                # Callback to update cache
                batch_callback(batch_df, batch_results, cache)

                # Log success
                successful_batches.append(f"extract_batch_{i}_{session_id}")
            else:
                print(f"Extraction Batch {i} failed with status {final_status['status']}")
                extraction_failed = True
                failed_batches.append(f"extract_batch_{i}_{session_id}")

        save_results_to_json(extract_results, extract_output_file)

    # If extraction failed at any batch, do not proceed to skills or occupation
    if extraction_failed:
        print("Extraction failed. Skipping Skills and Occupation tasks.")
    else:
        ################################
        # Batch 2: Skills
        ################################
        task_name = "skills"
        skills_output_file = f"{local_folder}{task_name}_results_{session_id}.json"
        skills_input_file_offline = "tests/cost-estimation/data/skills_inputs.jsonl.out"
        skills_output_file_offline = f"{local_folder}{task_name}_results_offline_{session_id}.json"

        skills_failed = False

        if offline_skills_mode:
            print("Offline mode enabled for skills. Reading local results.")
            skills_records = wrap_json_file_with_array(skills_input_file_offline)
            skills_results = load_batch_output(skills_records, "skills")
            save_results_to_json(skills_results, skills_output_file_offline)
        else:
            prompt_config = prompt_configs[task_name].copy()
            prompt_config.pop('modelId', None)

            skills_results = {}
            skill_batches = list(batch_dict(extract_results, batch_size=1500))

            for i, batch_data in enumerate(skill_batches):
                if skills_failed:
                    # If a previous skills batch failed, we may decide to stop further batches
                    break

                skills_jsonl_file = f"{task_name}_inputs_batch_{i}_{session_id}.jsonl"
                create_jsonl_for_skills_or_occupation(
                    batch_data, skills_jsonl_file, task_name, prompt_config
                )

                upload_file_to_s3(
                    skills_jsonl_file,
                    bucket_name,
                    f"bedrock-batch-inference/input-test/{skills_jsonl_file}",
                    s3_client
                )
                skills_response = create_bedrock_job(
                    bedrock_client,
                    role_arn,
                    model_id,
                    f"s3://{bucket_name}/bedrock-batch-inference/input-test/{skills_jsonl_file}",
                    output_s3_url
                )atg
                skills_job_arn = skills_response.get('jobArn')
                skills_status = check_job_status(skills_job_arn, bedrock_client, f"Skills Batch {i}")

                if skills_status['status'] == "Completed":
                    skills_s3_url = f"{output_s3_url}{skills_job_arn.split('/')[-1]}/{skills_jsonl_file}.out"
                    skills_local_file = download_file_from_s3(skills_s3_url, local_folder, s3_client)
                    skills_records = wrap_json_file_with_array(skills_local_file)
                    batch_skills_results = load_batch_output(skills_records, "skills")
                    # Update global skills results
                    skills_results.update(batch_skills_results)

                    successful_batches.append(f"skills_batch_{i}_{session_id}")
                else:
                    print(f"Skills Batch {i} failed with status {skills_status['status']}")
                    skills_failed = True
                    failed_batches.append(f"skills_batch_{i}_{session_id}")

                # Save partial results incrementally
                save_results_to_json(skills_results, skills_output_file)

        # If skills failed at any batch, do not proceed to occupation
        if skills_failed:
            print("Skills processing failed. Skipping Occupation tasks.")
        else:
            ################################
            # Batch 3: Occupation
            ################################
            task_name = "occupation"
            occupation_output_file = f"{local_folder}{task_name}_results_{session_id}.json"
            occupation_input_file_offline = "tests/cost-estimation/data/occupation_inputs.jsonl.out"
            occupation_output_file_offline = f"{local_folder}{task_name}_results_offline_{session_id}.json"

            occupation_failed = False

            if offline_occupation_mode:
                print("Offline mode enabled for occupation. Reading local results.")
                occupation_records = wrap_json_file_with_array(occupation_input_file_offline)
                occupation_results = load_batch_output(occupation_records, "occupation")
                save_results_to_json(occupation_results, occupation_output_file_offline)
            else:
                prompt_config = prompt_configs[task_name].copy()
                prompt_config.pop('modelId', None)

                occupation_results = {}
                occupation_batches = list(batch_dict(extract_results, batch_size=1500))

                for i, batch_data in enumerate(occupation_batches):
                    if occupation_failed:
                        break

                    occupation_jsonl_file = f"{task_name}_inputs_batch_{i}_{session_id}.jsonl"
                    create_jsonl_for_skills_or_occupation(
                        batch_data, occupation_jsonl_file, task_name, prompt_config
                    )

                    upload_file_to_s3(
                        occupation_jsonl_file,
                        bucket_name,
                        f"bedrock-batch-inference/input-test/{occupation_jsonl_file}",
                        s3_client
                    )
                    occupation_response = create_bedrock_job(
                        bedrock_client,
                        role_arn,
                        model_id,
                        f"s3://{bucket_name}/bedrock-batch-inference/input-test/{occupation_jsonl_file}",
                        output_s3_url
                    )
                    occupation_job_arn = occupation_response.get('jobArn')
                    occupation_status = check_job_status(occupation_job_arn, bedrock_client, f"Occupation Batch {i}")

                    if occupation_status['status'] == "Completed":
                        occupation_s3_url = f"{output_s3_url}{occupation_job_arn.split('/')[-1]}/{occupation_jsonl_file}.out"
                        occupation_local_file = download_file_from_s3(occupation_s3_url, local_folder, s3_client)
                        occupation_records = wrap_json_file_with_array(occupation_local_file)
                        batch_occupation_results = load_batch_output(occupation_records, "occupation")
                        occupation_results.update(batch_occupation_results)

                        successful_batches.append(f"occupation_batch_{i}_{session_id}")
                    else:
                        print(f"Occupation Batch {i} failed with status {occupation_status['status']}")
                        occupation_failed = True
                        failed_batches.append(f"occupation_batch_{i}_{session_id}")

                    save_results_to_json(occupation_results, occupation_output_file)

    # Save cache
    with open("cache.json", 'w') as file:
        json.dump(cache, file, indent=4)

    # Write out the logs for failed and successful batches
    if failed_batches:
        with open(failed_log_file, 'w') as f:
            for fb in failed_batches:
                f.write(f"{fb}\n")

    if successful_batches:
        with open(successful_log_file, 'w') as f:
            for sb in successful_batches:
                f.write(f"{sb}\n")

    print(f"Run Session ID: {session_id}")
    print(f"Failed batches logged to: {failed_log_file}")
    print(f"Successful batches logged to: {successful_log_file}")
