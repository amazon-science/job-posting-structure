import os
import json
import random
import ast
import time
import boto3
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
from jobstruct import Prompts, JobStructAI
from importlib import resources


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

    # Expand the DataFrame to meet the target record count
    repeat_times = (target_records // len(df)) + 1
    expanded_df = pd.concat([df] * repeat_times, ignore_index=True)
    expanded_df = expanded_df.iloc[:target_records]

    # Add a new column with unique 9-digit IDs
    expanded_df['new_id'] = [random.randint(100000000, 999999999) for _ in range(target_records)]

    return expanded_df


def create_jsonl_for_extract(df, output_file, task_name, prompt_config, skills=""):
    """
    Create JSONL file for the 'extract' task.
    Processes the DataFrame directly.

    Args:
        df (pd.DataFrame): Input DataFrame.
        output_file (str): Output JSONL file path.
        task_name (str): Name of the task (e.g., 'extract').
        prompt_config (dict): Prompt configuration for the task.
        skills (str): Optional skills to include in the prompt.

    Returns:
        None
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
    """
    Create JSONL file for the 'skills' or 'occupation' tasks.
    Processes the extracted results.

    Args:
        records (dict): Extracted results from the 'extract' task.
        output_file (str): Output JSONL file path.
        task_name (str): Name of the task (e.g., 'skills' or 'occupation').
        prompt_config (dict): Prompt configuration for the task.
        skills (str): Optional skills to include in the prompt.

    Returns:
        None
    """
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


def load_batch_output(output_records, task_name):
    results = {}
    for record in output_records:
        try:
            body = record["modelOutput"]["content"][0]["text"]
            if isinstance(body, str):
                body = body.strip()
            if task_name == "extract":
                results[record["recordId"]] = Prompts.safe_json(body, {})
            elif task_name == "occupation":
                results[record["recordId"]] = extract_occupation(body)
            elif task_name == "skills":
                results[record["recordId"]] = extract_skills(body)
            else:
                return

        except Exception as e:
            print(body)
            print(f"Error processing record: {e}")
    return results


def extract_skills(data_str):
    # Parse the input string
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        data = ast.literal_eval(data_str)

    skills = []
    if isinstance(data, dict):
        # First format: dictionary with qualifications as keys and skills as values
        for v in data.values():
            skills.extend(v)
    elif isinstance(data, list):
        if all(isinstance(item, dict) and 'skills' in item for item in data):
            # Second format: list of dictionaries with 'skills' keys
            for item in data:
                skills.extend(item.get('skills', []))
        elif all(isinstance(item, str) for item in data):
            # Third format: list of skills (strings)
            skills.extend(data)
        else:
            # Unknown format within the list
            pass
    else:
        # Data is neither a dict nor a list
        pass
    # Remove duplicates
    return list(set(skills))



def save_results_to_json(results, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")



def extract_occupation(data):
    try:
        # Try parsing as JSON
        data_json = json.loads(data)
    except json.JSONDecodeError:
        # If JSON parsing fails, try parsing as a Python literal
        data_json = ast.literal_eval(data)
    return data_json['occupation']


# Updated main execution block
if __name__ == "__main__":
    # Parameters
    region_name = 'us-east-1'
    profile_name = 'pssl-bedrock'
    role_name = "BedrockPermissionsRole"
    bucket_name = "fkkarami-projects"
    input_parquet = "tests/cost-estimation/job_descriptions_test_extraction_2024_09_30.parquet"
    local_folder = "tests/cost-estimation/data/"
    target_records = 2000
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    offline_extract_mode = False
    offline_skills_mode = False
    offline_occupation_mode = False

    # Initialize Clients
    s3_client = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn = get_iam_role_arn(role_name)

    # Load Prompt Configurations
    with resources.open_text("jobstruct.data", "prompt_configs.json") as f:
        prompt_configs = json.load(f)

    # **Batch 1: Extraction**
    task_name = "extract"
    jsonl_file = f"{task_name}_inputs.jsonl"
    extract_output_file = f"{local_folder}{task_name}_results.json"

    extract_input_file_offline = "tests/cost-estimation/data/extract_inputs.jsonl.out"
    extract_output_file_offline = f"{local_folder}{task_name}_results_offline.json"

    if offline_extract_mode:
        # Offline processing for extract
        print("Offline mode enabled for extraction. Reading local results.")
        # with open(extract_output_file, 'r') as f:
        #     extract_results = json.load(f)
        extract_records = wrap_json_file_with_array(extract_input_file_offline)
        extract_results = load_batch_output(extract_records, "extract")
        save_results_to_json(extract_results, extract_output_file_offline)
    else:
        # Online processing for extract
        expanded_df = expand_parquet_file_with_id(input_parquet, target_records)
        prompt_config = prompt_configs[task_name].copy()
        prompt_config.pop('modelId', None)

        # Create JSONL for extract
        create_jsonl_for_extract(expanded_df, jsonl_file, task_name, prompt_config)

        # Upload JSONL and submit job
        input_s3_key = f"bedrock-batch-inference/input/{jsonl_file}"
        upload_file_to_s3(jsonl_file, bucket_name, input_s3_key, s3_client)
        input_s3_url = f"s3://{bucket_name}/{input_s3_key}"
        output_s3_url = f"s3://{bucket_name}/bedrock-batch-inference/output/"
        response = create_bedrock_job(bedrock_client, role_arn, model_id, input_s3_url, output_s3_url)
        job_arn = response.get('jobArn')
        final_status = check_job_status(job_arn, bedrock_client, "Extraction")

        if final_status['status'] == "Completed":
            extract_s3_url = f"{output_s3_url}{job_arn.split('/')[-1]}/{jsonl_file}.out"
            extract_local_file = download_file_from_s3(extract_s3_url, local_folder, s3_client)
            extract_records = wrap_json_file_with_array(extract_local_file)
            extract_results = load_batch_output(extract_records, "extract")
            save_results_to_json(extract_results, extract_output_file)

    # **Batch 2: Skills**
    task_name = "skills"
    skills_jsonl_file = f"{task_name}_inputs.jsonl"
    skills_output_file = f"{local_folder}{task_name}_results.json"

    skills_input_file_offline = "tests/cost-estimation/data/skills_inputs.jsonl.out"
    skills_output_file_offline = f"{local_folder}{task_name}_results_offline.json"

    if offline_skills_mode:
        # Offline processing for skills
        print("Offline mode enabled for skills. Reading local results.")
        # with open(skills_output_file, 'r') as f:
        #     skills_results = json.load(f)
        skills_records = wrap_json_file_with_array(skills_input_file_offline)
        skills_results = load_batch_output(skills_records, "skills")
        save_results_to_json(skills_results, skills_output_file_offline)
    else:
        # Online processing for skills
        prompt_config = prompt_configs[task_name].copy()
        prompt_config.pop('modelId', None)

        # Create JSONL for skills
        create_jsonl_for_skills_or_occupation(
            extract_results, skills_jsonl_file, task_name, prompt_config
        )

        # Upload JSONL and submit job
        upload_file_to_s3(skills_jsonl_file, bucket_name, f"bedrock-batch-inference/input/{skills_jsonl_file}",
                          s3_client)
        skills_response = create_bedrock_job(
            bedrock_client, role_arn, model_id, f"s3://{bucket_name}/bedrock-batch-inference/input/{skills_jsonl_file}",
            output_s3_url
        )
        skills_job_arn = skills_response.get('jobArn')
        skills_status = check_job_status(skills_job_arn, bedrock_client, "Skills")

        if skills_status['status'] == "Completed":
            skills_s3_url = f"{output_s3_url}{skills_job_arn.split('/')[-1]}/{skills_jsonl_file}.out"
            skills_local_file = download_file_from_s3(skills_s3_url, local_folder, s3_client)
            skills_records = wrap_json_file_with_array(skills_local_file)
            skills_results = load_batch_output(skills_records, "skills")
            save_results_to_json(skills_results, skills_output_file)

    # **Batch 3: Occupation**
    task_name = "occupation"
    occupation_jsonl_file = f"{task_name}_inputs.jsonl"
    occupation_output_file = f"{local_folder}{task_name}_results.json"

    occupation_input_file_offline = "tests/cost-estimation/data/occupation_inputs.jsonl.out"
    occupation_output_file_offline = f"{local_folder}{task_name}_results_offline.json"

    if offline_occupation_mode:
        # Offline processing for occupation
        print("Offline mode enabled for occupation. Reading local results.")
        # with open(occupation_output_file, 'r') as f:
        #     occupation_results = json.load(f)
        occupation_records = wrap_json_file_with_array(occupation_input_file_offline)
        occupation_results = load_batch_output(occupation_records, "occupation")
        save_results_to_json(occupation_results, occupation_output_file_offline)

    else:
        # Online processing for occupation
        prompt_config = prompt_configs[task_name].copy()
        prompt_config.pop('modelId', None)

        # Create JSONL for occupation
        create_jsonl_for_skills_or_occupation(
            extract_results, occupation_jsonl_file, task_name, prompt_config
        )

        # Upload JSONL and submit job
        upload_file_to_s3(
            occupation_jsonl_file, bucket_name, f"bedrock-batch-inference/input/{occupation_jsonl_file}", s3_client
        )
        occupation_response = create_bedrock_job(
            bedrock_client, role_arn, model_id,
            f"s3://{bucket_name}/bedrock-batch-inference/input/{occupation_jsonl_file}", output_s3_url
        )
        occupation_job_arn = occupation_response.get('jobArn')
        occupation_status = check_job_status(occupation_job_arn, bedrock_client, "Occupation")

        if occupation_status['status'] == "Completed":
            occupation_s3_url = f"{output_s3_url}{occupation_job_arn.split('/')[-1]}/{occupation_jsonl_file}.out"
            occupation_local_file = download_file_from_s3(occupation_s3_url, local_folder, s3_client)
            occupation_records = wrap_json_file_with_array(occupation_local_file)
            occupation_results = load_batch_output(occupation_records, "occupation")
            save_results_to_json(occupation_results, occupation_output_file)

