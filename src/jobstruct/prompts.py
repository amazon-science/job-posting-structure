
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0



import ast
import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime
import boto3
import yaml
from importlib import resources
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from textwrap import dedent
from typing import Any, Dict, List, Union, Tuple
from botocore.exceptions import ClientError
import os

os.environ['AWS_MAX_ATTEMPTS'] = '10'

class Prompts:
    """
    Preconstructed prompts for generative AI operations.
    """

    extract = dedent("""
        Your task is to read the job posting inside the <text></text> tags and accurately extract relevant information in the JSON format shown in <schema></schema>. Be very careful. Follow the instructions to perform the task.
        <instructions>
        Think step-by-step.
         1. Read the job posting carefully and thoroughly, line by line.
         2. Identify all sentences that describe the job duties, responsibilities, or requirements, including any details that are not explicitly labeled as such.
         3. **Important Step** : Ensure you capture all details related to the job duties, job requirements even if they are not clearly labeled. Do not miss out on any information indirectly present in the job posting. Some general job related sentences can qualify as job duties if you pay close attention.
         4. Do not make any assumptions or leave out any details.
         5. Present the final JSON output exactly as specified in the schema, without any truncation, summarization, or modification of the original text.
        </instructions>
        <text>
        {text}
        </text>
        Return the information in JSON format using the schema below.
        <schema>
            ```json {{
                'job_title': <>Return the job title.</>,
                'details': <>Return a list of all duties and responsibilities associated with the job. Include all information.</>,
                'required': {{
                    'education': <>Return the lowest educational level required</>,
                    'major': <>Return a list of required majors or areas of study</>,
                    'experience': <>Return the required years of experience as an integer</>,
                    'qualifications': <>Return a list of all required qualifications, abilities, knowledge, skills, certifications, training, and licenses</>,
                }},
                'preferred': {{
                    'education': <>Return the lowest educational level preferred</>,
                    'major': <>Return a list of preferred majors or areas of study</>,
                    'experience': <>Return the preferred years of experience as an integer</>,
                    'qualifications': <>Return a list of all preferred qualifications, abilities, knowledge, skills, certifications, training, and licenses</>,
                }},
                'benefits': <>Return a list of the benefits offered</>,
                'salary': <>Return a list with the salary or the salary range of the position as numbers</>,
                'wage': <>Return a list with the wage or the wage range of the position as numbers</>,
                'entry_level': <>Return true if the position is an entry-level job, otherwise return false</>,
                'college_degree': <>Return true if the position requires a college degree or equivalent, otherwise return false</>,
                'full_time': <>Return true if the position is full-time, otherwise return false</>,
                'remote': <>Return true if the position offers remote work, otherwise return false</>
            }}```
         </schema>""")

    skills = dedent("""\
        You are a helpful assistant.

        The skill tree provided within <skills></skills> is a *hierarchical dictionary* where:
        - Each *key* is a skill name.
        - Its *value* is another dictionary representing child skills.
        - Leaf nodes are empty dictionaries.

        Your task is to:
        1. Read the job requirements in the <text></text> tags.
        2. Map each qualification to relevant skills **only** if they appear in the provided hierarchical skill dictionary.
        3. **If more than 10 relevant skills exist, return exactly the 10 most relevant** and omit the rest.
        4. Return a valid JSON array of **double-quoted** strings (e.g., ["SkillA", "SkillB"]) **without** additional commentary.
        5. Return an empty array ([]) if no skills match.

        **Additional strict instructions**:
        • **Select only skill names that appear exactly (verbatim) in the provided taxonomy.**  
            - If a skill is spelled differently or partially matches, skip it.
            - If any qualification does not match a skill’s exact name in the taxonomy, do not include it.
        • **Never introduce synonyms, expansions, or new skill labels**. No rephrasing or approximate matches.
        • **Never exceed 10 skills total in your final array.** If more than 10 are valid, select only the top 10 you judge as “most relevant.”
        • **MAKE SURE YOU ARE NOT REAPEATING OR BRINGING QUALIFICATIONS INTO THE OUTPUT. ONLY DEDUCTED SKILLS**
        • If you are unsure or cannot find an exact skill in the taxonomy, **omit** it.
        • The final output must be a valid JSON array with **double-quoted** strings, and **nothing else**.

        <text>
        {text}
        </text>

        <skills>
        {skills}
        </skills>

        Please output **only** the skills array in valid JSON. **No** additional reasoning or explanation.
        """)

    occupation = dedent("""
        You are a helpful assistant.
        <task>
        You must select the one or two most relevant Standard Occupational Classification (SOC) codes for the job description
        provided within the <text></text> tags.
        </task>
        <instructions>
        Here are some important rules for the task:
        - Read the entire job description within <text></text> carefully.
        <text>
        {text}
        </text>
        - Identify one or two SOC codes that best match the job description.
        - DO NOT include descriptive text or occupation titles in your final answer, only the numeric code string.
        - Surround each code with double quotes.

        Return your result exactly in JSON format using the schema below:
        <schema>
        ```json {{
            "occupation": ["<string code>", "<string code>"]
        }}```
        </schema>
        
        Skip the preamble and the explanation.
        Be careful, think, check your answers and only then return your response. 
        """)

    embedding = ""

    taxonomy_enrich = dedent("""
        You are a helpful assistant.
        Your task is to read the skills taxonomy containing the parent node - leaf node combination within <tree></tree> 
        tags and expand only the leaf node. You must make use of all your knowledge on job postings and expand the leaf 
        nodes using the related skills only. You must be very careful.
        <tree>
        {text}
        </tree>
        <note>
        - You must return your response in the same format of the tree.
        - You are free to expand the leaf nodes up to whatever depth you feel necessary, however make sure to add only 
        relevant skills as nodes. Use all your knowledge to create an expanded tree and make it comprehensive.
        </note>
        Review your output for correctness and check if all instructions have been followed. Skip the explanation and 
        the preamble and return your verified response only.""")

    taxonomy_refine = dedent("""
        You are a helpful assistant.
        Your task is to review the skills taxonomy contained in the <tree></tree> tags, remove skills that are duplicates or too specific, and add any important skills that are missing.
        <tree>
        {text}
        </tree>
        <note>
        - You must return your response in the same format of the tree.
        - Make sure the tree contains new and emerging skills that are important in the labor market.
        </note>
        Review your output for correctness and check if all instructions have been followed. Skip the explanation and the preamble and return your verified response only.""")

    taxonomy_prune = dedent("""
        You are a helpful assistant.
        Your task is to review the skills contained in the <tree></tree> tags.
        Some skills are duplicates with different parents. For each duplicate, you need to select one to keep based on the context (parent skills) and suggest removal of the rest.
        <tree>
        {text}
        </tree>
        <note>
        - Return your response in the same tree format, with duplicate skills removed.
        - For each removed skill, list it under a "removed" key in the output.
        - Ensure the remaining skill fits well with the parent nodes and the overall tree.
        - Do not change any skills unless they are duplicates.
        </note>
        Review your output for correctness and check if all instructions have been followed. Skip the explanation and the preamble and return your verified response only.
    """)

    def __init__(
        self,
        client: BedrockRuntimeClient,
        config_file: str = "",
    ):
        self.client = client
        if config_file:
            with open(config_file) as f:
                self.prompt_configs = json.load(f)
        else:
            with resources.open_text("jobstruct.data", "prompt_configs.json") as f:
                self.prompt_configs = json.load(f)

    @staticmethod
    def safe_literal(text: str, default: Any) -> Union[Dict, List]:
        """
        Safely parse python output from `text` after stripping extraneous text.
        Return the `default` value if parsing fails.
        """
        log = logging.getLogger("jobstruct.Prompts.safe_literal")

        text = re.sub(r"(^[^\{\[]*)|([^\]\}]*$)", "", text)
        log.debug("stripped text: {}".format(text))

        try:
            return ast.literal_eval(text)
        except SyntaxError:
            log.debug("ast.literal_eval failed")
            return default

    @staticmethod
    def safe_json(text: str, default: Any) -> Union[Dict, List]:
        """
        Safely parse JSON output from `text` after stripping extraneous text.
        Return the `default` value if parsing fails.
        """
        log = logging.getLogger("jobstruct.Prompts.safe_json")

        text = re.sub(r"(^[^\{\[]*)|([^\]\}]*$)", "", text)
        log.debug(f"stripped text: {text}")

        try:
            return json.loads(text)
        except json.decoder.JSONDecodeError:
            log.debug("json.loads failed")
            return default

    def invoke(
        self,
        name: str,
        text: str,
        skills: str = "",
    ) -> Union[Dict, List, Tuple]:
        """
        Synchronous method to invoke the LLM, with exponential backoff retry strategy.
        """
        if not hasattr(Prompts, name):
            raise ValueError(f"{name} is an unrecognized prompt")
        if name not in self.prompt_configs:
            raise ValueError(f"{name} is missing from prompt_configs")
        log = logging.getLogger("jobstruct.Prompts.invoke")

        if not hasattr(Prompts, name):
            raise ValueError(f"{name} is an unrecognized prompt")
        if name not in self.prompt_configs:
            raise ValueError(f"{name} is missing from prompt_configs")

        prompt_config = self.prompt_configs[name].copy()
        modelId = prompt_config.pop("modelId")
        if name == "embedding":
            prompt_config["inputText"] = text
        else:
            prompt_config["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": getattr(Prompts, name).format(
                                text=text,
                                skills=skills,
                            )
                        }
                    ]
                }
            ]
        # print(prompt_config["messages"][0]["content"])
        body = json.dumps(prompt_config)
        log.debug(f"'{name}' body: {body}")

        MAX_RETRIES = 8
        INITIAL_DELAY = 3  # seconds
        RETRYABLE_EXCEPTIONS = ('ThrottlingException', 'TooManyRequestsException')

        delay = INITIAL_DELAY
        llm_call_metadata = {}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.invoke_model(
                    body=body,
                    modelId=modelId,
                    accept="application/json",
                    contentType="application/json"
                )
                llm_call_metadata = response['ResponseMetadata']
                log.debug(f"response: {response}")

                if name == "embedding":
                    result = (
                        json
                        .loads(response.get("body").read())
                        .get("embedding")
                    )
                else:
                    result = (
                        json
                        .loads(response.get("body").read())
                        .get("content")[0]
                        .get("text")
                    )
                log.debug(f"result: {result}")

                return result, llm_call_metadata

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in RETRYABLE_EXCEPTIONS:
                    log.warning(f"Attempt {attempt} failed due to rate limit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    # Exponential backoff
                    delay = min(delay * 2, 60) + random.uniform(0, 1)
                else:
                    log.error(f"Non-retryable ClientError occurred: {e}")
                    raise e
            except Exception as e:
                log.error(f"An unexpected error occurred: {e}")
                raise e

        # If all retries are exhausted, raise an exception
        raise Exception("Maximum retry attempts exceeded for invoke method.")

    async def invoke_async(
        self,
        name: str,
        text: str,
        skills: str = "",
    ) -> Union[Dict, List]:
        """
        Asynchronous wrapper for the invoke method using asyncio.
        """
        loop = asyncio.get_running_loop()
        result, llm_call_metadata = await loop.run_in_executor(
            None,
            self.invoke,
            name,
            text,
            skills
        )
        # print(result)
        return result, llm_call_metadata

    def batch_invoke(self,
                     name,
                     bedrock_client,
                     s3_client,
                     input_file_s3_url,
                     output_file_s3_url,
                     input_descriptions, ):

        session = boto3.Session(profile_name='pssl-bedrock', region_name='us-east-1')
        s3_client = session.client("s3")
        bedrock_client = session.client(service_name="bedrock")

        def create_jsonl_from_dataframe_with_template(df, output_file, name, skills=""):
            prompt_config = self.prompt_configs[name].copy()
            prompt_config.pop('ModelId')

            with (open(output_file, 'w') as file):
                for idx, row in df.iterrows():
                    modelInput = prompt_config
                    text = row['description']
                    id = row['job_id']

                    modelInput['messages'] = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": getattr(Prompts, name).format(text=text, skills=skills)
                                }
                            ]
                        }
                    ]
                    record = {
                        "": id,
                        "modelInput": modelInput
                    }
                    file.write(json.dumps(record) + '\n')

        def upload_file_to_s3(file_path, bucket_name, s3_key):
            s3_client = boto3.client('s3', region_name='us-east-1')
            try:
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f"File uploaded successfully to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"Error uploading file: {e}")

        def get_role_arn(role_name):
            iam_client = boto3.client('iam')
            response = iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]

        role_arn = get_role_arn("BedrockPermissionsRole")

        input_data_config = {
            "s3InputDataConfig": {
                "s3Uri": "s3://fkkarami-projects/bedrock-batch-inference/input/input.jsonl"
            }
        }

        output_data_config = {
            "s3OutputDataConfig": {
                "s3Uri": "s3://fkkarami-projects/bedrock-batch-inference/output/"
            }
        }

        def generate_bedrock_job_name(prefix="bedrock-job"):
            current_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            job_name = f"{prefix}-{current_time}"
            return job_name

        response = bedrock_client.create_model_invocation_job(
            roleArn=role_arn,
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            jobName=generate_bedrock_job_name(),
            inputDataConfig=input_data_config,
            outputDataConfig=output_data_config,
        )

        job_arn = response.get('jobArn')
        print(f"Batch Inference Job ARN: {job_arn}")

        def check_job_status(job_arn):
            bedrock = boto3.client('bedrock', region_name='us-east-1')
            while True:
                status_response = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
                print(f"Job Status: {status_response['status']}")

                if status_response['status'] in ['Completed', 'Failed', 'Stopped']:
                    print("Job finished with status:", status_response['status'])
                    break

                time.sleep(5)

        check_job_status(job_arn)






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

CONFIG_FILE_PATH = "config.json"
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

# Add jobstruct_path to sys.path if needed
jobstruct_path = config["jobstruct_path"]
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

from jobstruct import Prompts

# Extract values from config
CREATE_AND_UPLOAD_INPUT_FILES = config["execution"]["create_and_upload_input_files"]
RUN_BEDROCK_JOBS_END_TO_END   = config["execution"]["run_bedrock_jobs_end_to_end"]
RECORD_INDEX                  = tuple(config["execution"]["record_index"])

TASK_LIST      = config["task"]["task_list"]
TAXONOMY_FILES = config["task"]["taxonomy_files"]
MAX_TOKENS     = config["task"]["max_tokens"]
batch_sizes    = config["task"]["batch_sizes"]

region_name  = config["aws"]["region_name"]
profile_name = config["aws"]["profile_name"]
role_name    = config["aws"]["role_name"]
bucket_name  = config["aws"]["bucket_name"]

model_id = config["model_id"]

extracted_data_path = config["paths"]["extracted_data_path"]
# local_folder, s3_folder_input_files, and s3_folder_output_files will be formatted with session_number at runtime


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

    # Format paths with the current session_number
    local_folder             = config["paths"]["local_folder"].format(session_number=session_number)
    s3_folder_input_files    = config["paths"]["s3_folder_input_files"].format(session_number=session_number)
    s3_folder_output_files   = config["paths"]["s3_folder_output_files"].format(session_number=session_number)

    print("[INFO] Setting up AWS clients...")
    s3_client      = get_boto3_client("s3", profile_name, region_name)
    bedrock_client = get_boto3_client("bedrock", profile_name, region_name)
    role_arn       = get_iam_role_arn(role_name)
    print("[INFO] AWS clients initialized.")


    if CREATE_AND_UPLOAD_INPUT_FILES:

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
                details   = clean_text("\n".join(record['external_qualifications']))
                required  = clean_text("\n".join(record['basic_qualifications']))
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
                    prefix=f"{task_name}-job",
                    session_number=session_number
                )
                job_arn = bedrock_response.get('jobArn')
                print(f"[INFO] Job ARN: {job_arn}")

            print(f"[INFO] All jobs for {task_name.capitalize()} Task have been submitted.")

        print("[INFO] All Bedrock jobs have been submitted.")






import sys
jobstruct_path = "/home/fkkarami/workspace/amazon/old_ebsvolume/home/fkkarami/workspace/NASWA/job-posting-structure/src"
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

import os
import json
import re
import boto3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from jobstruct import Prompts

def post_process_skills_results(input_data):
    output_dict = {}
    for key, value in input_data.items():
        try:
            output_dict[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Invalid JSON format for key: {key}")
    return output_dict

def post_process_occupation_results(raw_results):
    processed_results = {}
    for key, value in raw_results.items():
        try:
            parsed_value = json.loads(value)
            if isinstance(parsed_value, dict) and "occupation" in parsed_value:
                processed_results[key] = parsed_value["occupation"]
        except json.JSONDecodeError:
            print(f"Failed to parse value for key: {key}")
            continue
    return processed_results

def clean_text(text: str) -> str:
    if not text:
        return text
    
    text = re.sub(r'\n+', ' ', text)          # Replace multiple \n with single space
    text = re.sub(r'(?<!\s)\s(?!\s)', '', text)  # Remove spaces between letters within words
    text = re.sub(r'\s+', ' ', text)          # Collapse multiple spaces into one
    return text.strip()

def get_bedrock_runtime_client(profile_name="pssl-bedrock", region_name="us-east-1"):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client("bedrock-runtime")

async def run_skills_inference_async(
    prompts: Prompts,
    job_guid: str,
    text: str,
    skills_taxonomy_str: str
) -> str:
    """
    Runs the 'skills' prompt, printing the approximate prompt size for debugging.
    """
    prompt_size = len(text) + len(skills_taxonomy_str)
    print(f"[DEBUG] (Job {job_guid}) Skills prompt size ~ {prompt_size} characters.")

    # Actually call the LLM
    result, _metadata = await prompts.invoke_async(
        name="skills",
        text=text,
        skills=skills_taxonomy_str
    )
    return result

async def run_occupation_inference_async(prompts: Prompts, job_guid: str, text: str) -> str:
    """
    Runs the 'occupation' prompt, printing the approximate prompt size for debugging.
    """
    prompt_size = len(text)
    print(f"[DEBUG] (Job {job_guid}) Occupation prompt size ~ {prompt_size} characters.")

    result, _metadata = await prompts.invoke_async(
        name="occupation",
        text=text,
        skills=""
    )
    return result

def preprocess_records(df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        job_guid = row.get("job_guid", "")
        job_title = clean_text(row.get("external_title", ""))
        details = clean_text("\n".join(row.get("external_qualifications", [])))
        required = clean_text("\n".join(row.get("basic_qualifications", [])))
        preferred = clean_text("\n".join(row.get("preferred_qualifications", [])))

        input_text = "\n\n".join([
            job_title,
            details,
            required,
            preferred
        ])
        records.append((job_guid, input_text))
    return records

def build_compact_hierarchy(node: dict) -> dict:
    children = node.get("children", [])
    if not children:
        return {}
    result = {}
    for child in children:
        child_name = child["name"]
        result[child_name] = build_compact_hierarchy(child)
    return result

async def process_record(
    prompts: Prompts,
    job_guid: str,
    input_text: str,
    skills_taxonomy_str: str,
    skills_output_dict: dict,
    occupation_output_dict: dict,
    semaphore: asyncio.Semaphore,
    counters: dict
):
    async with semaphore:
        # Move from "pending" to "running"
        counters["running"] += 1
        counters["pending"] -= 1

        print(f"[INFO] Starting inference for job_guid={job_guid}. "
              f"(Pending={counters['pending']}, Running={counters['running']}, Completed={counters['completed']})")

        # Launch both tasks
        skills_task = run_skills_inference_async(prompts, job_guid, input_text, skills_taxonomy_str)
        occupation_task = run_occupation_inference_async(prompts, job_guid, input_text)

        skills_result, occupation_result = await asyncio.gather(skills_task, occupation_task)
        skills_output_dict[job_guid] = skills_result
        occupation_output_dict[job_guid] = occupation_result

        # Move from "running" to "completed"
        counters["running"] -= 1
        counters["completed"] += 1

        print(f"[INFO] Finished inference for job_guid={job_guid}. "
              f"(Pending={counters['pending']}, Running={counters['running']}, Completed={counters['completed']})")

async def main_async():
    extracted_data_path = "auxiliary_files/data_deduplicated.parquet"
    print(f"[INFO] Loading extraction data from {extracted_data_path}...")
    df = pd.read_parquet(extracted_data_path).iloc[:10000]
    print(f"[INFO] Extraction data loaded. Total records: {len(df)}")
    print(f"[INFO] Number of unique job IDs in the data: {df['job_guid'].nunique()}")

    taxonomy_file = "auxiliary_files/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json"
    print(f"[INFO] Loading taxonomy from {taxonomy_file}...")
    with open(taxonomy_file) as f:
        skills_taxonomy = json.load(f)
    # Build a compact hierarchy, then convert it to a JSON string
    skills_taxonomy_compact = build_compact_hierarchy(skills_taxonomy)
    skills_taxonomy_str = json.dumps(skills_taxonomy_compact)

    print("[INFO] Taxonomy loaded and converted to compact hierarchy (and then to JSON string).")

    records = preprocess_records(df)
    print(f"[INFO] Preprocessed {len(records)} records.\n")

    # Prepare results dictionaries
    skills_output_dict = {}
    occupation_output_dict = {}

    print("[INFO] Initializing Bedrock runtime client and Prompts object...")
    bedrock_client = get_bedrock_runtime_client()
    prompts = Prompts(client=bedrock_client)
    print("[INFO] Initialization complete.\n")

    SEMAPHORE_LIMIT = 100
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    os.makedirs("on_demand_results_async", exist_ok=True)

    BATCH_SIZE = 100
    total_records = len(records)
    total_batches = (total_records // BATCH_SIZE) + (1 if total_records % BATCH_SIZE else 0)

    print(f"[INFO] Starting async inference in {total_batches} batches (batch size = {BATCH_SIZE}).\n")

    # Track how many total queries are pending, running, completed
    counters = {
        "pending": len(records),  # All tasks start as "pending"
        "running": 0,
        "completed": 0
    }

    for batch_index in range(total_batches):
        start_i = batch_index * BATCH_SIZE
        end_i = start_i + BATCH_SIZE
        batch = records[start_i:end_i]

        print(f"[INFO] Processing batch {batch_index+1}/{total_batches} with {len(batch)} records...")

        tasks = []
        for (job_guid, input_text) in batch:
            task = process_record(
                prompts,
                job_guid,
                input_text,
                skills_taxonomy_str,
                skills_output_dict,
                occupation_output_dict,
                semaphore,
                counters
            )
            tasks.append(task)

        print(f"[INFO] Submitting {len(tasks)} tasks for batch {batch_index+1}. "
              f"(Pending={counters['pending']}, Running={counters['running']}, Completed={counters['completed']})")

        await asyncio.gather(*tasks)
        print(f"[INFO] [Batch {batch_index+1}] completed processing {len(tasks)} tasks.\n")

        # PARTIAL SAVE
        print(f"[INFO] [Batch {batch_index+1}/{total_batches}] Saving partial results...")

        partial_skills_processed = post_process_skills_results(skills_output_dict)
        partial_occupations_processed = post_process_occupation_results(occupation_output_dict)

        partial_skills_path = os.path.join(
            "on_demand_results_async",
            f"skills_output_part_{batch_index+1}.json"
        )
        partial_occupation_path = os.path.join(
            "on_demand_results_async",
            f"occupation_output_part_{batch_index+1}.json"
        )

        with open(partial_skills_path, "w") as f:
            json.dump(partial_skills_processed, f, indent=2)
        with open(partial_occupation_path, "w") as f:
            json.dump(partial_occupations_processed, f, indent=2)

        print(f"    [Partial Save] Skills -> {partial_skills_path}")
        print(f"    [Partial Save] Occupations -> {partial_occupation_path}\n")

    # Final save for entire dataset
    skills_output_path = os.path.join("on_demand_results_async", "skills_output.json")
    occupation_output_path = os.path.join("on_demand_results_async", "occupation_output.json")

    print("[INFO] Performing final post-processing and saving entire dataset...")
    final_skills_processed = post_process_skills_results(skills_output_dict)
    final_occupations_processed = post_process_occupation_results(occupation_output_dict)

    with open(skills_output_path, "w") as f:
        json.dump(final_skills_processed, f, indent=2)
    print(f"FINAL - Skills inference results saved to {skills_output_path}")

    with open(occupation_output_path, "w") as f:
        json.dump(final_occupations_processed, f, indent=2)
    print(f"FINAL - Occupation inference results saved to {occupation_output_path}\n")

    print("[INFO] All batches processed successfully. Exiting now.\n")

if __name__ == "__main__":
    asyncio.run(main_async())

