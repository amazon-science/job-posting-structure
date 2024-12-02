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

    skills = dedent("""
        You are a helpful assistant.
        Your task is to read the job requirements in the <text></text> tags and map each given qualification to relevant skills
        given within the <skills></skills> tags. Make sure to map each qualification. 
        Read the qualification carefully and make the correct mapping to the given set of skills.
        <text>
        {text}
        </text>
        Understand the qualifications above and map them to the skills:
        <skills>
        {skills}
        </skills>
        Return the mapped skills as a JSON list.
        Skip the preamble and the explanation.
        Be careful, think, check your answers and only then return your response. 
        You must not select skills at random, it must be through careful examination.""")

    occupation = dedent("""
        You are a helpful assistant.
        <task>
        You must select the two most relevant Standard Occupational Classification (SOC) codes for the job description
         provided within the <text></text> tags.
        </task>
        <instructions>
        Here are some important rules for the task:
        - Read the entire job description within <text></text> carefully.
        <text>
        {text}
        </text>
        - Based on your complete understanding of the job description, identify the two most relevant Standard Occupational
          Classification (SOC) major occupation code that best corresponds to the job description.
           Only and only if you are ambiguous about categorizing the job description into one single code, then return two codes.
           Otherwise you must return one code.
        </instructions>
        Return one or two codes using the schema below.
        <schema>
            ```json {{
                'occupation': <>Return a list of codes.</>
            }}```
        </schema>
        Skip the preamble and the explanation.
        Be careful, think, check your answers and only then return your response. 
        You must not select skills at random, it must be through careful examination.""")

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
