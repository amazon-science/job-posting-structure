
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

        The skill list provided within <skills></skills> is a list of candidate skills.

        Your task is to:
        1. Read the job requirements in the <text></text> tags.
        2. Map each qualification to relevant skills in the provided skills list.
        3. **If any skill is identified in the job posting that is not part of the skills list or has a closely related skill missing, append the 
                    marker [NEW_SKILL] directly to that skill (i.e., include it in the string) rather than listing it as a separate item.**
        4. Internally reason about your selections by assigning a confidence score to each selected skill based on their relevancy to the job requirements.
        5. Rank the deduced skills based on these confidence scores and output no more than 8 skills in total. If more than 8 relevant skills exist, 
                    select only the 8 most relevant.
        6. **Do not include any of your internal reasoning or the confidence scores in the final output.** The final output must be a valid JSON array 
                    of **double-quoted** strings (e.g., ["SkillA", "SkillB"]) without any additional commentary.
        7. If no skills match, return an empty array ([]) without additional commentary.

        **Additional strict instructions**:
        • **Select only skill names that appear exactly (verbatim) in the provided skills list.**  
            - Do not include skills that are spelled differently or only partially match.
            - Do not introduce synonyms, expansions, or new skill labels.
        • **Never exceed 8 skills total in your final output.**
        • **Do not repeat qualifications or include any content other than the deduced skills.**
        • If you are unsure or cannot find an exact match, **omit** the qualification.
        • **If a new skill is introduced that does not match an existing skills list skill, append the marker 
                    [NEW_SKILL] to that skill string instead of adding it as a separate element.**
        • The final output must be a valid JSON array with **double-quoted** strings, and **nothing else**.

        <text>
        {text}
        </text>

        <skills>
        {skills}
        </skills>

        Please output **only** the skills array in valid JSON. **No** additional reasoning or explanation.
        """)

    skills_resume = dedent("""\
        You are a helpful assistant.

        The skill list provided within <skills></skills> is a list of candidate skills.

        Your task is to:
        1. Read the resume text in the <text></text> tags.
        2. Map each qualification to relevant skills in the provided skills list.
        3. **If any skill is identified in the resume that is not part of the skills list or has a closely related skill missing, append the marker [NEW_SKILL] directly to that skill (i.e., include it in the string) rather than listing it as a separate item.**
        4. Internally reason about your selections by assigning a confidence score to each selected skill based on their relevancy to the resume content.
        5. Rank the deduced skills based on these confidence scores and output no more than 8 skills in total. If more than 8 relevant skills exist, select only the 8 most relevant.
        6. **Do not include any of your internal reasoning or the confidence scores in the final output.** The final output must be a valid JSON array of **double-quoted** strings (e.g., ["SkillA", "SkillB"]) without any additional commentary.
        7. If no skills match, return an empty array ([]) without additional commentary.

        **Additional strict instructions**:
        • **Select only skill names that appear exactly (verbatim) in the provided skills list.**  
            - Do not include skills that are spelled differently or only partially match.
            - Do not introduce synonyms, expansions, or new skill labels.
        • **Never exceed 8 skills total in your final output.**
        • **Do not repeat qualifications or include any content other than the deduced skills.**
        • If you are unsure or cannot find an exact match, **omit** the qualification.
        • **If a new skill is introduced that does not match an existing skills list skill, append the marker [NEW_SKILL] to that skill string instead of adding it as a separate element.**
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

        the SOC codes are provided withing the <soc codes></soc codes> tags.
        </task>
        <instructions>
        Here are some important rules for the task:
        - Read the entire job description within <text></text> carefully.
        <text>
        {text}
        </text>
        - Identify one or two SOC codes that best match the job description.
         <soc codes>
        {soc_codes}
         </soc codes>
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
        **“Return only the JSON and do not wrap your response in code blocks or triple backticks.”**
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
        # soc_codes:str = ""
    ) -> Union[Dict, List, Tuple]:
        """
        Synchronous method to invoke the LLM, with exponential backoff retry strategy.
        """
        with open('/home/fkkarami/workspace/amazon/job-posting-structure/src/jobstruct/data/soc_codes.json', 'r') as file:
            soc_codes = json.load(file)
        if not hasattr(Prompts, name):
            raise ValueError(f"{name} is an unrecognized prompt")
        # if name not in self.prompt_configs:
        #     raise ValueError(f"{name} is missing from prompt_configs")
        log = logging.getLogger("jobstruct.Prompts.invoke")

        if not hasattr(Prompts, name):
            raise ValueError(f"{name} is an unrecognized prompt")

        if name in ['skills','skills_resume']:
            task_name = 'skills'
        else:
            task_name = name
        prompt_config = self.prompt_configs[task_name].copy()
        modelId = prompt_config.pop("modelId")
        if name == "embedding":
            prompt_config["inputText"] = text
        else:
            if name == 'extract':
                # print(Prompts.extract)
                prompt_body = Prompts.extract.format(text=text)
            elif name == 'skills':
                prompt_body = Prompts.skills.format(text=text, skills=skills)
            elif name == 'skills_resume':
                prompt_body = Prompts.skills_resume.format(text=text, skills=skills)
            elif name == 'occupation':
                prompt_body = Prompts.occupation.format(text=text, soc_codes=soc_codes)
            prompt_config["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_body
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
        skills: str,
        # soc_codes: str = ""
    ) -> Union[Dict, List]:
        """
        Asynchronous wrapper for the invoke method using asyncio.
        """
        print(skills)
        loop = asyncio.get_running_loop()
        result, llm_call_metadata = await loop.run_in_executor(
            None,
            self.invoke,
            name,
            text,
            skills,
            # soc_codes
        )
        # print(result)
        return result, llm_call_metadata



