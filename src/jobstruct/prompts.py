# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import logging
import re
from importlib import resources
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from textwrap import dedent
from typing import Any, Dict, List, Union

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
        given within the <skills></skills> tags. Make sure to map each qualification. Read the qualification carefully and make the correct mapping to the given set of skills.
        <text>
        {text}
        </text>
        Understand the qualifications above and map them to the skills:
        <skills>
        {skills}
        </skills>
        Return the mapped skills as a JSON list.
        Skip the preamble and the explanation.
        Be careful, think, check your answers and only then return your response. You must not select skills at random, it must be through careful examination.""")

    occupation = dedent("""
        You are a helpful assistant.
        <task>
        You must select the two most relevant Standard Occupational Classification (SOC) codes for the job description provided within the <text></text> tags.
        </task>
        <instructions>
        Here are some important rules for the task:
        - Read the entire job description within <text></text> carefully.
        <text>
        {text}
        </text>
        - Based on your complete understanding of the job description, identify the two most relevant Standard Occupational Classification (SOC) major occupation code that best corresponds to the job description.
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
        Be careful, think, check your answers and only then return your response. You must not select skills at random, it must be through careful examination.""")

    embedding = ""

    taxonomy_enrich = dedent("""
        You are a helpful assistant.
        Your task is to read the skills taxonomy containing the parent node - leaf node combination within <tree></tree> tags and expand only the leaf node. You must make use of all your knowledge on job postings and expand the leaf nodes using the related skills only. You must be very careful.
        <tree>
        {text}
        </tree>
        <note>
        - You must return your response in the same format of the tree.
        - You are free to expand the leaf nodes up to whatever depth you feel necessary, however make sure to add only relevant skills as nodes. Use all your knowledge to create an expanded tree and make it comprehensive.
        </note>
        Review your output for correctness and check if all instructions have been followed. Skip the explanation and the preamble and return your verified response only.""")

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

    def __init__(
        self,
        client: BedrockRuntimeClient,
        config_file: str = "",
    ):
        """
        """
        self.client = client
        if config_file:
            with open(config_file) as f:
                self.prompt_configs = json.load(f)
        else:
            with resources.open_text("jobstruct.data", "prompt_configs.json") as f:
                self.prompt_configs = json.load(f)

    @staticmethod
    def safe_json(text: str, default: Any) -> Union[Dict, List]:
        """
        Safely parse JSON output from `text` after stripping extraneous text.
        Return the `default` value if parsing fails.
        """
        log = logging.getLogger("jobstruct.Prompts.safe_json")

        text = re.sub(r"(^[^\{\[]*)|([^\]\}]*$)", "", text)
        log.debug("stripped text: {}".format(text))

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
    ) -> Union[Dict, List]:
        """
        """
        log = logging.getLogger("jobstruct.Prompts.invoke")

        if not hasattr(Prompts, name):
            raise ValueError("{name} is an unrecognized prompt")
        if name not in self.prompt_configs:
            raise ValueError("{name} is missing from prompt_configs")

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
        log.debug("'{}' body: {}".format(name, body))

        response = self.client.invoke_model(
            body=body,
            modelId=modelId,
            accept="application/json",
            contentType="application/json"
        )
        log.debug("response: {}".format(response))

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
        log.debug("result: {}".format(result))

        return result
