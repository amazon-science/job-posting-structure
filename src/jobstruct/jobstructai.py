# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import logging
from typing import Any, Callable, List, Optional
from bs4 import BeautifulSoup
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from .prompts import Prompts
from .jobstructhtml import JobStructHTML
from .skillstaxonomyai import SkillsTaxonomyAI

class JobStructAI:
    """
    A class that represents a job posting that has been structured
    through Generative AI prompting.
    """

    def __init__(
        self,
        text: str,
        client: BedrockRuntimeClient,
        extract: bool = True,
        skills: Optional[SkillsTaxonomyAI] = None,
        occupation: bool = False,
        embedding: bool = False,
        config_file: str = "",
        skills_list=None,
    ):
        if skills_list is None:
            self.skills_list = self.load_strings()
        else:
            self.skills_list = skills_list

        self.results_ready = False

        self.client = client
        self.prompts = Prompts(client, config_file)
        self.text = text
        self.extract = extract
        self.skills_param = skills
        self.occupation_param = occupation
        self.embedding_param = embedding

        # Initialize placeholders for structured data
        self.results = {}
        self.skills = []
        self.occupation = []
        self.embedding = []

        self.extract_meta = {}
        self.skills_meta = {}
        self.occupation_meta = {}
        self.embedding_meta = {}

    async def create_results(self):
        """
        Asynchronously extract structured fields for the job posting.
        """
        if self.text and self.extract:
            body, self.extract_meta = await self.prompts.invoke_async("extract", self.text)
            self.results = Prompts.safe_json(body, {})
        else:
            self.results = {}

        # Populate structured fields from results
        self.job_title = self.results.get("job_title", "")
        self.details = self.results.get("details", [])
        self.required = self.results.get("required", {})
        self.preferred = self.results.get("preferred", {})
        self.benefits = self.results.get("benefits", [])
        self.salary = self.results.get("salary", [])
        self.wage = self.results.get("wage", [])
        self.entry_level = self.results.get("entry_level", False)
        self.college_degree = self.results.get("college_degree", False)
        self.full_time = self.results.get("full_time", False)
        self.remote = self.results.get("remote", False)

        self.results_ready = True

    async def create_skills(self):
        """
        Asynchronously generate skills based on the job posting details.
        """
        if not self.results_ready:
            raise RuntimeError("Results must be extracted before generating skills.")
        input_text = "\n\n".join([
            self.job_title or "",
            "\n".join(self.details or []),
            "\n".join(self.required.get("qualifications", []) or []),
            "\n".join(self.preferred.get("qualifications", []) or []),
        ])
        if input_text.strip() and self.skills_param:
            skill_result, skills_meta = await self.prompts.invoke_async("skills", input_text, self.skills_param)
            self.skills = list(sorted(set(self.validate_list(Prompts.safe_json(skill_result, []), str))))
            self.skills_meta = skills_meta

    async def create_occupation(self):
        """
        Asynchronously generate the occupation field based on the job posting details.
        """
        if not self.results_ready:
            raise RuntimeError("Results must be extracted before generating occupation.")
        input_text = "\n\n".join([
            self.job_title or "",
            "\n".join(self.details or []),
            "\n".join(self.required.get("qualifications", []) or []),
            "\n".join(self.preferred.get("qualifications", []) or []),
        ])
        if input_text.strip():
            occupation_result, occupation_meta = await self.prompts.invoke_async("occupation", input_text)
            self.occupation = list(sorted(set(self.validate_list(
                Prompts.safe_json(occupation_result, {}).get("occupation", []),
                str
            ))))
            self.occupation_meta = occupation_meta

    async def create_embedding(self):
        """
        Asynchronously generate the embedding field based on the job posting details.
        """
        if not self.results_ready:
            raise RuntimeError("Results must be extracted before generating embeddings.")
        input_text = "\n\n".join([
            self.job_title or "",
            "\n".join(self.details or []),
            "\n".join(self.required.get("qualifications", []) or []),
            "\n".join(self.preferred.get("qualifications", []) or []),
        ])
        if input_text.strip():
            embedding_result, embedding_meta = await self.prompts.invoke_async("embedding", input_text)
            self.embedding = self.validate_list(embedding_result, float)
            self.embedding_meta = embedding_meta

    @staticmethod
    def extract_leaf_names(nested_dict):
        leaf_names = []

        def recurse(node):
            # If "children" is an empty list, it's a leaf node
            if 'children' in node and not node['children']:
                leaf_names.append(node['name'])
            # Otherwise, continue recursively for each child
            elif 'children' in node:
                for child in node['children']:
                    recurse(child)

        recurse(nested_dict)
        return leaf_names

    def load_strings(self):
        with open('/local/home/fkkarami/work-desktop/NASWA/job-posting-structure/src/jobstruct/data/skill_list.json', 'r') as f:
            skills_dict = json.load(f)
        return self.extract_leaf_names(skills_dict)

    @staticmethod
    def validate_field(value: Any, type_func: Callable) -> Any:
        """
        Cast `value` with `type_func` or return None if unsuccessful.
        """
        try:
            return type_func(value)
        except:
            logging.getLogger("jobstruct.JobStructAI.validate_field").debug(
                "value '{}' did not validate as type '{}'".format(
                    str(value),
                    str(type_func),
                )
            )
            return None

    @staticmethod
    def validate_list(values: List, type_func: Callable) -> Any:
        """
        Cast each value in `values` with `type_func` and return
        a list containing only the successful ones.
        """
        return list(filter(
            lambda x: x is not None,
            [JobStructAI.validate_field(x, type_func) for x in values]
        ))

    def to_dict(self):
        """
        Convert the JobStructAI object to a dictionary containing the
        structured fields.
        """
        return {
            "job_title": self.job_title,
            "details": self.details,
            "required": self.required,
            "preferred": self.preferred,
            "benefits": self.benefits,
            "salary": self.salary,
            "wage": self.wage,
            "entry_level": self.entry_level,
            "college_degree": self.college_degree,
            "full_time": self.full_time,
            "remote": self.remote,
            "skills": self.skills,
            "occupation": self.occupation,
            "embedding": self.embedding,
        }



class JobStructAIBatch:

    def __init__(
            self,
            job_records,
            bedrock_client: BedrockRuntimeClient,
            s3_client,
            input_s3_folder_url,
            output_s3_folder_url,
            extract: bool = True,
            skills: Optional[SkillsTaxonomyAI] = None,
            occupation: bool = False,
            embedding: bool = False,
            config_file: str = "",
            skills_list = None,
    ):
        if skills_list is None:
            self.skills_list = self.load_strings()
        else:
            self.skills_list = skills_list

        self.bedrock_client = bedrock_client
        self.s3_client = s3_client
        self.prompts = Prompts(bedrock_client, config_file)
        self.job_records_text = [r[1] for r in job_records]
        self.job_records_ids = [r[0] for r in job_records]
        self.extract = extract
        self.skills_param = skills
        self.occupation_param = occupation
        self.embedding_param = embedding

        # Initialize placeholders for structured data
        self.results = {}
        self.skills = []
        self.occupation = []
        self.embedding = []

        # Process the job posting
        if self.text:
            self.create_results()
            if skills:
                self.create_skills()
            if occupation:
                self.create_occupation()
            if embedding:
                self.create_embedding()

    def load_batch_output_file(self, output_raw_data):
        results = []
        for record in output_raw_data:
            try:
                record_id = record["recordId"]
                result = record["modelOutput"]["content"]["text"]
                results.append(result)
            except Exception as e:
                logging.info(f"Record could not retieved due to {e}")

        return results

    def create_results(self, input_s3_folder_url, output_s3_folder_url):
        """
        Extract structured fields for the job posting.
        """
        self.prompts.batch_invoke('extract', input_s3_folder_url, output_s3_folder_url)



        if self.text and self.extract:
            body, self.extract_meta = self.prompts.invoke("extract", self.text)
            self.results = Prompts.safe_json(body, {})
        else:
            self.results = {}


    def create_skills(self):
        """
        Generate skills based on the job posting details.
        """
        input_text = "\n\n".join([
            self.job_title,
            "\n".join(self.details),
            "\n".join(self.required.get("qualifications", [])),
            "\n".join(self.preferred.get("qualifications", [])),
        ])
        if input_text.strip() and self.skills_param:
            skill_result, skills_meta = self.prompts.invoke("skills", input_text, self.skills_param)
            self.skills = list(sorted(set(self.validate_list(Prompts.safe_json(skill_result, []), str))))
            self.skills_meta = skills_meta

    def create_occupation(self):
        """
        Generate the occupation field based on the job posting details.
        """
        input_text = "\n\n".join([
            self.job_title,
            "\n".join(self.details),
            "\n".join(self.required.get("qualifications", [])),
            "\n".join(self.preferred.get("qualifications", [])),
        ])
        if input_text.strip():
            occupation_result, occupation_meta = self.prompts.invoke("occupation", input_text)
            self.occupation = list(sorted(set(self.validate_list(
                Prompts.safe_json(occupation_result, {}).get("occupation", []),
                str
            ))))
            self.occupation_meta = occupation_meta

    def create_embedding(self):
        """
        Generate the embedding field based on the job posting details.
        """
        input_text = "\n\n".join([
            self.job_title,
            "\n".join(self.details),
            "\n".join(self.required.get("qualifications", [])),
            "\n".join(self.preferred.get("qualifications", [])),
        ])
        if input_text.strip():
            embedding_result, embedding_meta = self.prompts.invoke("embedding", input_text)
            self.embedding = self.validate_list(embedding_result, float)
            self.embedding_meta = embedding_meta

    @staticmethod
    def extract_leaf_names(nested_dict):
        leaf_names = []

        def recurse(node):
            # If "children" is an empty list, it's a leaf node
            if 'children' in node and not node['children']:
                leaf_names.append(node['name'])
            # Otherwise, continue recursively for each child
            elif 'children' in node:
                for child in node['children']:
                    recurse(child)

        recurse(nested_dict)
        return leaf_names


    def load_strings(self):
        with open('/local/home/fkkarami/work-desktop/NASWA/job-posting-structure/src/jobstruct/data/skill_list.json', 'r') as f:
            skills_dict = json.load(f)
        # print(skills_dict)
        return self.extract_leaf_names(skills_dict)

    @staticmethod
    def validate_field(value: Any, type_func: Callable) -> Any:
        """
        Cast `value` with `type_func` or return None if unsuccessful.
        """
        try:
            return type_func(value)
        except:
            logging.getLogger("jobstruct.JobStructAI.validate_field").debug(
                "value '{}' did not validate as type '{}'".format(
                    str(value),
                    str(type_func),
                )
            )
            return None

    @staticmethod
    def validate_list(values: List, type_func: Callable) -> Any:
        """
        Cast each value in `values` with `type_func` and return
        a list containing only the successful ones.
        """
        return list(filter(
            lambda x: x is not None,
            [JobStructAI.validate_field(x, type_func) for x in values]
        ))

    @classmethod
    def from_file(
        cls,
        filename: str,
        client: BedrockRuntimeClient,
        skills: Optional[SkillsTaxonomyAI] = None,
        occupation: bool = False,
        embedding: bool = False,
        config_file: str = "",
    ) -> "JobStructAI":
        """
        Creates a JobStructAI object from the text in `filename`.
        """
        with open(filename) as f:
            text: str = f.read()
        return cls(
            text,
            client,
            skills,
            occupation,
            embedding,
            config_file,
        )

    @classmethod
    def from_html(
        cls,
        html: str,
        client: BedrockRuntimeClient,
        skills: Optional[SkillsTaxonomyAI] = None,
        occupation: bool = False,
        embedding: bool = False,
        config_file: str = "",
    ) -> "JobStructAI":
        """
        Creates a JobStructAI object from an `html` string.
        """
        soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        # Extract all text contained in the relevant HTML tags
        text = "\n".join(
            element.get_text(separator="\n").strip()
            for element in soup.body.find_all(JobStructHTML.tags)
            if all(
                element.find(tag) is None
                for tag in JobStructHTML.tags
            )
        )
        return cls(
            text,
            client,
            skills,
            occupation,
            embedding,
            config_file,
        )

    @classmethod
    def from_html_file(
        cls,
        filename: str,
        client: BedrockRuntimeClient,
        skills: Optional[SkillsTaxonomyAI] = None,
        occupation: bool = False,
        embedding: bool = False,
        config_file: str = "",
    ) -> "JobStructAI":
        """
        Creates a JobStructAI object from the HTML in `filename`.
        """
        with open(filename) as f:
            html: str = f.read()
        return cls.from_html(
            html,
            client,
            skills,
            occupation,
            embedding,
            config_file,
        )

    def to_dict(self):
        """
        Convert the JobStructAI object to a dictionary containing the
        structured fields.
        """
        return {
            "job_title"      : self.job_title,
            "details"        : self.details,
            "required"       : self.required,
            "preferred"      : self.preferred,
            "benefits"       : self.benefits,
            "salary"         : self.salary,
            "wage"           : self.wage,
            "entry_level"    : self.entry_level,
            "college_degree" : self.college_degree,
            "full_time"      : self.full_time,
            "remote"         : self.remote,
            "skills"         : self.skills,
            "occupation"     : self.occupation,
            "embedding"      : self.embedding,
        }

