# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import logging
from bs4 import BeautifulSoup
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from typing import Any, Callable, List, Optional
from .prompts import Prompts
from .jobstructhtml import JobStructHTML
from .skillstaxonomyai import SkillsTaxonomyAI

class JobStructAI:
    """
    A class that represents a job posting that has been structured
    through Generative AI prompting, starting from either a text
    filename, an HTML filename, a text string, or an HTML string.
    The structured fields are:

        job_title: str
        details: List[str]
        required:
            education: str
            major: List[str]
            experience: int
            qualifications: List[str]
        preferred:
            education: str
            major: List[str]
            experience: int
            qualifications: List[str]
        benefits: List[str]
        salary: List[float]
        wage: List[float]
        entry_level: bool
        college_degree: bool
        full_time: bool
        remote: bool

    Optionally, the following fields can be estimated with additional
    prompting:

        skills: List[str]
        occupation: List[str]
        embedding: List[float]
    """

    def __init__(
        self,
        text: str,
        client: BedrockRuntimeClient,
        skills: Optional[SkillsTaxonomyAI] = None,
        occupation: bool = False,
        embedding: bool = False,
        config_file: str = "",
    ):
        """
        Extracts structured fields from the job posting `text` using
        a Generative AI prompt and provides them as attributes. If no
        `text` is provided, returns an empty structure.

        Runs additional prompts to provide attributes for `skills`,
        `occupation`, and `embedding` if those parameters are provided.

        Optionally, provide the path to a JSON `config_file` that overrides
        prompt configurations. See the file `prompt_configs.json` in the
        package for the default configurations.
        """

        prompts = Prompts(client, config_file)

        # Extract
        if text:
            result = Prompts.safe_json(prompts.invoke("extract", text), {})
        else:
            result = {}

        # Data structure
        self.job_title      = JobStructAI.validate_field(result.get("job_title"), str)
        self.details        = JobStructAI.validate_list(result.get("details", []), str)
        self.required = {
            "education"     : JobStructAI.validate_field(result.get("required", {}).get("education"), str),
            "major"         : JobStructAI.validate_list(result.get("required", {}).get("major", []), str),
            "experience"    : JobStructAI.validate_field(result.get("required", {}).get("experience"), int),
            "qualifications": JobStructAI.validate_list(result.get("required", {}).get("qualifications", []), str),
        }
        self.preferred = {
            "education"     : JobStructAI.validate_field(result.get("preferred", {}).get("education"), str),
            "major"         : JobStructAI.validate_list(result.get("preferred", {}).get("major", []), str),
            "experience"    : JobStructAI.validate_field(result.get("preferred", {}).get("experience"), int),
            "qualifications": JobStructAI.validate_list(result.get("preferred", {}).get("qualifications", []), str),
        }
        self.benefits       = JobStructAI.validate_list(result.get("benefits", []), str)
        self.salary         = JobStructAI.validate_list(result.get("salary", []), float)
        self.wage           = JobStructAI.validate_list(result.get("wage", []), float)
        self.entry_level    = JobStructAI.validate_field(result.get("entry_level"), bool)
        self.college_degree = JobStructAI.validate_field(result.get("college_degree"), bool)
        self.full_time      = JobStructAI.validate_field(result.get("full_time"), bool)
        self.remote         = JobStructAI.validate_field(result.get("remote"), bool)
        self.skills         = []
        self.occupation     = []
        self.embedding      = None

        # Use extracted details/qualifications as input for skills, occupation,
        # and embedding.
        text = "\n\n".join([
            self.job_title,
            "\n".join(self.details),
            "\n".join(self.required["qualifications"]),
            "\n".join(self.preferred["qualifications"]),
        ])
        if text.strip():
            if skills is not None:
                self.skills = list(sorted(set(JobStructAI.validate_list(
                    Prompts.safe_json(
                        prompts.invoke(
                            "skills",
                            text,
                            json.dumps(skills.to_dict())
                        ),
                        []
                    ),
                    str
                ))))
            if occupation:
                self.occupation = list(sorted(set(JobStructAI.validate_list(
                    Prompts.safe_json(
                        prompts.invoke("occupation", text),
                        {}
                    ).get("occupation", []),
                    str
                ))))
            if embedding:
                self.embedding = JobStructAI.validate_list(
                    prompts.invoke("embedding", text),
                    float
                )

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
