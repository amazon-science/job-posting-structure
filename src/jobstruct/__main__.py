# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import boto3
import jobstruct
import json
import logging
import sys
from argparse import ArgumentParser, Namespace
from botocore.config import Config
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient


def get_client(args: Namespace, timeout: int = 60) -> BedrockRuntimeClient:
    """
    Establish a Bedrock client in the specified region using the
    specified AWS profile.
    """

    if args.profile:
        session = boto3.Session(profile_name=args.profile)
    else:
        session = boto3.Session()

    return session.client(
        service_name="bedrock-runtime",
        region_name=args.region,
        config=Config(read_timeout=timeout),
    )


def run_extract(args: Namespace) -> None:
    """
    Extract structured information from a list of input files and
    write to json output.
    """

    results = []

    if args.skills:
        skills = jobstruct.SkillsTaxonomyAI.from_file(args.skills)
    else:
        skills = None

    def constructor(filename):
        if filename.endswith(".html") or filename.endswith(".htm"):
            return jobstruct.JobStructAI.from_html_file
        else:
            return jobstruct.JobStructAI.from_file

    results = [
        constructor(filename)(
            filename,
            get_client(args),
            skills,
            args.occupation,
            args.embedding,
            args.prompt_config,
        ).to_dict()
        for filename in args.inputs
    ]

    with open(args.output, "w") if args.output != "-" else sys.stdout as f:
        json.dump(results, f)


def run_enrich(args: Namespace) -> None:
    """
    Enrich a skills taxonomy.
    """
    if args.input:
        skills = jobstruct.SkillsTaxonomyAI.from_file(args.input)
    else:
        skills = jobstruct.SkillsTaxonomyAI()

    skills.enrich(get_client(args), args.prompt_config)

    with open(args.output, "w") if args.output != "-" else sys.stdout as f:
        json.dump(skills.to_dict(), f, indent=2)


def run_refine(args: Namespace) -> None:
    """
    Refine a skills taxonomy.
    """
    if args.input:
        skills = jobstruct.SkillsTaxonomyAI.from_file(args.input)
    else:
        skills = jobstruct.SkillsTaxonomyAI()

    skills.refine(get_client(args, timeout=600), args.prompt_config)

    with open(args.output, "w") if args.output != "-" else sys.stdout as f:
        json.dump(skills.to_dict(), f, indent=2)


def main():

    parser = ArgumentParser(description=jobstruct.__doc__)
    parser.set_defaults(run=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()

    # common arguments

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="jobstruct {}".format(jobstruct.__version__),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress progress bars and info messages",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="show all logging messages, including debugging output",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="use AWS profile when establishing Bedrock client",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="establish Bedrock client in specified region",
    )
    parser.add_argument(
        "--prompt-config",
        default="",
        help="prompt configuration file",
    )

    # extract command

    extract = subparsers.add_parser("extract")
    extract.set_defaults(run=run_extract)
    extract.add_argument(
        "inputs",
        help="input HTML or text files",
        nargs="+"
    )
    extract.add_argument(
        "-o",
        "--output",
        default="-",
        help="output file (default: stdout)",
    )
    extract.add_argument(
        "--skills",
        default="",
        help="skills taxonomy file to use for extracting skills",
    )
    extract.add_argument(
        "--occupation",
        action="store_true",
        help="estimate the occupational code from the extracted information",
    )
    extract.add_argument(
        "--embedding",
        action="store_true",
        help="estimate an embedding of the extracted information",
    )

    # enrich command

    enrich = subparsers.add_parser("enrich")
    enrich.set_defaults(run=run_enrich)
    enrich.add_argument(
        "input",
        help="input json file with starting taxonomy [default: included O*NET]",
        nargs="?",
    )
    enrich.add_argument(
        "-o",
        "--output",
        default="-",
        help="output file (default: stdout)",
    )

    # refine command

    refine = subparsers.add_parser("refine")
    refine.set_defaults(run=run_refine)
    refine.add_argument(
        "input",
        help="input json file with starting taxonomy [default: included O*NET]",
        nargs="?",
    )
    refine.add_argument(
        "-o",
        "--output",
        default="-",
        help="output file (default: stdout)",
    )

    # final parse

    args = parser.parse_args()

    # set log level

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    # run command

    args.run(args)


if __name__ == "__main__":
    main()
