import sys
jobstruct_path = "/home/fkkarami/workspace/amazon/old_ebsvolume/home/fkkarami/workspace/NASWA/job-posting-structure/src"
if jobstruct_path not in sys.path:
    sys.path.append(jobstruct_path)

import os
import json
import re
import boto3
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from jobstruct import Prompts


def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Removing extra spaces between characters.
    - Removing excessive newline characters.
    - Reassembling words if characters are split.
    - Removing leading/trailing whitespace.
    """
    if not text:
        return text
    
    # Replace multiple newline characters with a single space
    text = re.sub(r'\n+', ' ', text)
    
    # Reassemble words by removing spaces between characters but keeping proper word boundaries
    text = re.sub(r'(?<!\s)\s(?!\s)', '', text)  # Removes spaces between letters in words
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    return text.strip()


def get_bedrock_client(profile_name="pssl-bedrock", region_name="us-east-1"):
    """Initializes a Bedrock boto3 client."""
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client("bedrock-runtime")


def run_skills_inference(text: str, skills_taxonomy: str) -> str:
    """
    Invoke the 'skills' prompt on demand.
    :param text: The job posting text (title, details, required/preferred qualifications).
    :param skills_taxonomy: The JSON string of all known skills.
    :return: The raw string returned by the LLM (usually JSON).
    """
    # 1. Create Bedrock client
    bedrock_client = get_bedrock_client()
    
    # 2. Create Prompts object (point to your real config file if needed)
    prompts = Prompts(client=bedrock_client)

    # 3. Invoke the "skills" prompt
    result, _metadata = prompts.invoke(
        name="skills",
        text=text,
        skills=skills_taxonomy
    )
    return result

def run_occupation_inference(text: str) -> str:
    """
    Invoke the 'occupation' prompt on demand.
    :param text: The job posting text.
    :return: The raw string returned by the LLM (usually JSON).
    """
    # 1. Create Bedrock client
    bedrock_client = get_bedrock_client()

    # 2. Create Prompts object
    prompts = Prompts(client=bedrock_client)

    # 3. Invoke the "occupation" prompt
    result, _metadata = prompts.invoke(
        name="occupation",
        text=text,
        skills=""  # No taxonomy needed for occupation
    )
    return result


if __name__ == "__main__":

    # 1. Path to your Parquet file
    extracted_data_path = "auxiliary_files/data_deduplicated.parquet"  # Adjust as needed

    # 2. Load the Parquet file
    print(f"Loading extraction data from {extracted_data_path}...")
    amazon_job_data = pd.read_parquet(extracted_data_path).iloc[:100]
    print(f"Extraction data loaded. Total records: {len(amazon_job_data)}")

    # (Optional) If you only want a subset, e.g. rows 100,000 to 200,000:
    # amazon_job_data = amazon_job_data.iloc[100000:200000]

    # 3. Load skills taxonomy (JSON) if needed
    #    Adjust path or method as appropriate for your environment
    taxonomy_file = "auxiliary_files/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json"
    with open(taxonomy_file) as f:
        skills_taxonomy = f.read()

    # 4. Prepare results dictionaries
    skills_output_dict = {}
    occupation_output_dict = {}

    # 5. Iterate over each job record
    print("Processing records for on-demand inference...")
    for _, record in tqdm(amazon_job_data.iterrows(), total=len(amazon_job_data), desc="Preparing On-Demand Records"):
        # Extract the text fields
        job_guid = record['job_guid']
        job_title = clean_text(record['external_title'])
        details = clean_text("\n".join(record['external_qualifications']))
        required = clean_text("\n".join(record['basic_qualifications']))
        preferred = clean_text("\n".join(record['preferred_qualifications']))

        # Combine them into a single input text
        input_text = "\n\n".join([
            job_title,
            details,
            required,
            preferred,
        ])

        # 6. Run the LLM calls
        # -- Skills
        skills_result = run_skills_inference(input_text, skills_taxonomy)
        skills_output_dict[job_guid] = skills_result

        # -- Occupation
        occupation_result = run_occupation_inference(input_text)
        occupation_output_dict[job_guid] = occupation_result

    # 7. Save results to JSON
    os.makedirs("on_demand_results", exist_ok=True)

    skills_output_path = os.path.join("on_demand_results", "skills_output.json")
    with open(skills_output_path, "w") as f:
        json.dump(skills_output_dict, f, indent=2)
    print(f"Skills inference results saved to {skills_output_path}")

    occupation_output_path = os.path.join("on_demand_results", "occupation_output.json")
    with open(occupation_output_path, "w") as f:
        json.dump(occupation_output_dict, f, indent=2)
    print(f"Occupation inference results saved to {occupation_output_path}")
