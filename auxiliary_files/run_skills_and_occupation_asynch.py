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