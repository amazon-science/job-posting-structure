import sys
import os
import json
import re
import boto3
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import secrets  

# ---------------------------------------------------------------------------
# 1. CENTRALIZED PARAMETERS / CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    # Paths
    "JOBSTRUCT_PATH": "/home/fkkarami/workspace/amazon/job-posting-structure/src",
    "EXTRACTED_DATA_PATH": "auxiliary_files/data_deduplicated.parquet",
    "TAXONOMY_FILE": "auxiliary_files/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json",
    "SOC_CODES_PATH": "auxiliary_files/soc_codes.json",  # <-- NEW: path to your soc_codes.json

    # Bedrock / AWS Settings
    "BEDROCK_PROFILE": "pssl-bedrock",
    "BEDROCK_REGION": "us-east-1",
    "NOVA_MODEL_ID": "us.amazon.nova-pro-v1:0",  # Adjust if needed

    # Inference / Async Settings
    "SAMPLES_NUMBER": 100,
    "SEMAPHORE_LIMIT": 1,
    "BATCH_SIZE": 100,

    # Results Output Directory (the session subfolder will be created inside this directory)
    "RESULTS_DIR": "on_demand_results_async",

    # Debug Mode (True/False)
    "DEBUG": False
}

# ---------------------------------------------------------------------------
# 2. PATH / ENV SETUP
# ---------------------------------------------------------------------------
if CONFIG["JOBSTRUCT_PATH"] not in sys.path:
    sys.path.append(CONFIG["JOBSTRUCT_PATH"])

from jobstruct import Prompts  # We only use Prompts for prompt templates

# ---------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def strip_code_fences(value: str) -> str:
    """
    Removes any ``` or ```json fences (and whitespace) from the string.
    """
    # Remove occurrences of ```json, ``` or ```
    stripped = re.sub(r'```\s*json|```', '', value)
    return stripped.strip()

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
            # 1. Remove code fences
            clean_value = strip_code_fences(value)
            # 2. Now parse the cleaned JSON
            parsed_value = json.loads(clean_value)

            if isinstance(parsed_value, dict) and "occupation" in parsed_value:
                processed_results[key] = parsed_value["occupation"]
        except json.JSONDecodeError:
            print(f"Failed to parse value for key: {key}")
            continue
    return processed_results

def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\n+', ' ', text)            # Replace multiple \n with single space
    text = re.sub(r'(?<!\s)\s(?!\s)', '', text)  # Remove spaces between letters within words
    text = re.sub(r'\s+', ' ', text)            # Collapse multiple spaces into one
    return text.strip()

def get_bedrock_runtime_client(profile_name, region_name):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# 4. NOVA INFERENCE FUNCTIONS (skills & occupation)
# ---------------------------------------------------------------------------
async def run_nova_skills_inference(
    client,
    text: str,
    skills_taxonomy_str: str,
) -> str:
    """
    Builds and sends a request to the Nova model using Prompts.skills text.
    Returns the raw string response from the model.
    """
    # -----------------------------------------------------------------------
    # 1. Create the final user prompt by filling in Prompts.skills
    # -----------------------------------------------------------------------
    user_prompt = Prompts.skills.format(text=text, skills=skills_taxonomy_str)

    # Debug print: Show the final prompt if CONFIG["DEBUG"] is True
    if CONFIG["DEBUG"]:
        print("\n[DEBUG] Nova Skills Prompt:\n---------------------------------")
        print(user_prompt)
        print("---------------------------------")

    # -----------------------------------------------------------------------
    # 2. Build system message, user messages, and inference config
    # -----------------------------------------------------------------------
    system_list = [
        {
            "text": ("You are a helpful assistant. Extract up to 10 matching "
                     "skills from the user's text and output in valid JSON.")
        }
    ]
    message_list = [
        {
            "role": "user",
            "content": [
                {"text": user_prompt}
            ]
        }
    ]
    inf_params = {
        "max_new_tokens": 500,
        "top_p": 0.9,
        "top_k": 20,
        "temperature": 0.0
    }

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    # -----------------------------------------------------------------------
    # 3. Call the Nova model (simple invoke, not streaming)
    # -----------------------------------------------------------------------
    response = client.invoke_model(
        modelId=CONFIG["NOVA_MODEL_ID"],
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json"
    )

    # -----------------------------------------------------------------------
    # 4. Parse the response
    # -----------------------------------------------------------------------
    response_payload = response["body"].read().decode("utf-8")
    response_json = json.loads(response_payload)

    # Extract text from "output -> message -> content -> [0] -> text"
    model_output = response_json["output"]["message"]["content"][0]["text"]

    # Debug print: Show the model output if CONFIG["DEBUG"] is True
    if CONFIG["DEBUG"]:
        print("\n[DEBUG] Nova Skills Model Output:\n---------------------------------")
        print(model_output)
        print("---------------------------------")

    return model_output


async def run_nova_occupation_inference(
    client,
    text: str,
    soc_codes_str: str
) -> str:
    """
    Builds and sends a request to the Nova model using Prompts.occupation text
    (including the SOC codes), and returns the raw string response.
    """
    # Use the occupation prompt template from jobstruct
    user_prompt = Prompts.occupation.format(text=text, soc_codes=soc_codes_str)

    # Debug print: Show the final prompt if CONFIG["DEBUG"] is True
    if CONFIG["DEBUG"]:
        print("\n[DEBUG] Nova Occupation Prompt:\n---------------------------------")
        print(user_prompt)
        print("---------------------------------")

    system_list = [
        {
            "text": ("You are a helpful assistant. Identify the top 1-2 SOC "
                     "codes that best match the user's job posting. Provide a JSON output.")
        }
    ]
    message_list = [
        {
            "role": "user",
            "content": [
                {"text": user_prompt}
            ]
        }
    ]
    inf_params = {
        "max_new_tokens": 500,
        "top_p": 0.9,
        "top_k": 20,
        "temperature": 0.0
    }

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    response = client.invoke_model(
        modelId=CONFIG["NOVA_MODEL_ID"],
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json"
    )

    response_payload = response["body"].read().decode("utf-8")
    response_json = json.loads(response_payload)

    model_output = response_json["output"]["message"]["content"][0]["text"]

    if CONFIG["DEBUG"]:
        print("\n[DEBUG] Nova Occupation Model Output:\n---------------------------------")
        print(model_output)
        print("---------------------------------")

    return model_output


# ---------------------------------------------------------------------------
# 5. DATA PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_records(df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        job_guid = row.get("job_guid", "")
        job_title = clean_text(row.get("external_title", ""))
        details = clean_text("\n".join(row.get("external_qualifications", [])))
        required = clean_text("\n".join(row.get("basic_qualifications", [])))
        preferred = clean_text("\n".join(row.get("preferred_qualifications", [])))

        input_text = "\n\n".join([job_title, details, required, preferred])
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


# ---------------------------------------------------------------------------
# 6. PROCESSING A SINGLE RECORD
# ---------------------------------------------------------------------------
async def process_record(
    client,
    job_guid: str,
    input_text: str,
    skills_taxonomy_str: str,
    soc_codes_str: str,
    skills_output_dict: dict,
    occupation_output_dict: dict,
    semaphore: asyncio.Semaphore,
    counters: dict
):
    """
    Asynchronously processes a single job record by calling our Nova inference functions,
    storing results into the dictionaries.
    """
    async with semaphore:
        # Move from "pending" to "running"
        counters["running"] += 1
        counters["pending"] -= 1

        print(f"[INFO] Starting inference for job_guid={job_guid}. "
              f"(Pending={counters['pending']}, Running={counters['running']}, Completed={counters['completed']})")

        # Launch both tasks
        skills_task = run_nova_skills_inference(client, input_text, skills_taxonomy_str)
        occupation_task = run_nova_occupation_inference(client, input_text, soc_codes_str)

        skills_result, occupation_result = await asyncio.gather(skills_task, occupation_task)
        skills_output_dict[job_guid] = skills_result
        occupation_output_dict[job_guid] = occupation_result

        # Move from "running" to "completed"
        counters["running"] -= 1
        counters["completed"] += 1

        print(f"[INFO] Finished inference for job_guid={job_guid}. "
              f"(Pending={counters['pending']}, Running={counters['running']}, Completed={counters['completed']})")


# ---------------------------------------------------------------------------
# 7. MAIN ASYNC FUNCTION
# ---------------------------------------------------------------------------
async def main_async():
    # Generate a unique session ID: date/time + short UBID
    session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_ubid = secrets.token_hex(2)  # e.g., a short hex string for uniqueness
    session_id = f"{session_ts}_{short_ubid}"

    print(f"[INFO] Generated session_id: {session_id}")

    # Create a subfolder named after the session_id
    session_dir = os.path.join(CONFIG["RESULTS_DIR"], session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[INFO] Created session directory: {session_dir}")

    # Load data
    extracted_data_path = CONFIG["EXTRACTED_DATA_PATH"]
    print(f"[INFO] Loading extraction data from {extracted_data_path}...")
    df = pd.read_parquet(extracted_data_path).iloc[:CONFIG['SAMPLES_NUMBER']]
    df = df.drop_duplicates(subset=['job_guid'])
    print(f"[INFO] Extraction data loaded. Total records: {len(df)}")
    print(f"[INFO] Number of unique job IDs in the data: {df['job_guid'].nunique()}")

    # Load taxonomy and build compact hierarchy
    taxonomy_file = CONFIG["TAXONOMY_FILE"]
    print(f"[INFO] Loading taxonomy from {taxonomy_file}...")
    with open(taxonomy_file) as f:
        skills_taxonomy = json.load(f)
    skills_taxonomy_compact = build_compact_hierarchy(skills_taxonomy)
    skills_taxonomy_str = json.dumps(skills_taxonomy_compact)
    print("[INFO] Taxonomy loaded and converted to compact hierarchy (and then to JSON string).")

    # Load SOC codes
    soc_codes_path = CONFIG["SOC_CODES_PATH"]
    print(f"[INFO] Loading SOC codes from {soc_codes_path}...")
    with open(soc_codes_path, "r") as f:
        soc_codes = json.load(f)
    # Convert SOC codes to a JSON string so it can be embedded in the prompt
    soc_codes_str = json.dumps(soc_codes)
    print("[INFO] SOC codes loaded and converted to JSON string.")

    # Preprocess records
    records = preprocess_records(df)
    print(f"[INFO] Preprocessed {len(records)} records.\n")

    # Prepare results dictionaries
    skills_output_dict = {}
    occupation_output_dict = {}

    # Initialize Bedrock runtime client
    print("[INFO] Initializing Bedrock runtime client for Nova model...")
    bedrock_client = get_bedrock_runtime_client(CONFIG["BEDROCK_PROFILE"], CONFIG["BEDROCK_REGION"])
    print("[INFO] Initialization complete.\n")

    # Batching parameters
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    total_records = len(records)
    total_batches = (total_records // BATCH_SIZE) + (1 if total_records % BATCH_SIZE else 0)

    print(f"[INFO] Starting async inference in {total_batches} batches (batch size = {BATCH_SIZE}).\n")

    # Track how many total queries are pending, running, completed
    counters = {
        "pending": len(records),
        "running": 0,
        "completed": 0
    }

    # Semaphore limit
    semaphore = asyncio.Semaphore(CONFIG["SEMAPHORE_LIMIT"])

    # Process in batches
    for batch_index in range(total_batches):
        start_i = batch_index * BATCH_SIZE
        end_i = start_i + BATCH_SIZE
        batch = records[start_i:end_i]

        print(f"[INFO] Processing batch {batch_index+1}/{total_batches} with {len(batch)} records...")

        tasks = []
        for (job_guid, input_text) in batch:
            task = process_record(
                bedrock_client,
                job_guid,
                input_text,
                skills_taxonomy_str,
                soc_codes_str,
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

        # Partial save
        print(f"[INFO] [Batch {batch_index+1}/{total_batches}] Saving partial results...")

        partial_skills_processed = post_process_skills_results(skills_output_dict)
        partial_occupations_processed = post_process_occupation_results(occupation_output_dict)

        partial_skills_path = os.path.join(session_dir, f"skills_output_part_{batch_index+1}.json")
        partial_occupation_path = os.path.join(session_dir, f"occupation_output_part_{batch_index+1}.json")

        with open(partial_skills_path, "w") as f:
            json.dump(partial_skills_processed, f, indent=2)
        with open(partial_occupation_path, "w") as f:
            json.dump(partial_occupations_processed, f, indent=2)

        print(f"    [Partial Save] Skills -> {partial_skills_path}")
        print(f"    [Partial Save] Occupations -> {partial_occupation_path}\n")

    # Final save
    skills_output_path = os.path.join(session_dir, "skills_output.json")
    occupation_output_path = os.path.join(session_dir, "occupation_output.json")

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
    print(f"[INFO] Session ID for reference: {session_id}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main_async())
