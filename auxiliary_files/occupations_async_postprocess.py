import json
import pandas as pd

# Path to the input JSON file
input_json_path = "/home/fkkarami/workspace/amazon/job-posting-structure/on_demand_results_async/20250120_165919_c37b/skills_output.json"

# Load the JSON file
with open(input_json_path, "r") as file:
    data = json.load(file)

# Create a DataFrame from the JSON data
df = pd.DataFrame({
    "job_guid": data.keys(),
    "skills": [";".join(value) for value in data.values()]  # Join the list of SOC codes with ';'
})

# Path to the output CSV file
output_csv_path = "/home/fkkarami/workspace/amazon/job-posting-structure/on_demand_results_async/20250120_165919_c37b/post_processed/skills_first_100.csv"

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"CSV file has been saved to {output_csv_path}.")
