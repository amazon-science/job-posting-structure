import pandas as pd

def create_two_column_df(original_df):
    """
    Converts a DataFrame with 5 columns into a 2-column DataFrame:
    - 'job_guid': taken directly from the original 'job_guid' column.
    - 'job_description': concatenation of the other four string columns.

    Args:
        original_df (pd.DataFrame): Input DataFrame with the following columns:
            - 'job_guid': Unique identifier for each job.
            - 'external_title': Job title.
            - 'external_qualifications': List of qualifications or description.
            - 'basic_qualifications': Required qualifications as a list.
            - 'preferred_qualifications': Preferred qualifications as a list.

    Returns:
        pd.DataFrame: A new DataFrame with two columns:
            - 'job_guid'
            - 'job_description': Combined string of other fields.
    """
    # Ensure the input DataFrame contains the required columns
    required_columns = [
        "job_guid",
        "external_title",
        "external_qualifications",
        "basic_qualifications",
        "preferred_qualifications",
    ]
    for col in required_columns:
        if col not in original_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create the new DataFrame
    two_column_df = pd.DataFrame()
    two_column_df["job_guid"] = original_df["job_guid"]

    # Concatenate the other columns into a single 'job_description' column
    two_column_df["job_description"] = (
        original_df["external_title"].fillna("").astype(str) + "\n\n" +
        original_df["external_qualifications"].fillna("").astype(str) + "\n\n" +
        original_df["basic_qualifications"].fillna("").astype(str) + "\n\n" +
        original_df["preferred_qualifications"].fillna("").astype(str)
    )

    # Clean up any extra spaces or newlines in the concatenated strings
    two_column_df["job_description"] = two_column_df["job_description"].str.strip()

    return two_column_df

# Example usage:
# # Sample input DataFrame
# data = {
#     "job_guid": ["job1", "job2"],
#     "external_title": ["Software Engineer", "Data Scientist"],
#     "external_qualifications": ["Bachelor's degree", "Master's degree"],
#     "basic_qualifications": ["Python, SQL", "Python, R"],
#     "preferred_qualifications": ["Experience with AWS", "Experience with TensorFlow"],
# }

# # Create the input DataFrame
# original_df = pd.DataFrame(data)

original_df = pd.read_parquet("/home/fkkarami/workspace/amazon/job-posting-structure/auxiliary_files/data_deduplicated.parquet")
original_df = original_df.iloc[:1500]
# Convert the DataFrame
two_column_df = create_two_column_df(original_df)

# Display the result
# print(two_column_df)

two_column_df.to_parquet( "/home/fkkarami/workspace/amazon/job-posting-structure/auxiliary_files/two_column_df.parquet")
