import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import boto3
import jobstruct

test_taxonomy = {
    'name': 'Skills',
    'children': [
        {
            'name': 'Communication',
            'children': [
                {'name': 'Writing', 'children': [], 'attributes': {}},
                {'name': 'Speaking', 'children': [], 'attributes': {}},
                {'name': 'Teamwork', 'children': [], 'attributes': {}},  # New skill
            ],
            'attributes': {}
        },
        {
            'name': 'Technical',
            'children': [
                {
                    'name': 'Programming',
                    'children': [
                        {'name': 'Python', 'children': [], 'attributes': {}},  # Existing duplicate
                        {'name': 'Java', 'children': [], 'attributes': {}},
                        {'name': 'C#', 'children': [], 'attributes': {}},       # New skill
                    ],
                    'attributes': {}
                },
                {
                    'name': 'Data Analysis',
                    'children': [
                        {'name': 'Python', 'children': [], 'attributes': {}},  # Existing duplicate
                        {'name': 'SQL', 'children': [], 'attributes': {}},
                        {'name': 'R', 'children': [], 'attributes': {}},       # New skill
                    ],
                    'attributes': {}
                },
                {
                    'name': 'Project Management',  # Duplicate skill
                    'children': [
                        {'name': 'Agile Methodologies', 'children': [], 'attributes': {}},
                        {'name': 'Scrum', 'children': [], 'attributes': {}},
                    ],
                    'attributes': {}
                },
            ],
            'attributes': {}
        },
        {
            'name': 'Management',
            'children': [
                {'name': 'Leadership', 'children': [], 'attributes': {}},       # Duplicate skill
                {'name': 'Project Management', 'children': [], 'attributes': {}},  # Duplicate skill
                {'name': 'Strategic Planning', 'children': [], 'attributes': {}},  # New skill
            ],
            'attributes': {}
        },
        {
            'name': 'Leadership',  # Duplicate skill at the root level
            'children': [],
            'attributes': {}
        },
        {
            'name': 'Python',  # Duplicate skill at the root level (existing)
            'children': [],
            'attributes': {}
        },
        {
            'name': 'Creativity',  # New skill at the root level
            'children': [],
            'attributes': {}
        }
    ],
    'attributes': {}
}




REGION_NAME = "us-east-1"

async def main():
    client = boto3.client("bedrock-runtime", region_name=REGION_NAME)

    # Load or define your taxonomy
    taxonomy_ai = jobstruct.skillstaxonomyai.SkillsTaxonomyAI(tree=test_taxonomy)

    # Enrich
    await taxonomy_ai.enrich(client)

    with open('tests/jobstruct/enriched_tree.txt', 'w') as f:
        f.write(taxonomy_ai.to_tree_string_with_duplicates())
    post_stats = taxonomy_ai.print_taxonomy_stats()
    with open('tests/jobstruct/enriched_stats.json', 'w') as f:
        json.dump(post_stats, f)

    # Prune
    await taxonomy_ai.prune(client)

    with open('tests/jobstruct/pruned_tree.txt', 'w') as f:
        f.write(taxonomy_ai.to_tree_string_with_duplicates())
    post_stats = taxonomy_ai.print_taxonomy_stats()
    with open('tests/jobstruct/pruned_stats.json', 'w') as f:
        json.dump(post_stats, f)



if __name__ == "__main__":
    asyncio.run(main())
