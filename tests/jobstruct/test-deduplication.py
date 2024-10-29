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

def main():
    client = boto3.client("bedrock-runtime", region_name=REGION_NAME)

    with open('tests/jobstruct/skilltaxonomy_deduplication_test.json', 'r') as f:
        test_taxonomy = json.load(f)

    taxonomy_ai = jobstruct.skillstaxonomyai.SkillsTaxonomyAI(tree=test_taxonomy)

    # print("Taxonomy before pruning:")
    # print(taxonomy_ai.to_tree_string_with_duplicates())

    with open('tests/jobstruct/raw_tree.txt', 'w') as f:
        f.write(taxonomy_ai.to_tree_string_with_duplicates())

    pre_stats = taxonomy_ai.print_taxonomy_stats()
    with open('tests/jobstruct/pre_stats.json', 'w') as f:
        json.dump(pre_stats, f)

    # Enrich
    print("Enrichment started.")
    asyncio.run(taxonomy_ai.enrich(client))

    print("\nTaxonomy after enrichment:")
    print(taxonomy_ai.to_tree_string_with_duplicates())
    with open('tests/jobstruct/enriched_tree.txt', 'w') as f:
        f.write(taxonomy_ai.to_tree_string_with_duplicates())

    enriched_stats = taxonomy_ai.print_taxonomy_stats()
    with open('tests/jobstruct/enriched_stats.json', 'w') as f:
        json.dump(enriched_stats, f)

    with open('tests/jobstruct/skilltaxonomy_deduplication_test_enriched.json', 'w') as f:
        json.dump(taxonomy_ai.to_dict(), f)

    # Run the prune method
    # taxonomy_ai.prune(client=client, debug=False)

    print("Deduplication started.")
    asyncio.run(taxonomy_ai.prune(client))

    print("\nTaxonomy after pruning:")
    print(taxonomy_ai.to_tree_string_with_duplicates())
    with open('tests/jobstruct/pruned_tree.txt', 'w') as f:
        f.write(taxonomy_ai.to_tree_string_with_duplicates())

    post_stats = taxonomy_ai.print_taxonomy_stats()
    with open('tests/jobstruct/pruned_stats.json', 'w') as f:
        json.dump(post_stats, f)

    with open('tests/jobstruct/skilltaxonomy_deduplication_test_pruned.json', 'w') as f:
        json.dump(taxonomy_ai.to_dict(), f)

if __name__ == "__main__":
    main()
