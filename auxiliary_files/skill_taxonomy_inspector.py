import json

def print_leaf_nodes(data):
    # Check if the current node is a terminal (leaf) node
    if "children" in data and isinstance(data["children"], list) and not data["children"]:
        # Print the name of the terminal node
        print(data["name"])
    else:
        # If there are children, recursively call the function for each child
        if "children" in data:
            for child in data["children"]:
                print_leaf_nodes(child)


def simplify_skills_tree(node: dict) -> dict:
    """
    Recursively extracts only the 'name' and 'children' fields from the original
    skill dictionary/tree, dropping everything else (e.g. 'attributes').
    """
    # Start by storing only 'name'
    simplified = {"name": node["name"]}

    # If children exist, recurse
    children = node.get("children", [])
    if children:
        simplified["children"] = [simplify_skills_tree(child) for child in children]
    else:
        # If there are no children, set it to an empty list
        simplified["children"] = []

    return simplified


def build_compact_hierarchy(node: dict) -> dict:
    """
    Recursively convert the original tree into a more compact form:
      - Drop 'name' and 'children' keys.
      - Each child's 'name' becomes a dictionary key, and its children become nested dictionaries.
      - Leaf nodes become empty dictionaries ({}).
    """
    # If there are no children, return an empty dict
    children = node.get("children", [])
    if not children:
        return {}

    # Otherwise, build a dictionary of { child_name: sub_hierarchy }
    result = {}
    for child in children:
        child_name = child["name"]
        result[child_name] = build_compact_hierarchy(child)
    return result


def gather_skill_names(skill_dict: dict) -> set:
    skill_names = set()
    for skill_name, sub_dict in skill_dict.items():
        skill_names.add(skill_name)
        if isinstance(sub_dict, dict):
            skill_names.update(gather_skill_names(sub_dict))
    return skill_names

def check_skills_in_taxonomy(job_skills: dict, skill_taxonomy: dict) -> dict:
    valid_skills = gather_skill_names(skill_taxonomy)
    result = {}
    for job_id, skills in job_skills.items():
        found = []
        not_found = []
        for skill in skills:
            if skill in valid_skills:
                found.append(skill)
            else:
                not_found.append(skill)
        result[job_id] = {
            "found_in_taxonomy": found,
            "not_in_taxonomy": not_found
        }
    return result



with open('tests/2024-10-29-Skills-Taxonomy-Proposal-1385-Skills.json', 'r') as f:
    taxonomy = json.load(f)

with open('on_demand_results_async/skills_output.json', 'r') as f:
    skills_outputs = json.load(f)

cleaned_taxonomy = build_compact_hierarchy(taxonomy)
# print(json.dumps(cleaned_taxonomy, indent=2))

validation_results = check_skills_in_taxonomy(skills_outputs, cleaned_taxonomy)

# Print results
print(json.dumps(validation_results, indent=4))