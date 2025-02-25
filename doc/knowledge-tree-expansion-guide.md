# KnowledgeTaxonomy User Guide

## Overview
KnowledgeTaxonomy is a Python class for managing a hierarchical knowledge taxonomy. It uses Amazon Bedrock (LLM) to expand, reduce, and validate the taxonomy. Nodes (or "families") in the taxonomy are stored as dictionaries:

```json
{
  "Some Knowledge Family": {
    "description": "...",
    "knowledge": {
      "Sub Family A": { "description": "...", "knowledge": { ... } },
      "Sub Family B": { "description": "...", "knowledge": {} }
    }
  }
}
```

## Initialization

```python
taxonomy = KnowledgeTaxonomy(config=your_boto_config)
```
`config` is optional. If provided, it is passed as the boto3 client config for Bedrock.

## Methods

### LLM Invocation

- **get_bedrock_response(content, ...)**: Submits a prompt (content) to Amazon Bedrock, with optional parameters like model_id, temperature, etc. Returns the model's raw text output or None if it fails.

### Prompts Creation

- **create_subknowledge_prompt(area_name, current_subtree, sub_prompt, max_sub_knowledge)**: Builds an LLM prompt to add sub-knowledge under a given family.
- **create_quality_check_prompt(tree)**: Creates a prompt for performing a quality check on a knowledge subtree.
- **create_reduction_prompt(family_name, snapshot)**: Builds an LLM prompt to remove or merge sub-knowledge that's too granular or irrelevant.

### Finding Families

- **find_family_by_name(knowledge_tree, target_family)**: Finds the node (target_family) at any depth.
- **find_family_by_name_and_level(knowledge_tree, target_family, target_level)**: Locates the node with name target_family specifically at target_level.

### Expansion

- **expand_single_knowledge_area(knowledge_tree, area_name, target_level, sub_prompt, max_sub_knowledge, debug_level)**: Finds a family area_name at target_level and expands it by adding new sub-knowledge (fetched from LLM). Returns an updated copy of the tree.
- **expand_all_families(knowledge_tree, max_sub_knowledge, debug_level)**: Expands all "leaf" families in the tree by adding new sub-knowledge.

### Reduction

- **reduce_single_knowledge_area(knowledge_tree, area_name, debug_level)**: Calls the LLM to remove or merge sub-knowledge that's unfit or too granular. Updates the subtree accordingly.

### Removal of All Sub-Knowledge

- **remove_all_subknowledge(knowledge_tree, family_name, debug_level)**: Empties family_name's knowledge dictionary while keeping its description.

### Quality Checking

- **quality_check_subtree(subtree_dict, debug_level)**: Sends a subtree to the LLM for a quality check, returning the updated structure and a text description of changes.
- **quality_check_tree(knowledge_tree, enable_chunking, debug_level)**: Like quality_check_subtree, but applies to the entire tree. Can chunk the tree if enable_chunking is True.

### Tree Stats

- **compute_tree_stats(knowledge_tree)**: Returns a dictionary counting total families, leaf families, and max depth.
- **gather_all_families(knowledge_tree)**: Returns a list of all knowledge families.
- **gather_leaf_families(knowledge_tree)**: Returns a list of leaf families.

### Chunking

- **chunking_plan_prompt(tree_stats, all_families)**: Builds a prompt for the LLM to produce chunking plans.
- **extract_subtree_for_families(knowledge_tree, families_list)**: Extracts a partial knowledge tree for the specified families.
- **merge_updated_subtree_into_tree(original_tree, updated_subtree)**: Merges a partial updated subtree into the main taxonomy.

### Iterative Building

- **build_knowledge_tree(initial_tree, current_depth, last_level, max_sub_knowledge, debug_level)**: Iterates expansions and quality checks up to last_level. Saves the final version as final_knowledge_tree.json.

### Hierarchy Printing
- **print_hierarchy_tree(data, max_depth, current_level=1, prefix="")**: Recursively prints the tree hierarchy up to max_depth levels.

## Usage Flow

1. Create or load an initial knowledge tree structure.
2. Call build_knowledge_tree(...) to systematically expand and improve the taxonomy.
3. Alternatively, use expand_single_knowledge_area(...) for targeted expansions, or reduce_single_knowledge_area(...) to prune.
4. Run quality_check_tree(...) as needed.
5. Access final results or partial expansions as Python dictionaries.

## Note

- Requires AWS credentials with access to Amazon Bedrock.
- Debug verbosity (debug_level) can be set to 2 for extensive logging.

This completes the guide for the KnowledgeTaxonomy class. You may adapt prompts, model parameters, or incorporate additional logic to suit your domain.
