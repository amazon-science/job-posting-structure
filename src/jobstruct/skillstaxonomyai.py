# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import asyncio
import json
import logging
from collections import Counter, defaultdict
from importlib import resources
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Optional, Union
from .prompts import Prompts
from .skillsnode import SkillsNode


class SkillsTaxonomyAI:
    """
    A class that represents a skills taxonomy as a tree and
    uses generative AI prompting to enrich the tree with
    additional skills granularity.
    """

    def __init__(
        self,
        tree: Optional[Dict] = None,
    ):
        """
        Expands every leaf node in the starting `taxonomy` to create an
        enriched taxonomy. If no starting taxonomy is provided, use an
        O*NET taxonomy that is included in the package data.
        """
        if tree is None:
            with resources.open_text("jobstruct.data", "onet_taxonomy_renamed.json") as f:
                tree = json.load(f)
        self.root = SkillsNode.from_tree_dict(tree)
        self.names = set(self.root.names())

    @classmethod
    def from_file(cls, filename: str) -> "SkillsTaxonomyAI":
        """
        Creates a SkillsTaxonomyAI object from the tree JSON in `filename`.
        """
        with open(filename) as f:
            return cls(json.load(f))

    async def enrich(
        self,
        client: BedrockRuntimeClient,
        config_file: str = "",
    ) -> None:
        """
        Enrich the taxonomy by expanding each leaf node through generative
        AI prompting. Skip leaf nodes that have been marked as terminal in
        previous iterations (e.g., if they contain duplicates of existing
        skills in the taxonomy).
        """
        # Setup logging
        log = logging.getLogger("jobstruct.SkillsTaxonomyAI.enrich")

        # Load prompts
        prompts = Prompts(client, config_file)

        # Use a semaphore to limit the number of concurrent tasks if needed
        semaphore = asyncio.Semaphore(2)  # Adjust based on your API limits

        tasks = []
        for leaf in self.root.leaves():

            # Skip terminal nodes, which expanded to duplicate skills in a previous iteration.
            if leaf.attributes.get("terminal"):
                log.info(f"skipping terminal node '{leaf.name}'")
                continue

            # Create a task for each leaf node
            task = asyncio.create_task(
                self._process_leaf(leaf, prompts, semaphore)
            )
            tasks.append(task)

        # Await the completion of all tasks
        # await asyncio.gather(*tasks)
        await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Enriching Skills Taxonomy")

    async def _process_leaf(self, leaf, prompts, semaphore):
        async with semaphore:
            log = logging.getLogger("jobstruct.SkillsTaxonomyAI._process_leaf")
            # Create a query tree that includes the leaf and its parent.
            query = leaf.parent.to_dict()
            query["children"] = leaf.to_dict()

            try:
                response = await prompts.invoke_async(
                    "taxonomy_enrich", json.dumps(query)
                )
                result = Prompts.safe_json(
                    response.removeprefix("<tree>").removesuffix("</tree>"),
                    {}
                )

                # Parse the prompt result.
                try:
                    root = SkillsNode.from_tree_dict(result)
                except Exception:
                    log.warning(
                        f"could not parse prompt result for leaf node '{leaf.name}': {result}"
                    )
                    return
                if len(root.children) != 1:
                    log.warning(
                        f"prompt result includes siblings of leaf node '{leaf.name}': {root.to_tree_dict()}"
                    )
                    return
                root = root.children[0]
                if root.name != leaf.name:
                    log.warning(
                        f"prompt result does not align with leaf node '{leaf.name}': {root.to_tree_dict()}"
                    )
                    return

                # Determine if the result contains a duplicate of an existing skill.
                terminal = any(name in self.names for name in root.names()[1:])

                for child in root.children:
                    if not child.name.endswith(" Skills"):
                        leaf.add_child(child)
                        child.attributes["terminal"] = terminal
                        self.names.add(child.name)  # Update names to include new skills

            except Exception as e:
                log.warning(f"Error processing leaf node '{leaf.name}': {e}")

    async def prune(
        self,
        client: BedrockRuntimeClient,
        config_file: str = "",
        debug: bool = False
    ) -> None:
        """
        Prune the taxonomy by identifying and removing duplicate skills using
        code logic to ensure only unique skills remain in the taxonomy.
        The LLM assists in deciding which occurrence to keep.

        If `debug` is True, prints debugging information at each step.
        """
        log = logging.getLogger("jobstruct.SkillsTaxonomyAI.prune")

        prompts = Prompts(client, config_file)

        skill_occurrences = {}

        def traverse(node: SkillsNode, parent_path=None):
            parent_path = parent_path or []
            current_path = parent_path + [node.name]

            if node.name in skill_occurrences:
                skill_occurrences[node.name].append(current_path)
            else:
                skill_occurrences[node.name] = [current_path]

            for child in node.children:
                traverse(child, current_path)

        traverse(self.root)

        duplicates = {name: paths for name, paths in skill_occurrences.items() if len(paths) > 1}

        tasks = []
        semaphore = asyncio.Semaphore(3)  # Adjust based on your API limits
        lock = asyncio.Lock()  # To ensure thread-safe modifications

        for skill_name, paths in duplicates.items():
            task = asyncio.create_task(
                self._process_duplicate(skill_name, paths, prompts, semaphore, lock, debug)
            )
            tasks.append(task)

        # Await the completion of all tasks
        await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Enriching Skills Taxonomy")

    async def _process_duplicate(self, skill_name, paths, prompts, semaphore, lock, debug=False):
        async with semaphore:
            log = logging.getLogger("jobstruct.SkillsTaxonomyAI._process_duplicate")

            if debug:
                print(f"\nProcessing duplicates for skill: {skill_name}")
                for idx, path in enumerate(paths):
                    print(f"Occurrence {idx + 1}: Path: {path}, Parent: {path[-2] if len(path) > 1 else 'None'}")

            query = {
                "duplicate": skill_name,
                "contexts": [
                    {
                        "path": path,
                        "parent": path[-2] if len(path) > 1 else None
                    }
                    for path in paths
                ]
            }

            if debug:
                print("\nSending query to LLM:")
                print(json.dumps(query, indent=4))

            query_json = json.dumps(query)
            try:
                response = await prompts.invoke_async("taxonomy_prune", query_json)
                result = Prompts.safe_json(response.removeprefix("<tree>").removesuffix("</tree>"), {})

                if debug:
                    print("\nResult from LLM:")
                    print(json.dumps(result, indent=4))

                if 'keep' in result and result['keep']:
                    keep_path = result['keep']['path']
                else:
                    keep_path = paths[0]

                paths_to_remove = [path for path in paths if path != keep_path]

                if debug:
                    print("\nPath to keep:")
                    print(keep_path)
                    print("\nPaths to remove:")
                    print(paths_to_remove)

                # Use the lock when modifying the tree
                async with lock:
                    for path in paths_to_remove:
                        self._remove_node_by_path(path, debug)

            except Exception as e:
                log.warning(f"Could not parse prompt result for duplicate '{skill_name}': {e}")
                if debug:
                    print(f"Warning: Could not parse result for duplicate {skill_name}")
                    print(f"Exception: {e}")
                # Use the lock when modifying the tree
                async with lock:
                    paths_to_remove = paths[1:]
                    for path in paths_to_remove:
                        self._remove_node_by_path(path, debug)

    def _remove_node_by_path(self, path, debug=False):
        """
        Remove the node at the specified path.
        """
        if len(path) < 2:
            if debug:
                print(f"Cannot remove root node: {path}")
            return

        parent_path = path[:-1]
        node_name_to_remove = path[-1]

        parent_node = self._get_node_by_path(parent_path)
        if parent_node is None:
            if debug:
                print(f"Parent node not found for path: {parent_path}")
            return

        child_found = False
        for idx, child in enumerate(parent_node.children):
            if child.name == node_name_to_remove:
                if debug:
                    print(f"Removing node: {child.name} under parent: {parent_node.name}")
                del parent_node.children[idx]
                child_found = True
                break
        if not child_found and debug:
            print(f"Node to remove not found: {node_name_to_remove} under parent: {parent_node.name}")

    def _get_node_by_path(self, path):
        """
        Helper function to get node by path.
        Returns the node at the specified path.
        """
        node = self.root
        if node.name != path[0]:
            return None
        for name in path[1:]:
            found = False
            for child in node.children:
                if child.name == name:
                    node = child
                    found = True
                    break
            if not found:
                return None
        return node

    def refine(
            self,
            client: BedrockRuntimeClient,
            config_file: str = "",
    ) -> None:
        """
        Refine the taxonomy through generative AI prompting.
        """
        # Setup logging
        log = logging.getLogger("jobstruct.SkillsTaxonomyAI.refine")

        # Load prompts
        prompts = Prompts(client, config_file)

        # Create query from current taxonomy
        query = json.dumps(self.root.to_tree_dict(attributes=True))
        result = (
            prompts
            .invoke("taxonomy_refine", query)
            .removeprefix("<tree>")
            .removesuffix("</tree>")
        )
        print(result)
        result = Prompts.safe_json(result, {})

        # Parse the prompt result.
        try:
            self.root = SkillsNode.from_tree_dict(result)
        except:
            log.warn("could not parse prompt result: {}".format(result))

    def to_dict(self, attributes: bool = True) -> Dict:
        """
        Convert the SkillsTaxonomyAI object to a dictionary.
        """
        return self.root.to_tree_dict(attributes=attributes)

    def to_keys(self) -> List[str]:
        """
        Convert the SkillsTaxonomyAI object to a list of skill keys.
        """
        return self.root.to_keys()

    def to_list(self) -> List[str]:
        """
        Convert the SkillsTaxonomyAI object to a list of skill names.
        """
        return self.root.to_list()

    def to_taxonomy(self) -> str:
        """
        Convert the SkillsTaxonomyAI object to a taxonomy string.
        """
        return self.root.to_taxonomy()

    def to_yaml(self) -> str:
        """
        Convert the SkillsTaxonomyAI object to a YAML string.
        """
        return self.root.to_tree_string(yaml=True)

    def __str__(self) -> str:
        """
        String representation of the SkillsTaxonomyAI object showing
        the node names in the tree.
        """
        return self.root.to_tree_string(yaml=False)

    def to_tree_string_with_duplicates(self) -> str:
        """
        String representation of the SkillsTaxonomyAI object showing
        the node names in the tree, indented by level, and marking duplicates
        based on global occurrences.
        """
        from collections import defaultdict

        skill_counts = defaultdict(int)

        def count_skills(node: "SkillsNode") -> None:
            skill_counts[node.name] += 1
            for child in node.children:
                count_skills(child)

        count_skills(self.root)

        result = []

        def traverse(node: "SkillsNode", level: int) -> None:
            indent = "    " * level
            duplicate_str = " (duplicate)" if skill_counts[node.name] > 1 else ""
            result.append(f"{indent}- {node.name}{duplicate_str}")
            for child in node.children:
                traverse(child, level + 1)

        traverse(self.root, 0)
        return "\n".join(result)


    def print_taxonomy_stats(self) -> dict:
        """
        Analyzes the taxonomy tree and returns statistics in a dictionary, including:
        - Total number of skills.
        - Number of skills at each level.
        - Maximum depth of the tree.
        - Number of duplicate skills.
        - List of duplicate skills.
        - Average branching factor.
        - List of leaf nodes.
        - Skills with the most children.
        """
        total_skills = 0
        level_counts = defaultdict(int)
        skill_names = []
        max_depth = 0

        # Traverse the tree to collect stats
        def traverse(node: "SkillsNode", depth: int) -> None:
            nonlocal total_skills, max_depth
            total_skills += 1
            level_counts[depth] += 1
            max_depth = max(max_depth, depth)
            skill_names.append(node.name)
            for child in node.children:
                traverse(child, depth + 1)

        traverse(self.root, 0)

        # Calculate duplicates
        skill_name_counts = Counter(skill_names)
        duplicate_skills = [name for name, count in skill_name_counts.items() if count > 1]
        num_duplicates = sum(count - 1 for count in skill_name_counts.values() if count > 1)

        # Calculate average branching factor (number of child nodes per skill)
        total_branches = sum(len(node.children) for node in self._all_nodes())
        average_branching_factor = total_branches / total_skills if total_skills > 0 else 0

        # List of leaf nodes
        leaf_nodes = [name for name, count in skill_name_counts.items() if count == 1 and self._is_leaf_node(name)]

        # Skills with the most children
        max_children = 0
        skills_with_most_children = []
        for node in self._all_nodes():
            num_children = len(node.children)
            if num_children > max_children:
                max_children = num_children
                skills_with_most_children = [node.name]
            elif num_children == max_children:
                skills_with_most_children.append(node.name)

        # Create a dictionary with all the stats
        stats = {
            "total_skills": total_skills,
            "level_counts": dict(level_counts),
            "max_depth": max_depth,
            "num_duplicates": num_duplicates,
            "duplicate_skills": duplicate_skills,
            "average_branching_factor": average_branching_factor,
            "leaf_nodes": leaf_nodes,
            "skills_with_most_children": skills_with_most_children,
            "max_children": max_children
        }

        return stats

    def _all_nodes(self):
        """
        Generator that yields all nodes in the tree.
        """
        def traverse(node):
            yield node
            for child in node.children:
                yield from traverse(child)
        return traverse(self.root)

    def _is_leaf_node(self, name):
        """
        Determines if a node with the given name is a leaf node (has no children).
        """
        for node in self._all_nodes():
            if node.name == name:
                return len(node.children) == 0
        return False
