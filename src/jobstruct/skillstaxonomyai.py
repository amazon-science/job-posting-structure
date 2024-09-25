# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import json
import logging
from importlib import resources
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from typing import Dict, Optional
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

    def enrich(
        self,
        client: BedrockRuntimeClient,
        config_file: str = "",
    ) -> None:
        """
        Enrich the taxonomy by expanding each leaf node through generative
        AI prompting. Skip leaf nodes that have been marked as terminal in
        previous iterations (e.g. if they contain duplicates of existing
        skills in the taxonomy).
        """
        # Setup logging
        log = logging.getLogger("jobstruct.SkillsTaxonomyAI.enrich")
        
        # Load prompts
        prompts = Prompts(client, config_file)

        # Loop over each leaf node
        for leaf in self.root.leaves():

            # Skip terminal nodes, which expanded to duplicate skills in a previous iteration.
            if leaf.attributes.get("terminal"):
                log.info("skipping terminal node '{}'".format(leaf.name))

            # Create a query tree that includes the leaf and its parent.
            query = leaf.parent.to_dict()
            query["children"] = leaf.to_dict()
            result = Prompts.safe_json(
                (
                    prompts
                    .invoke("taxonomy_enrich", json.dumps(query))
                    .removeprefix("<tree>")
                    .removesuffix("</tree>")
                ),
                {}
            )

            # Parse the prompt result.
            try:
                root = SkillsNode.from_tree_dict(result)
            except:
                log.warn(
                    "could not parse prompt result for leaf node '{}': {}".format(
                        leaf.name,
                        result
                    )
                )
                continue
            if len(root.children) != 1:
                log.warn(
                    "prompt result includes siblings of leaf node '{}': {}".format(
                        leaf.name,
                        root.to_tree_dict()
                    )
                )
                continue
            root = root.children[0]
            if root.name != leaf.name:
                log.warn(
                    "prompt result does not align with leaf node '{}': {}".format(
                        leaf.name,
                        root.to_tree_dict()
                    )
                )
                continue

            # Determine if the result contains a duplicate of an existing skill.
            terminal = any(name in self.names for name in root.names()[1:])

            for child in root.children:
                if not child.name.endswith(" Skills"):
                    leaf.add_child(child)
                    child.attributes["terminal"] = terminal

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
        result = Prompts.safe_json(result)

        # Parse the prompt result.
        try:
            self.root = SkillsNode.from_tree_dict(result)
        except:
            log.warn("could not parse prompt result: {}".format(result))

    def to_dict(self):
        """
        Convert the SkillsTaxonomyAI object to a dictionary.
        """
        return self.root.to_tree_dict(attributes=True)

    def __str__(self) -> str:
        """
        String representation of the SkillsTaxonomyAI object showing
        the node names in the tree.
        """
        return self.root.to_tree_string()
