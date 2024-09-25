# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

from typing import Dict, List

class SkillsNode:
    """
    Class representing a node in the skills taxonomy.
    """

    def __init__(self, name: str, attributes: Dict = {}):
        self.name = name
        self.attributes = attributes
        self.root = True
        self.parent = None
        self.children = []

    @classmethod
    def from_dict(cls, node: Dict) -> "SkillsNode":
        """
        Convert a `node` dictionary of name/attributes into a SkillsNode,
        validating that the name and code attributes are present.
        """
        assert "name" in node, "invalid node: missing name"
        if "attributes" in node:
            assert isinstance(node["attributes"], dict), "invalid node: attributes is not a dict"
        return cls(node["name"], node.get("attributes", {}))

    @classmethod
    def from_tree_dict(cls, tree: Dict) -> "SkillsNode":
        """
        Convert a flattened `tree` into a linked list of nodes.
        Return the root SkillsNode.
        """
        root = SkillsNode.from_dict(tree)

        def traverse(node: "SkillsNode", subtree: Dict) -> None:
            if "children" in subtree:
                # Fix singleton children
                if isinstance(subtree["children"], dict):
                    subtree = subtree.copy()
                    subtree["children"] = [subtree["children"]]
                assert isinstance(subtree["children"], list), "invalid node: children is not a list"
                for child_tree in subtree["children"]:
                    child_node = node.add_child(SkillsNode.from_dict(child_tree))
                    traverse(child_node, child_tree)

        traverse(root, tree)

        return root

    def add_child(self, child: "SkillsNode") -> "SkillsNode":
        """
        Add a child SkillsNode to this SkillsNode, then return the child
        (for chaining).
        """
        self.children.append(child)

        child.parent = self
        child.root = False

        return child

    def leaves(self) -> List["SkillsNode"]:
        """
        Return a list of leaf nodes.
        """
        result = []

        def traverse(node: "SkillsNode") -> None:
            if not node.children:
                result.append(node)
            else:
                for child in node.children:
                    traverse(child)

        traverse(self)

        return result

    def names(self) -> List[str]:
        """
        Return a list of names for the node and all it's children.
        """
        result = []

        def traverse(node: "SkillsNode") -> None:
            result.append(node.name)
            for child in node.children:
                traverse(child)

        traverse(self)

        return result

    def to_dict(self, attributes: bool = False) -> Dict:
        """
        Flatten the node into a dict.
        """
        result = {
            "name": self.name,
        }
        if attributes:
            result["attributes"] = self.attributes
        return result

    def to_tree_dict(self, attributes: bool = False) -> Dict:
        """
        Flatten the node and it's children into a dict.
        Optionally include node attributes.
        """
        def traverse(node: "SkillsNode") -> Dict:
            result = {
                "name": node.name,
                "children": [traverse(child) for child in node.children],
            }
            if attributes:
                result["attributes"] = node.attributes
            return result
 
        return traverse(self)

    def to_tree_string(self) -> str:
        """
        Flatten the node and it's children into a string representation
        of node names in the tree.
        """
        result = []
        def traverse(node: "SkillsNode", level: int) -> None:
            result.append("|{} {}".format("-" * level, node.name))
            for child in node.children:
                traverse(child, level + 1)
        traverse(self, 0)
        return "\n".join(result)
