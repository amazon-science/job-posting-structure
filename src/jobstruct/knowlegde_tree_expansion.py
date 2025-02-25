import json
import time
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
import copy

class KnowledgeTaxonomy:
    def __init__(self, config=None):
        self.config = config

    def get_bedrock_response(self, content, max_tokens=1024*4,
                             model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                             anthropic_version="bedrock-2023-05-31", max_retries=100,
                             backoff_factor=2, temperature=0, debug_level=0):
        """
        Sends a prompt to Amazon Bedrock LLM and returns the generated response content.
        """
        if debug_level >= 2:
            print("\n--- get_bedrock_response() called ---")
            print("Prompt content:\n", content)

        client = boto3.client(service_name="bedrock-runtime", config=self.config)
        body = json.dumps({
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "anthropic_version": anthropic_version,
            "temperature": temperature
        })
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.invoke_model(body=body, modelId=model_id)
                response_body = json.loads(response.get("body").read())
                output = response_body.get("content")
                if isinstance(output, list) and len(output) > 0:
                    first_item = output[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        output = first_item["text"]
                if debug_level >= 2:
                    print("LLM response received:\n", output)
                return output
            except ClientError as e:
                retry_count += 1
                wait_time = backoff_factor ** retry_count
                print(f"ClientError occurred: {e}. Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
        print("Max retries reached. Unable to get response from Bedrock.")
        return None

    def create_subknowledge_prompt(self, 
                                   area_name, 
                                   current_subtree, 
                                   sub_prompt="", 
                                   max_sub_knowledge=10):
        """
        Creates a prompt instructing the LLM to expand a knowledge taxonomy.

        Parameters:
          - area_name (str): Name of the knowledge family being expanded.
          - current_subtree (dict): A snippet of the taxonomy containing the family structure so far.
          - sub_prompt (str): Additional instructions or context the user wants to add to the prompt.
          - max_sub_knowledge (int): Maximum number of new sub-knowledge areas to generate.

        The returned prompt instructs the LLM to produce ONLY a JSON structure that 
        should be merged under the "knowledge" key of 'area_name'.
        """
        base_prompt = f"""
        You are an AI assistant tasked with expanding a knowledge taxonomy tree. 
        Your role is to add new sub-knowledge areas under the given knowledge family, 
        ensuring that each new knowledge item is distinct from others.

        Definition:
        "Knowledge" refers to domains or fields of understanding encompassing concepts, facts, principles, 
        and information that enable individuals to comprehend and reason about a specific subject area.

        Constraints & Clarifications:
        - Knowledge is not the same as a skill. Skills are actionable, demonstrable abilities,
          whereas knowledge is the conceptual or factual understanding of a domain.
        - Knowledge can be broad or specialized, but each new entry should be relevant to the parent family.

        Context:
        The current knowledge family is named "{area_name}" and has the following JSON structure:
        {json.dumps(current_subtree, indent=2)}

        Additional instructions from user:
        {sub_prompt}

        Instructions:
        1. Add up to {max_sub_knowledge} new sub-knowledge items under "{area_name}".
        2. If you cannot generate {max_sub_knowledge} quality, relevant sub-knowledge, provide fewer.
        3. Each new knowledge item must have:
            - "description": a short string describing that knowledge area.
            - "knowledge": either an empty dictionary or further nested structure.
        4. Ensure each new sub-knowledge is clearly distinct and doesn't overlap existing siblings.

        Output Requirement:
        Return ONLY the JSON object that should be merged under the "knowledge" key of "{area_name}". 
        No extra commentary or markdown.
        """
        return base_prompt.strip()
    
    def find_family_by_name_and_level(self, knowledge_tree, target_family, target_level, current_level=1):
        """
        Recursively searches the knowledge_tree for a node named `target_family` 
        **exactly** at `target_level`. If found, returns (node, parent_dict, node_key).
        Otherwise, returns (None, None, None).

        - knowledge_tree: dictionary representing the overall knowledge structure.
        - target_family: the name of the family we want to locate.
        - target_level: the level at which this family must reside.
        - current_level: tracks the traversal depth (start with 1 at the root).

        Example:
          - If target_level=2, we only want the family name that appears at depth=2, 
            ignoring any deeper or shallower occurrences of the same name.
        """
        if not isinstance(knowledge_tree, dict):
            return None, None, None
        for family_name, family_data in knowledge_tree.items():
            if family_name == target_family and current_level == target_level:
                return family_data, knowledge_tree, family_name
            if (isinstance(family_data, dict) 
                and "knowledge" in family_data 
                and isinstance(family_data["knowledge"], dict)):
                found, parent, parent_key = self.find_family_by_name_and_level(
                    family_data["knowledge"], 
                    target_family, 
                    target_level, 
                    current_level=current_level+1
                )
                if found is not None:
                    return found, parent, parent_key
        return None, None, None

    def create_quality_check_prompt(self, tree):
        """
        Creates a prompt for verifying and improving a knowledge taxonomy. The LLM
        should add missing items and remove overly narrow or irrelevant ones.
        """
        prompt = f"""
        You are an expert in knowledge taxonomy quality control.
        Below is a knowledge taxonomy (in JSON) with families structured as:
        {{ "Family Name": {{ "description": "...", "knowledge": {{ ... }} }} }}

        ** Perform a quality check and make sure that the knowledge items are not too granular 
           and that they remain consistent with their parent and siblings.
        ** In "updated_tree", return an improved version that adds new entries where needed, 
           and omits irrelevant or too narrow keys. 
        ** In "changes_description", explain all changes (both additions and omissions) and your rationale.
        ** Return ONLY a valid JSON object with exactly these two keys: "updated_tree" and "changes_description".

        The input tree is:
        {json.dumps(tree, indent=2)}
        """
        return prompt.strip()

    def create_reduction_prompt(self, family_name, snapshot):
        """
        Creates a prompt for the LLM to remove or merge knowledge items that might be 
        unfit or too granular within a specified knowledge area.
        The LLM is expected to respond with JSON describing how to modify the subtree:
          {
            "<existing_sub_area_name>": "remove",
            "<existing_sub_area_name>": { "merge_into": "someOtherArea" },
            ...
          }
        or an empty dict if no changes are recommended.
        """
        prompt_text = f"""
        You are an AI assistant that reviews a knowledge family for quality. Some sub-knowledge items might be unfit 
        with respect to their parent or siblings, or too granular.

        The JSON below shows the current '{family_name}' subtree:

        {json.dumps(snapshot, indent=2)}

        Please return a JSON object indicating which sub-knowledge items should be removed or merged. Possible actions:
        1) "<sub_area_name>": "remove"
        2) "<sub_area_name>": {{ "merge_into": "<another_sub_area_name>" }}

        If no changes are needed, return an empty JSON ({{}}).
        Return ONLY the JSON. No extra commentary.
        """
        return prompt_text.strip()

    def find_family_by_name(self, knowledge_tree, target_family):
        """
        Recursively searches the knowledge_tree for a node named target_family.
        Returns (node, parent_dict, node_key) or (None, None, None) if not found.
        """
        if not isinstance(knowledge_tree, dict):
            return None, None, None
        for family_name, family_data in knowledge_tree.items():
            if family_name == target_family:
                return family_data, knowledge_tree, family_name
            if (isinstance(family_data, dict) 
                and "knowledge" in family_data 
                and isinstance(family_data["knowledge"], dict)):
                found, parent, parent_key = self.find_family_by_name(family_data["knowledge"], target_family)
                if found is not None:
                    return found, parent, parent_key
        return None, None, None

    def gather_leaf_families(self, knowledge_tree):
        """
        Returns a list of tuples (fam_name, fam_data, parent_dict, fam_key)
        representing knowledge families that have no children under 'knowledge'.
        """
        results = []
        def dfs_recursive(current_dict):
            if not isinstance(current_dict, dict):
                return
            for fam_name, fam_data in current_dict.items():
                if isinstance(fam_data, dict):
                    children = fam_data.get("knowledge", {})
                    if not children:
                        results.append((fam_name, fam_data, current_dict, fam_name))
                    else:
                        dfs_recursive(children)
        dfs_recursive(knowledge_tree)
        return results

    def apply_knowledge_expansion(self, parent_dict, expansion_data):
        """
        Merges expansion_data into the 'knowledge' sub-dictionary of the parent_dict.
        If 'knowledge' key doesn't exist, it is created.
        """
        if "knowledge" not in parent_dict:
            parent_dict["knowledge"] = {}
        for new_area_name, new_area_info in expansion_data.items():
            parent_dict["knowledge"][new_area_name] = new_area_info

    def apply_reduction(self, subtree, reduction_instructions):
        """
        Applies the LLM's reduction instructions to the subtree in place:
          - Removing sub-knowledge items or merging them.
        The 'reduction_instructions' is expected to be a dict like:
          {
            "TooNarrowArea": "remove",
            "RedundantArea": { "merge_into": "AnotherArea" }
          }
        """
        knowledge_dict = subtree.get("knowledge", {})
        if not isinstance(knowledge_dict, dict):
            return
        merges = []
        removals = []
        for area_name, instruction in reduction_instructions.items():
            if instruction == "remove":
                removals.append(area_name)
            elif isinstance(instruction, dict) and "merge_into" in instruction:
                merges.append((area_name, instruction["merge_into"]))
        for r in removals:
            if r in knowledge_dict:
                del knowledge_dict[r]
        for old_area, target_area in merges:
            if old_area not in knowledge_dict or target_area not in knowledge_dict:
                continue
            old_data = knowledge_dict[old_area]
            target_data = knowledge_dict[target_area]
            if "knowledge" not in target_data or not isinstance(target_data["knowledge"], dict):
                target_data["knowledge"] = {}
            for sub_sub_area, sub_sub_data in old_data.get("knowledge", {}).items():
                target_data["knowledge"][sub_sub_area] = sub_sub_data
            del knowledge_dict[old_area]
        subtree["knowledge"] = knowledge_dict

    def expand_single_knowledge_area(self, 
                                     knowledge_tree, 
                                     area_name, 
                                     target_level=1, 
                                     sub_prompt="", 
                                     max_sub_knowledge=10, 
                                     debug_level=0):
        """
        Expands the specified knowledge area at the exact 'target_level'. 
        If sub-knowledge already exists, merges newly generated items.

        Parameters:
          - knowledge_tree (dict): The entire knowledge taxonomy.
          - area_name (str): The name of the knowledge family to expand.
          - target_level (int): The level in the tree at which this family must appear.
          - sub_prompt (str): Additional instructions to include in the final LLM prompt.
          - max_sub_knowledge (int): Max number of sub-items to add.
          - debug_level (int): Debug verbosity (0 = silent, 1 = basic, 2 = detailed).

        Returns:
          - A new copy of the updated knowledge_tree with expansions applied.
            If the family is not found at that level or expansions fail, returns the copy unmodified.
        """
        tree_copy = copy.deepcopy(knowledge_tree)
        if debug_level >= 2:
            print("\n--- expand_single_knowledge_area() ---")
            print(f"Looking for '{area_name}' at level={target_level} in the knowledge tree:\n", 
                  json.dumps(tree_copy, indent=2))
        subtree, parent, key = self.find_family_by_name_and_level(
            tree_copy, 
            area_name, 
            target_level=target_level
        )
        if subtree is None:
            if debug_level >= 1:
                print(f"ERROR: Knowledge family '{area_name}' not found at level {target_level}.")
            return tree_copy
        snapshot = {area_name: subtree}
        prompt = self.create_subknowledge_prompt(
            area_name, 
            snapshot, 
            sub_prompt=sub_prompt, 
            max_sub_knowledge=max_sub_knowledge
        )
        if debug_level >= 2:
            print(f"\nPrompt for '{area_name}' at level={target_level}:\n", prompt)
        response = self.get_bedrock_response(prompt, debug_level=debug_level)
        if response is None:
            if debug_level >= 1:
                print(f"ERROR: LLM expansion returned None for '{area_name}'.")
            return tree_copy
        try:
            expansion_data = json.loads(response)
            if debug_level >= 2:
                print(f"\nParsed expansion data for '{area_name}' at level={target_level}:\n{json.dumps(expansion_data, indent=2)}")
        except Exception as e:
            if debug_level >= 1:
                print(f"ERROR: Could not parse JSON from LLM for '{area_name}'. Raw response:\n{response}")
            return tree_copy
        self.apply_knowledge_expansion(subtree, expansion_data)
        if debug_level >= 2:
            print(f"\nUpdated sub-tree for '{area_name}' at level={target_level}:\n", json.dumps(subtree, indent=2))
        return tree_copy

    def expand_all_families(self, knowledge_tree, max_sub_knowledge=10, debug_level=0):
        """
        Expands all leaf families in the knowledge_tree by generating new sub-knowledge items 
        for each leaf. Returns an updated copy and the original tree.
        """
        tree_copy = copy.deepcopy(knowledge_tree)
        if debug_level >= 2:
            print("\n--- expand_all_families() ---")
            print("Initial Knowledge Tree:\n", json.dumps(knowledge_tree, indent=2))
        leaf_fams = self.gather_leaf_families(tree_copy)
        for fam_name, fam_data, parent, key in leaf_fams:
            snapshot = {fam_name: fam_data}
            if debug_level >= 2:
                print(f"\nExpanding leaf family '{fam_name}' ...")
                print("Sub-tree snapshot:\n", json.dumps(snapshot, indent=2))
            prompt = self.create_subknowledge_prompt(fam_name, snapshot, max_sub_knowledge=max_sub_knowledge)
            if debug_level >= 2:
                print("\nPrompt:\n", prompt)
            response = self.get_bedrock_response(prompt, debug_level=debug_level)
            if response is None:
                if debug_level >= 1:
                    print(f"LLM expansion failed for '{fam_name}'.")
                continue
            try:
                expansion_data = json.loads(response)
                if debug_level >= 2:
                    print(f"Parsed expansion data for '{fam_name}':\n", json.dumps(expansion_data, indent=2))
            except Exception as e:
                if debug_level >= 1:
                    print(f"Could not parse JSON for '{fam_name}'. Raw:\n{response}")
                continue
            self.apply_knowledge_expansion(fam_data, expansion_data)
            if debug_level >= 2:
                print(f"Updated sub-tree for '{fam_name}':\n", json.dumps(fam_data, indent=2))
        if debug_level >= 2:
            print("\n--- Final Updated Tree ---")
            print(json.dumps(tree_copy, indent=2))
        return tree_copy, knowledge_tree

    def compute_tree_stats(self, knowledge_tree):
        """
        Returns a dictionary with basic stats about the knowledge tree:
          {
            "total_families": (count of all nodes),
            "leaf_families": (count of nodes with no children),
            "max_depth": (maximum tree depth)
          }
        """
        stats = {"total_families": 0, "leaf_families": 0, "max_depth": 0}
        def dfs(tree, depth):
            if not isinstance(tree, dict):
                return
            for _, data in tree.items():
                if isinstance(data, dict):
                    stats["total_families"] += 1
                    stats["max_depth"] = max(stats["max_depth"], depth)
                    if not data.get("knowledge"):
                        stats["leaf_families"] += 1
                    else:
                        dfs(data["knowledge"], depth+1)
        dfs(knowledge_tree, 1)
        return stats

    def gather_all_families(self, knowledge_tree):
        """
        Returns a list of tuples (fam_name, fam_data, parent_dict, fam_key)
        for every knowledge family in the entire taxonomy.
        """
        results = []
        def dfs_recursive(current_dict):
            if not isinstance(current_dict, dict):
                return
            for fam_name, fam_data in current_dict.items():
                results.append((fam_name, fam_data, current_dict, fam_name))
                if isinstance(fam_data, dict) and "knowledge" in fam_data:
                    dfs_recursive(fam_data["knowledge"])
        dfs_recursive(knowledge_tree)
        return results

    def chunking_plan_prompt(self, tree_stats, all_families):
        """
        Creates a prompt asking the LLM to produce a chunking strategy for a large knowledge taxonomy.
        Output must be JSON with a "chunks" key containing an array of arrays (chunk lists).
        """
        prompt = f"""
        You are an AI assistant that helps plan how to split a large knowledge taxonomy tree into manageable chunks. 
        Balance the size of trees between the chunks.
        Stats:
        - total_families: {tree_stats["total_families"]}
        - leaf_families: {tree_stats["leaf_families"]}
        - max_depth: {tree_stats["max_depth"]}
        Family names:
        {json.dumps(all_families, indent=2)}
        Return a JSON with a key "chunks" whose value is an array of arrays of family names.
        Return ONLY the JSON.
        """
        return prompt.strip()

    def extract_subtree_for_families(self, knowledge_tree, families_list):
        """
        Given a list of family names, extracts a partial tree containing just those families 
        (and any nested knowledge under them).
        """
        def dfs_extract(tree):
            out = {}
            if not isinstance(tree, dict):
                return out
            for fam, data in tree.items():
                if fam in families_list:
                    out[fam] = data
                else:
                    if isinstance(data, dict) and data.get("knowledge"):
                        deeper = dfs_extract(data["knowledge"])
                        if deeper:
                            out[fam] = {"description": data.get("description", ""), "knowledge": deeper}
            return out
        return dfs_extract(knowledge_tree)

    def merge_updated_subtree_into_tree(self, original_tree, updated_subtree):
        """
        Inserts updated_subtree's nodes into the original_tree in place, 
        overwriting any existing families with the same names.
        """
        def dfs_merge(orig, upd):
            for fam, data in upd.items():
                orig[fam] = data
            return orig
        return dfs_merge(original_tree, updated_subtree)

    def quality_check_subtree(self, subtree_dict, debug_level=0):
        """
        Sends the subtree to the LLM for a quality check. The LLM returns:
          {
            "updated_tree": { ... },
            "changes_description": "explanation of changes"
          }
        If there's an error or missing keys, returns None and an error message.
        """
        prompt = self.create_quality_check_prompt(subtree_dict)
        if debug_level >= 2:
            print("\n--- quality_check_subtree() Prompt ---\n", prompt)
        response = self.get_bedrock_response(prompt, debug_level=debug_level)
        if not response:
            err = "No response from LLM for quality_check_subtree."
            if debug_level >= 1:
                print(err)
            return None, err
        if debug_level >= 2:
            print("\n--- quality_check_subtree() LLM Response ---\n", response)
        try:
            data = json.loads(response)
        except Exception as e:
            err = f"Error parsing LLM response: {e}"
            if debug_level >= 1:
                print(err)
            return None, err
        if not isinstance(data, dict) or "updated_tree" not in data or "changes_description" not in data:
            err = "LLM response missing required keys."
            if debug_level >= 1:
                print(err)
            return None, err
        return data["updated_tree"], data["changes_description"]

    def quality_check_tree(self, knowledge_tree, enable_chunking=False, debug_level=0):
        """
        Runs a quality check on the entire knowledge_tree. If enable_chunking is True,
        the LLM is asked for a plan to chunk the tree. Otherwise, it is processed as a whole.

        Returns a tuple: (original_tree_copy, updated_tree, changes_description)
        """
        original_tree_copy = copy.deepcopy(knowledge_tree)
        if not enable_chunking:
            updated, changes = self.quality_check_subtree(original_tree_copy, debug_level=debug_level)
            if updated is None:
                return original_tree_copy, original_tree_copy, changes
            return original_tree_copy, updated, changes
        else:
            stats = self.compute_tree_stats(knowledge_tree)
            all_fams = self.gather_all_families(knowledge_tree)
            if debug_level >= 1:
                print("\n--- Tree Statistics ---")
                print(json.dumps(stats, indent=2))
                print("\n--- All Family Names ---")
                print(json.dumps(all_fams, indent=2))
            plan_prompt = self.chunking_plan_prompt(stats, all_fams)
            if debug_level >= 2:
                print("\n--- Chunking Plan Prompt ---\n", plan_prompt)
            plan_response = self.get_bedrock_response(plan_prompt, debug_level=debug_level)
            if debug_level >= 2:
                print("\n--- Chunking Plan Response ---\n", plan_response)
            if not plan_response:
                err = "LLM returned no chunking plan. Aborting chunking."
                if debug_level >= 1:
                    print(err)
                return original_tree_copy, original_tree_copy, err
            try:
                plan_data = json.loads(plan_response)
            except Exception as e:
                err = f"Could not parse chunking plan: {e}"
                if debug_level >= 1:
                    print(err)
                return original_tree_copy, original_tree_copy, err
            if "chunks" not in plan_data or not isinstance(plan_data["chunks"], list):
                err = "LLM chunking plan missing 'chunks' list."
                if debug_level >= 1:
                    print(err)
                return original_tree_copy, original_tree_copy, err
            chunks = plan_data["chunks"]
            if debug_level >= 1:
                print(f"\n--- Chunks from Plan ---\n{chunks}")
            combined_changes = []
            working_tree = copy.deepcopy(original_tree_copy)
            def process_chunk(chunk_list, chunk_id):
                subtree = self.extract_subtree_for_families(working_tree, chunk_list)
                chunk_stats = self.compute_tree_stats(subtree)
                print(f"\n--- Stats for Chunk #{chunk_id} ({chunk_list}) ---")
                print(json.dumps(chunk_stats, indent=2))
                updated_sub, desc = self.quality_check_subtree(subtree, debug_level=debug_level)
                return (chunk_id, updated_sub, desc)
            futures = []
            with ThreadPoolExecutor() as executor:
                for i, chunk_list in enumerate(chunks, start=1):
                    futures.append(executor.submit(process_chunk, chunk_list, i))
                for future in futures:
                    chunk_id, updated_sub, desc = future.result()
                    if updated_sub is None:
                        combined_changes.append(f"Chunk {chunk_id} error: {desc}")
                    else:
                        working_tree = self.merge_updated_subtree_into_tree(working_tree, updated_sub)
                        combined_changes.append(f"Chunk {chunk_id} changes:\n{desc}")
            final_changes = "\n\n".join(combined_changes)
            return original_tree_copy, working_tree, final_changes

    def build_knowledge_tree(self, initial_tree, current_depth, last_level, max_sub_knowledge, debug_level=0):
        """
        Iteratively expands the knowledge tree from the given current_depth up to last_level.
        After each expansion iteration, a quality check is performed. The final tree is saved to 
        'final_knowledge_tree.json' when done, and also returned.
        """
        tree = copy.deepcopy(initial_tree)
        total_iterations = last_level - 1
        print(f"\nStarting iterative build: 0 / {total_iterations} iterations complete. (Current depth: 1)")
        if debug_level >= 2:
            print("\nInitial tree:")
            print(json.dumps(tree, indent=2))
        iteration = 0
        while current_depth < last_level:
            iteration += 1
            if debug_level >= 2:
                print(f"\n--- Iteration {iteration} (Expanding at depth {current_depth}) ---")
            else:
                print(f"Iteration {iteration} / {total_iterations} (Current depth: {current_depth})")
            updated_tree, _ = self.expand_all_families(tree, max_sub_knowledge, debug_level=debug_level)
            tree = updated_tree
            current_depth += 1
            if debug_level >= 2:
                stats = self.compute_tree_stats(tree)
                print(f"\nAfter expansion to depth {current_depth}, tree statistics:")
                print(json.dumps(stats, indent=2))
            else:
                print(f"Iteration {iteration} complete. New depth: {current_depth}.")
            qc_orig, qc_updated, qc_changes = self.quality_check_tree(tree, enable_chunking=False, debug_level=debug_level)
            if debug_level >= 2:
                print(f"\nQuality control check after depth {current_depth}:")
                print("QC Changes Description:")
                print(qc_changes)
            tree = qc_updated
        with open("final_knowledge_tree.json", "w") as f:
            json.dump(tree, f, indent=2)
        print(f"\nFinal tree saved to 'final_knowledge_tree.json'.")
        return tree

    def print_hierarchy_tree(self, data, max_depth, current_level=1, prefix=""):
        """
        Recursively prints the hierarchy of knowledge areas up to 'max_depth' levels.
        """
        if current_level > max_depth:
            return
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└──" if is_last else "├──"
            print(f"{prefix}{connector} {key}")
            next_items = value.get("knowledge")
            if isinstance(next_items, dict):
                new_prefix = prefix + ("    " if is_last else "│   ")
                self.print_hierarchy_tree(next_items, max_depth, current_level + 1, new_prefix)

    def reduce_single_knowledge_area(self, knowledge_tree, area_name, debug_level=0):
        """
        Reduces the number of knowledge items under a given family by calling the LLM
        to identify which sub-knowledge areas are unfit or excessively granular.
        The function then removes or merges these knowledge items based on the LLM's recommendation.

        Parameters:
            knowledge_tree (dict): The current knowledge taxonomy tree.
            area_name (str): The name of the knowledge family to reduce.
            debug_level (int): Debug verbosity (0 = silent, 1 = basic info, 2 = detailed).

        Returns:
            (tree_copy, original_tree, updated_subtree_or_message):
                A tuple containing:
                    - The updated copy of the knowledge tree,
                    - The original (unmodified) knowledge tree,
                    - A dictionary mapping {area_name: subtree} or an error message string.
        """
        tree_copy = copy.deepcopy(knowledge_tree)
        if debug_level >= 2:
            print("\n--- reduce_single_knowledge_area() ---")
            print(f"Looking for '{area_name}' in the knowledge tree:\n", json.dumps(knowledge_tree, indent=2))
        subtree, parent, key = self.find_family_by_name(knowledge_tree, area_name)
        if subtree is None:
            msg = f"ERROR: Knowledge family '{area_name}' not found in the tree."
            if debug_level >= 1:
                print(msg)
            return tree_copy, knowledge_tree, msg
        snapshot = {area_name: subtree}
        prompt = self.create_reduction_prompt(area_name, snapshot)
        if debug_level >= 2:
            print(f"\nPrompt for reducing '{area_name}':\n", prompt)
        response = self.get_bedrock_response(prompt, debug_level=debug_level)
        if response is None:
            msg = f"ERROR: LLM reduction returned None for '{area_name}'."
            if debug_level >= 1:
                print(msg)
            return tree_copy, knowledge_tree, msg
        try:
            reduction_data = json.loads(response)
            if debug_level >= 2:
                print(f"\nParsed reduction data for '{area_name}':\n", json.dumps(reduction_data, indent=2))
        except Exception as e:
            msg = f"ERROR: Could not parse JSON from LLM for '{area_name}'. Raw response:\n{response}"
            if debug_level >= 1:
                print(msg)
            return tree_copy, knowledge_tree, msg
        self.apply_reduction(subtree, reduction_data)
        if debug_level >= 2:
            print(f"\nUpdated sub-tree for '{area_name}' after reduction:\n", json.dumps(subtree, indent=2))
        return tree_copy, knowledge_tree, {area_name: subtree}
    
    def remove_all_subknowledge(self, knowledge_tree, family_name, debug_level=0):
        """
        Removes all sub-knowledge under the specified 'family_name'.
        Keeps the 'description' intact but empties the 'knowledge' dictionary.

        Parameters:
            knowledge_tree (dict): The full taxonomy tree (keys = family names, values = dicts).
            family_name (str): The name of the family whose subknowledge should be cleared.
            debug_level (int): Debug verbosity (0 = silent, 1 = basic info, 2 = detailed).

        Returns:
            dict: A copy of the updated knowledge tree where 'family_name' has an empty 'knowledge' dict.
        """
        tree_copy = copy.deepcopy(knowledge_tree)
        if debug_level >= 2:
            print("\n--- remove_all_subknowledge() ---")
            print(f"Attempting to remove all sub-knowledge for '{family_name}'")
        subtree, parent, key = self.find_family_by_name(tree_copy, family_name)
        if subtree is None:
            if debug_level >= 1:
                print(f"Family '{family_name}' not found. No changes made.")
            return tree_copy
        if debug_level >= 2:
            print(f"Before removal, subtree for '{family_name}':", json.dumps(subtree, indent=2))
        subtree["knowledge"] = {}
        if debug_level >= 2:
            print(f"After removal, subtree for '{family_name}':", json.dumps(subtree, indent=2))
        return tree_copy
