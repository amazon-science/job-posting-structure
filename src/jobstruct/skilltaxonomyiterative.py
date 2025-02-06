import json
import time
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
import copy

class SkillTaxonomy:
    def __init__(self, config=None):
        self.config = config

    def get_bedrock_response(self, content, max_tokens=1024*4,
                              model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                                anthropic_version="bedrock-2023-05-31", max_retries=100,
                                  backoff_factor=2, temperature=0, debug_level=0):
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

    def create_subskills_prompt(self, family_name, current_subtree, max_sub_skills=10):
        prompt = f"""
You are an AI assistant helping to expand a skill taxonomy tree.
We have a skill family named "{family_name}" with this JSON structure:

{json.dumps(current_subtree, indent=2)}

Please add up to {max_sub_skills} new sub-skills under "{family_name}".
If you cannot generate {max_sub_skills} quality and relevant sub-skills, provide only as many as appropriate.
Each sub-skill must have:
  - "description": a short string describing that sub-skill
  - "sub_skills": an empty dict or further nested structure if needed

Return ONLY the JSON that should be merged under this skill family's "sub_skills" key.
No extra explanation, no markdown.
        """
        return prompt.strip()

    def find_family_by_name(self, skill_tree, target_family):
        if not isinstance(skill_tree, dict):
            return None, None, None
        for family_name, family_data in skill_tree.items():
            if family_name == target_family:
                return family_data, skill_tree, family_name
            if isinstance(family_data, dict) and "sub_skills" in family_data and isinstance(family_data["sub_skills"], dict):
                found, parent, parent_key = self.find_family_by_name(family_data["sub_skills"], target_family)
                if found is not None:
                    return found, parent, parent_key
        return None, None, None

    def gather_leaf_families(self, skill_tree):
        results = []
        def dfs_recursive(current_dict):
            if not isinstance(current_dict, dict):
                return
            for fam_name, fam_data in current_dict.items():
                if isinstance(fam_data, dict):
                    sub_skills = fam_data.get("sub_skills", {})
                    if not sub_skills:
                        results.append((fam_name, fam_data, current_dict, fam_name))
                    else:
                        dfs_recursive(sub_skills)
        dfs_recursive(skill_tree)
        return results

    def apply_expansion(self, parent_dict, expansion_data):
        if "sub_skills" not in parent_dict:
            parent_dict["sub_skills"] = {}
        for new_family_name, new_family_info in expansion_data.items():
            parent_dict["sub_skills"][new_family_name] = new_family_info

    def expand_single_family(self, skill_tree, family_name, max_sub_skills=10, debug_level=0):
        tree_copy = copy.deepcopy(skill_tree)
        if debug_level >= 2:
            print("\n--- expand_single_family() ---")
            print(f"Looking for '{family_name}' in the skill tree:\n", json.dumps(skill_tree, indent=2))
        subtree, parent, key = self.find_family_by_name(skill_tree, family_name)
        if subtree is None:
            msg = f"ERROR: Skill family '{family_name}' not found in the tree."
            if debug_level >= 1:
                print(msg)
            return tree_copy, skill_tree, msg
        if "sub_skills" in subtree and subtree["sub_skills"]:
            msg = f"ERROR: Skill family '{family_name}' is not a leaf (it already has sub-skills)."
            if debug_level >= 1:
                print(msg)
            return tree_copy, skill_tree, msg
        snapshot = {family_name: subtree}
        prompt = self.create_subskills_prompt(family_name, snapshot, max_sub_skills=max_sub_skills)
        if debug_level >= 2:
            print(f"\nPrompt for '{family_name}':\n", prompt)
        response = self.get_bedrock_response(prompt, debug_level=debug_level)
        if response is None:
            msg = f"ERROR: LLM expansion returned None for '{family_name}'."
            if debug_level >= 1:
                print(msg)
            return tree_copy, skill_tree, msg
        try:
            expansion_data = json.loads(response)
            if debug_level >= 2:
                print(f"\nParsed expansion data for '{family_name}':\n", json.dumps(expansion_data, indent=2))
        except Exception as e:
            msg = f"ERROR: Could not parse JSON from LLM for '{family_name}'. Raw response:\n{response}"
            if debug_level >= 1:
                print(msg)
            return tree_copy, skill_tree, msg
        self.apply_expansion(subtree, expansion_data)
        if debug_level >= 2:
            print(f"\nUpdated sub-tree for '{family_name}':\n", json.dumps(subtree, indent=2))
        return tree_copy, skill_tree, {family_name: subtree}

    def expand_all_families(self, skill_tree, max_sub_skills=10, debug_level=0):
        tree_copy = copy.deepcopy(skill_tree)
        if debug_level >= 2:
            print("\n--- expand_all_families() ---")
            print("Initial Skill Tree:\n", json.dumps(skill_tree, indent=2))
        leaf_fams = self.gather_leaf_families(skill_tree)
        for fam_name, fam_data, parent, key in leaf_fams:
            snapshot = {fam_name: fam_data}
            if debug_level >= 2:
                print(f"\nExpanding leaf family '{fam_name}' ...")
                print("Sub-tree snapshot:\n", json.dumps(snapshot, indent=2))
            prompt = self.create_subskills_prompt(fam_name, snapshot, max_sub_skills=max_sub_skills)
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
            self.apply_expansion(fam_data, expansion_data)
            if debug_level >= 2:
                print(f"Updated sub-tree for '{fam_name}':\n", json.dumps(fam_data, indent=2))
        if debug_level >= 2:
            print("\n--- Final Updated Tree ---")
            print(json.dumps(skill_tree, indent=2))
        return tree_copy, skill_tree

    def compute_tree_stats(self, skill_tree):
        stats = {"total_families": 0, "leaf_families": 0, "max_depth": 0}
        def dfs(tree, depth):
            if not isinstance(tree, dict):
                return
            for _, data in tree.items():
                if isinstance(data, dict):
                    stats["total_families"] += 1
                    stats["max_depth"] = max(stats["max_depth"], depth)
                    if not data.get("sub_skills"):
                        stats["leaf_families"] += 1
                    else:
                        dfs(data["sub_skills"], depth+1)
        dfs(skill_tree, 1)
        return stats

    def gather_families_list(self, skill_tree):
        results = []
        def dfs(tree):
            if not isinstance(tree, dict):
                return
            for fam, data in tree.items():
                results.append(fam)
                if data.get("sub_skills"):
                    dfs(data["sub_skills"])
        dfs(skill_tree)
        return results

    def chunking_plan_prompt(self, tree_stats, all_families):
        prompt = f"""
You are an AI assistant that helps plan how to split a large skill taxonomy tree into manageable chunks.
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

    def extract_subtree_for_families(self, skill_tree, families_list):
        def dfs_extract(tree):
            out = {}
            if not isinstance(tree, dict):
                return out
            for fam, data in tree.items():
                if fam in families_list:
                    out[fam] = data
                else:
                    if isinstance(data, dict) and data.get("sub_skills"):
                        deeper = dfs_extract(data["sub_skills"])
                        if deeper:
                            out[fam] = {"description": data.get("description", ""), "sub_skills": deeper}
            return out
        return dfs_extract(skill_tree)

    def merge_updated_subtree_into_tree(self, original_tree, updated_subtree):
        def dfs_merge(orig, upd):
            for fam, data in upd.items():
                orig[fam] = data
            return orig
        return dfs_merge(original_tree, updated_subtree)

    def create_quality_check_prompt(self, tree):
        prompt = f"""
You are an expert in skill taxonomy quality control.
Below is a skill taxonomy (in JSON) with families structured as:
{{ "Family Name": {{ "description": "...", "sub_skills": {{ ... }} }} }}
Perform a quality check. In "updated_tree", return an improved version that preserves existing sub-skills,
adds additional ones where needed, and omits irrelevant or too narrow keys. In "changes_description",
explain all changes (both additions and omissions) and your rationale.
Return ONLY a valid JSON object with exactly these two keys.
The input tree is:
{json.dumps(tree, indent=2)}
        """
        return prompt.strip()

    def quality_check_subtree(self, subtree_dict, debug_level=0):
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

    def quality_check_tree(self, skill_tree, enable_chunking=False, debug_level=0):
        original_tree_copy = copy.deepcopy(skill_tree)
        if not enable_chunking:
            updated, changes = self.quality_check_subtree(original_tree_copy, debug_level=debug_level)
            if updated is None:
                return original_tree_copy, original_tree_copy, changes
            return original_tree_copy, updated, changes
        else:
            stats = self.compute_tree_stats(skill_tree)
            all_fams = self.gather_families_list(skill_tree)
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

    def build_skill_tree(self, initial_tree, last_level, debug_level=0):
        tree = copy.deepcopy(initial_tree)
        total_iterations = last_level - 1
        print(f"\nStarting iterative build: 0 / {total_iterations} iterations complete. (Current depth: 1)")
        if debug_level >= 2:
            print("\nInitial tree:")
            print(json.dumps(tree, indent=2))
        current_depth = 1
        iteration = 0
        while current_depth < last_level:
            iteration += 1
            if debug_level >= 2:
                print(f"\n--- Iteration {iteration} (Expanding at depth {current_depth}) ---")
            else:
                print(f"Iteration {iteration} / {total_iterations} (Current depth: {current_depth})")
            _, tree = self.expand_all_families(tree, max_sub_skills=3, debug_level=debug_level)
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
        with open("final_skill_tree.json", "w") as f:
            json.dump(tree, f, indent=2)
        print(f"\nFinal tree saved to 'final_skill_tree.json'.")
        return tree

    def print_hierarchy_tree(self, data, max_depth, current_level=1, prefix=""):
        if current_level > max_depth:
            return
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└──" if is_last else "├──"
            print(f"{prefix}{connector} {key}")
            next_items = None
            if "skills" in value:
                next_items = value["skills"]
            elif "sub_skills" in value:
                next_items = value["sub_skills"]
            if isinstance(next_items, dict):
                new_prefix = prefix + ("    " if is_last else "│   ")
                self.print_hierarchy_tree(next_items, max_depth, current_level + 1, new_prefix)
