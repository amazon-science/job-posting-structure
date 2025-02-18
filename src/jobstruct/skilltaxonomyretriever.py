from sentence_transformers import SentenceTransformer
import numpy as np

class SkillTaxonomyRetriever:
    def __init__(self, skill_tree, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the retriever with the given skill taxonomy and loads the embedding model.
        """
        self.skill_tree = skill_tree
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def compute_embedding(self, text):
        """
        Computes the embedding for the given text using the loaded SentenceTransformer model.
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Computes the cosine similarity between two vectors.
        """
        vec1, vec2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def compute_skill_embeddings(self, target_level, debug_level=1):
        """
        Computes embeddings for each skill family at the specified target level in the skill taxonomy.
        """
        embeddings = {}

        def dfs(node, current_depth, path):
            if current_depth == target_level:
                title = path[-1]
                description = node.get("description", "")
                full_path = " -> ".join(path)
                text = f"Path: {full_path}. Title: {title}. Description: {description}."
                if debug_level >= 2:
                    print(f"\nComputing embedding for node at level {target_level}:")
                    print(text)
                emb = self.compute_embedding(text)
                embeddings[title] = {"path": path.copy(), "embedding": emb}
            else:
                for child_name, child_data in node.get("skills", {}).items():
                    dfs(child_data, current_depth + 1, path + [child_name])

        for family_name, family_data in self.skill_tree.items():
            dfs(family_data, current_depth=1, path=[family_name])

        if debug_level >= 1:
            print(f"\nComputed embeddings for {len(embeddings)} nodes at level {target_level}.")
        return embeddings

    def find_similar_skill_families(self, embeddings_dict, query_text, top_n, debug_level=1):
        """
        Computes cosine similarity between the query_text's embedding and each stored embedding.
        Returns a list of tuples: (family_title, similarity_score, full_path) sorted by similarity.
        """
        query_embedding = self.compute_embedding(query_text)
        results = []
        for title, info in embeddings_dict.items():
            score = self.cosine_similarity(query_embedding, info["embedding"])
            results.append((title, score, info["path"]))
        results.sort(key=lambda x: x[1], reverse=True)
        if debug_level >= 1:
            print(f"Found {len(results)} candidates. Top {top_n} returned.")
        return results[:top_n]

    def extract_branch_for_target(self, target_family):
        """
        Given a target skill family name, returns a pruned tree that preserves the original structure but
        includes only the branch from the root down to the target family (including the full subtree under it).
        If the target_family is not found, returns an empty dict.
        """
        def helper(node, current_name):
            if current_name == target_family:
                return {current_name: node}
            pruned_children = {}
            for child_name, child_node in node.get("skills", {}).items():
                result = helper(child_node, child_name)
                if result:
                    # Merge result into pruned_children
                    pruned_children = self.merge_trees(pruned_children, result)
            if pruned_children:
                return {current_name: {"description": node.get("description", ""),
                                       "skills": pruned_children}}
            return None

        pruned_tree = {}
        for family_name, family_node in self.skill_tree.items():
            branch = helper(family_node, family_name)
            if branch:
                pruned_tree = self.merge_trees(pruned_tree, branch)
        return pruned_tree

    def extract_branches_for_targets(self, embeddings_dict, query_text, top_n, debug_level=1):
        """
        Given a query_text, this method:
          - Finds the top_n similar skill families using find_similar_skill_families.
          - For each candidate, extracts the branch (the full subtree from the root leading to that candidate)
            using extract_branch_for_target.
        Returns a dictionary that maps candidate family names to their merged pruned branches.
        """
        candidates = self.find_similar_skill_families(embeddings_dict, query_text, top_n=top_n, debug_level=debug_level)
        if not candidates:
            if debug_level >= 1:
                print("No similar skill families found.")
            return {}
        
        pruned_branches = {}
        for candidate in candidates:
            target_family = candidate[0]
            if debug_level >= 1:
                print(f"Extracting branch for candidate family: {target_family}")
            branch = self.extract_branch_for_target(target_family)
            if branch:
                pruned_branches = self.merge_trees(pruned_branches, branch)
        return pruned_branches
    
    @staticmethod
    def merge_trees(dest, src):
        """
        Recursively merges two branch trees. If a key exists in both trees,
        their 'skills' subtrees are merged.
        """
        for key, src_value in src.items():
            if key in dest:
                # If both have a "skills" field, merge them recursively.
                dest_skills = dest[key].get("skills", {})
                src_skills = src_value.get("skills", {})
                merged_skills = SkillTaxonomyRetriever.merge_trees(dest_skills, src_skills)
                dest[key]["skills"] = merged_skills
            else:
                dest[key] = src_value
        return dest

    @staticmethod
    def extract_leaf_skills(branches_tree):
        """
        Recursively extracts only the leaf skills (nodes with no further nested skills)
        from the given branches tree. The returned list includes only the skill names,
        omitting their descriptions.
        """
        leaves = []

        def recurse(tree):
            for skill_name, node in tree.items():
                if not node.get("skills"):
                    leaves.append(skill_name)
                else:
                    recurse(node.get("skills"))
        
        recurse(branches_tree)
        return leaves
