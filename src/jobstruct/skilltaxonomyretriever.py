from sentence_transformers import SentenceTransformer
import numpy as np

class SkillTaxonomyRetriever:
    def __init__(self, skill_tree, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the retriever with the given skill taxonomy and loads the embedding model.

        Parameters:
          - skill_tree (dict): The complete skills taxonomy.
          - embedding_model_name (str): The name of the Hugging Face SentenceTransformer model to use.
        """
        self.skill_tree = skill_tree
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def compute_embedding(self, text):
        """
        Computes the embedding for the given text using the loaded SentenceTransformer model.
        
        Parameters:
          - text (str): The input text to encode.
        
        Returns:
          - embedding (list): The embedding vector as a list.
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

    def compute_skill_embeddings(self, target_level, debug_level=0):
        """
        Computes embeddings for each skill family at the specified target level in the skill taxonomy.
        
        For every node at the target level, this function builds a text string by concatenating:
          - The full path from the root to the node (separated by " -> "),
          - The node's title (i.e., its family name),
          - The node's description.
        
        The embedding is then computed by passing this text to compute_embedding(text).
        
        Parameters:
          - target_level (int): The hierarchy level (1-indexed) at which to compute embeddings.
          - debug_level (int): Debug verbosity level:
                0 = silent,
                1 = progress-only messages,
                2 = full details.
        
        Returns:
          - embeddings (dict): A dictionary mapping family names (at the target level) to a dict with:
                {
                  "path": [list of family names from the root to the node],
                  "embedding": [the computed embedding vector]
                }
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

    def find_similar_skill_families(self, embeddings_dict, query_text, top_n=5, debug_level=0):
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
            if not isinstance(node, dict):
                return {current_name: node} if current_name == target_family else None
            if current_name == target_family:
                return {current_name: node}
            pruned_children = {}
            for child_name, child_node in node.get("skills", {}).items():
                result = helper(child_node, child_name)
                if result:
                    pruned_children.update(result)
            if pruned_children:
                return {current_name: {"description": node.get("description", ""),
                                       "skills": pruned_children}}
            return None

        pruned_tree = {}
        for family_name, family_node in self.skill_tree.items():
            branch = helper(family_node, family_name)
            if branch:
                pruned_tree.update(branch)
        return pruned_tree

    def extract_branches_for_targets(self, embeddings_dict, query_text, top_n=2, debug_level=0):
        """
        Given a query_text, this method:
          - Finds the top_n similar skill families using find_similar_skill_families.
          - For each candidate, extracts the branch (the full subtree from the root leading to that candidate)
            using extract_branch_for_target.
        
        Returns a dictionary that maps candidate family names to their pruned branches.
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
                pruned_branches.update(branch)
        return pruned_branches

    


