# query_processing/suggestion_system.py
import json
import os
from collections import defaultdict, Counter
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity

class QuerySuggester:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.query_file = f"data/queries_{dataset_name}.json"
        self.user_queries_file = f"data/user_queries_{dataset_name}.json"
        self.initialize_files()
        self.load_queries()
        self.vectorizer = TfidfVectorizer()
        self.train_tfidf()
        
    def initialize_files(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.query_file):
            with open(self.query_file, "w") as f:
                json.dump({"original_queries": [], "user_queries": {}}, f)
        if not os.path.exists(self.user_queries_file):
            with open(self.user_queries_file, "w") as f:
                json.dump([], f)
    
    def load_queries(self):
        with open(self.query_file) as f:
            data = json.load(f)
            self.original_queries = data.get("original_queries", [])
            self.user_queries = data.get("user_queries", {})
        
        with open(self.user_queries_file) as f:
            self.all_user_queries = json.load(f)
    
    def train_tfidf(self):
        all_queries = self.original_queries + list(self.user_queries.keys())
        if all_queries:
            self.tfidf_matrix = self.vectorizer.fit_transform(all_queries)
    
    def log_query(self, query: str):
        self.user_queries[query] = self.user_queries.get(query, 0) + 1
        
        with open(self.query_file, "w") as f:
            json.dump({
                "original_queries": self.original_queries,
                "user_queries": self.user_queries
            }, f)
        
        self.all_user_queries.append(query)
        with open(self.user_queries_file, "w") as f:
            json.dump(self.all_user_queries, f)
        
        self.train_tfidf()
    
    def get_suggestions(self, partial_query: str, n: int = 5) -> List[str]:
        suggestions = []
        
        # 1. Autocomplete from popular queries
        popular = sorted(self.user_queries.items(), key=lambda x: x[1], reverse=True)[:n]
        suggestions.extend([q for q, _ in popular if q.lower().startswith(partial_query.lower())])
        
        # 2. Similar queries (cosine similarity)
        if hasattr(self, 'tfidf_matrix') and partial_query:
            query_vec = self.vectorizer.transform([partial_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            similar_indices = similarities.argsort()[-n:][::-1]
            all_queries = self.original_queries + list(self.user_queries.keys())
            suggestions.extend([all_queries[i] for i in similar_indices if similarities[i] > 0.3])
        
        # 3. Spelling corrections
        if len(partial_query.split()) == 1:
            spelling_suggestions = get_close_matches(
                partial_query, 
                list(self.user_queries.keys()) + self.original_queries,
                n=2,
                cutoff=0.7
            )
            suggestions.extend(spelling_suggestions)
        
        # Remove duplicates and return
        return list(dict.fromkeys(suggestions))[:n]