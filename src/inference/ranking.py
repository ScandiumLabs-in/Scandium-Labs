import numpy as np


class ParetoRanker:
    def __init__(self):
        self.objective_names = [
            "ionic_conductivity",
            "neg_energy_above_hull",
            "confidence",
        ]

    def rank(self, candidates):
        if not candidates:
            return []

        obj_matrix = np.array(
            [
                [
                    np.log10(max(c.get("ionic_conductivity", {}).get("value", 1e-10), 1e-10)),
                    -c.get("energy_above_hull", {}).get("value", 1.0),
                    c.get("ood", {}).get("ood_score", -1),
                ]
                for c in candidates
            ]
        )

        pareto_ranks = self._pareto_rank(obj_matrix)
        weights = np.array([0.5, 0.3, 0.2])
        normalized = self._normalize(obj_matrix)
        weighted_scores = normalized @ weights

        final_scores = -pareto_ranks * 1000 + weighted_scores
        sorted_indices = np.argsort(-final_scores)

        ranked_candidates = []
        for rank_idx, orig_idx in enumerate(sorted_indices):
            candidate = dict(candidates[orig_idx])
            candidate["rank"] = rank_idx + 1
            candidate["pareto_rank"] = int(pareto_ranks[orig_idx])
            candidate["composite_score"] = float(weighted_scores[orig_idx])
            ranked_candidates.append(candidate)

        return ranked_candidates

    def _pareto_rank(self, obj_matrix):
        n = len(obj_matrix)
        ranks = np.zeros(n, dtype=int)
        remaining = set(range(n))
        current_rank = 1

        while remaining:
            non_dominated = []
            remaining_list = list(remaining)

            for i in remaining_list:
                dominated = False
                for j in remaining_list:
                    if i != j and self._dominates(obj_matrix[j], obj_matrix[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(i)

            for i in non_dominated:
                ranks[i] = current_rank
                remaining.remove(i)

            current_rank += 1

        return ranks

    def _dominates(self, a, b):
        return np.all(a >= b) and np.any(a > b)

    def _normalize(self, matrix):
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        denom = max_vals - min_vals + 1e-8
        return (matrix - min_vals) / denom
