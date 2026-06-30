from src.inference.ranking import ParetoRanker


class TestParetoRanker:
    def test_rank_empty(self):
        ranker = ParetoRanker()
        assert ranker.rank([]) == []

    def test_rank_single(self):
        ranker = ParetoRanker()
        candidates = [
            {
                'ionic_conductivity': {'value': 1e-3},
                'energy_above_hull': {'value': 0.01},
                'ood': {'ood_score': 0.5}
            }
        ]
        ranked = ranker.rank(candidates)
        assert len(ranked) == 1
        assert ranked[0]['rank'] == 1

    def test_pareto_dominance(self):
        ranker = ParetoRanker()
        candidates = [
            {
                'ionic_conductivity': {'value': 1e-3},
                'energy_above_hull': {'value': 0.01},
                'ood': {'ood_score': 0.9}
            },
            {
                'ionic_conductivity': {'value': 1e-6},
                'energy_above_hull': {'value': 0.1},
                'ood': {'ood_score': -1.0}
            }
        ]
        ranked = ranker.rank(candidates)
        assert len(ranked) == 2
        assert ranked[0]['rank'] == 1
        assert ranked[0]['pareto_rank'] == 1
