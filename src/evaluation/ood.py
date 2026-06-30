from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OODDetector:
    def __init__(self, contamination=0.05):
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = None

    def fit(self, training_embeddings):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(training_embeddings)
        self.detector.fit(X)

    def score(self, embedding):
        X = self.scaler.transform(embedding.reshape(1, -1))
        score = self.detector.score_samples(X)[0]
        is_ood = self.detector.predict(X)[0] == -1
        return {
            "ood_score": float(score),
            "is_ood": bool(is_ood),
            "confidence": "LOW" if is_ood else "HIGH",
        }
