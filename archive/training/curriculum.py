from torch.utils.data import DataLoader, Subset


class CurriculumDataLoader:
    def __init__(self, dataset, n_epochs):
        self.dataset = dataset
        self.n_epochs = n_epochs

        self.complexity_scores = [len(s.composition.elements) * len(s) for s in dataset.structures]
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: self.complexity_scores[i])

    def get_loader(self, epoch, batch_size):
        fraction = min(1.0, 0.3 + 0.7 * epoch / self.n_epochs)
        n = int(len(self.sorted_indices) * fraction)
        subset_indices = self.sorted_indices[:n]
        subset = Subset(self.dataset, subset_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)
