import torch
from torch.utils.data import DataLoader, Dataset
from seedvox.utils.g2p_factory import get_phoneme_generator

class SimpleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return list(batch)

class MockCollate:
    def __init__(self, g2p):
        self.g2p = g2p
    def __call__(self, batch):
        # This mirrors ExplicitCollate behavior
        results = self.g2p(batch)
        print(f"DEBUG: Input batch len: {len(batch)}")
        print(f"DEBUG: G2P output len: {len(results)}")
        return results

if __name__ == "__main__":
    texts = ["Hello world", "This is a test", "Another sentence"]
    dataset = SimpleDataset(texts)
    g2p = get_phoneme_generator('g2p_en')
    
    # Test with workers
    loader = DataLoader(dataset, batch_size=3, collate_fn=MockCollate(g2p), num_workers=2)
    for batch in loader:
        print(f"DEBUG: Loader batch output len: {len(batch)}")
        # Check first result
        print(f"DEBUG: First result len: {len(batch[0])}")
