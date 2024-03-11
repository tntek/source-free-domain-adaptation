from torch.utils.data import Dataset


class Dataset_Idx(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1], index

    def __len__(self):
        return len(self.dataset)
