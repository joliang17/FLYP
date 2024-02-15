import os
import torch
import torchvision
import wilds
import pdb

from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from wilds.common.data_loaders import get_train_loader, get_eval_loader

class OxfordPet:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        # pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
        # pets_train_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", download=True, transform=preprocess)
        self.train_loader = None

        print("Loading Test Data from OxfordPets Test")
        pets_path_test = os.path.join(location, 'OxfordPets', self.test_subset)
        os.makedirs(pets_path_test, exist_ok=True)
        self.test_dataset = torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", download=True, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.classnames = self.test_dataset.classes
        self.class_cat = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'English Cocker Spaniel', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx', ]
        self.class_dog = [item for item in self.classnames if item not in self.class_cat]

        self.index_cat = [i for i, name in enumerate(self.classnames) if name in self.class_cat]
        self.index_dog = [i for i, name in enumerate(self.classnames) if name in self.class_dog]

class OxfordPetVal(OxfordPet):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        # self.test_subset = 'val'
        super().__init__(*args, **kwargs)

class OxfordPetTest(OxfordPet):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)