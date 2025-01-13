import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_base_dataset(data_root: str = './data/') -> tuple[datasets.ImageFolder, datasets.ImageFolder]:

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])

    # transform_test = transforms.ToTensor()

    trainset = datasets.ImageFolder(os.path.join(data_root, 'train'),transform=transform)
    testset = datasets.ImageFolder(os.path.join(data_root, 'test'),transform=transform)

    return trainset, testset
