from torch.utils.data import DataLoader
from utils import load_base_dataset


def create_data_loader(data_name: str,
                num_classes: int = 2,
                batch_size: int = 128,
                data_root: str = './data/') -> tuple[DataLoader, DataLoader]:

    if data_name == 'base':
        trainset, testset = load_base_dataset(data_root=data_root)

    # elif data_name == 'augmented':
    #     _, trainset = imagenet_dataset(data_root=data_root,
    #                                       num_classes=num_classes,
    #                                       seed=seed)
    #     testset = None
    else:
        raise ValueError('The given dataset config is not supported!')

    train_loader = DataLoader(trainset,
                                   batch_size=batch_size,
                                   num_workers=4,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   persistent_workers=False)

    test_loader: DataLoader = None
    if testset is not None:
        test_loader = DataLoader(testset,
                                      batch_size=batch_size * 2,
                                      num_workers=4,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=False)
    return train_loader, test_loader
