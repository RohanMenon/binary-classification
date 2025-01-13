import os
import argparse
from data_loader import create_data_loader
import timm
import time
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(
        'Training a vision model for binary classification')
    parser.add_argument('--work_dir', default='checkpoints', type=str)
    parser.add_argument('--ckpt_prefix', default='baseline', type=str)
    parser.add_argument('--max_save', default=1, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    print("here")

    print(f'Use checkpoint prefix: {args.ckpt_prefix}')

    train_loader, test_loader = create_data_loader(
        data_name='base',
        batch_size=32)
    
    print(len(train_loader))
    print(len(test_loader))
    
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(192, 2)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                            betas=(0.9, 0.999), weight_decay=5e-5)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, 
                                                         eta_min=1e-5)

    print('Begin Training')
    t = time.time()

    for epoch in range(0, 10):
        model.train()
        train_correct = train_total = 0.
        for idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            bs = inputs.shape[0]
            y = model(inputs)
            loss = criterion(y, targets)
            loss.backward()

            optimizer.step()

            train_correct += y.argmax(1).eq(targets)[:bs].sum().item()
            train_total += bs

        scheduler.step(epoch)
    
        model.eval()
        val_correct = val_total = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                y_val = model(inputs)
                val_loss = criterion(y_val, targets)

                val_correct += y_val.argmax(1).eq(targets).sum().item()
                val_total += targets.size(0)
        
        acc_train = 100. * train_correct / train_total
        acc_val = 100. * val_correct / val_total

        used = time.time() - t
        t = time.time()

        string = (f'Epoch {epoch}: '
                f'Train acc {acc_train: .2f}%, '
                f'Val acc {acc_val: .2f}%, '
                f'Time:{used / 60: .2f} mins.')

        print(string)

        state = dict(backbone=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    start_epoch=epoch + 1)

        try: 
            path = f'{args.work_dir}/{args.ckpt_prefix}_{epoch}.pth'
            torch.save(state, path)
        except PermissionError:
            print('Error saving checkpoint!')
            pass
        if epoch >= args.max_save:
            path = f'{args.work_dir}/{args.ckpt_prefix}_{epoch - args.max_save}.pth'
            os.system('rm -f ' + (path))


if __name__ == '__main__':
    main()