import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import AutoEncoder
from util import FontDataset

NUM_EPOCHS = 10
BATCH_SIZE = 64
RANDOM_SEED = 1337
DEVICE = "cuda:2"

def get_arguments():
    parser = argparse.ArgumentParser(description='Font model')
    parser.add_argument('data', type=str, help='The HDF5 font data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    return parser.parse_args()


# Set learning rate
# if i < 2:
#     lr = 1e-3
# elif i < 5:
#     lr = 1e-4
# elif i < 10:
#     lr = 1e-5
# elif i < 20:
#     lr = 1e-6
# else:
#     lr = 1e-7


def epoch(model, device, dataloader, criterion, optimizer, save_distrib=False):
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    train_loss = 0.0

    batch_bar = tqdm(desc="Train", total=len(dataloader), dynamic_ncols=True)
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        labels = batch[0].repeat_interleave(62).to(device)
        fonts = batch[1].reshape(-1, 1, 64, 64).to(device)
        
        with torch.cuda.amp.autocast():
            y_hat = model(fonts)
            loss = criterion(y_hat, fonts)
            train_loss += loss.item()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss is an average because we sum over the batches
        batch_bar.set_postfix(
            loss=f"{train_loss/(i+1):.7f}",
            lr=f"{optimizer.param_groups[0]['lr']:.7f}"
        )
        batch_bar.update()
        
        del y_hat, labels, fonts
    
    batch_bar.close()
    train_loss /= len(dataloader)
    
    return train_loss


def main():
    args = get_arguments()
    np.random.seed(args.random_seed)
    
    print("Loading dataset...")
    dataset = FontDataset(args.data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    print("Creating model...")
    model = AutoEncoder()
    model.to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    if os.path.isfile(args.model):
        print("Restoring weights from checkpoint...")
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for i in range(args.epochs):
        curr_lr = float(optimizer.param_groups[0]["lr"])
        loss = epoch(model, DEVICE, data_loader, criterion, optimizer, save_distrib=True)

        print(f"Epoch {i+1}/{args.epochs}\nTrain loss: {loss:.7f}\tlr: {curr_lr:.7f}")

        torch.save({
            'epoch': i+1,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"checkpoints/chkpt_epoch_{i+1}.pth")
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


