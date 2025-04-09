import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fvcore.nn.precise_bn import update_bn_stats

import csv
import os

# from model import SeqNN
# from model_v2_compatible import SeqNN
from model_v2_horiz_checkpoint import SeqNN

import schedulefree


class HiCDataset(Dataset):
    def __init__(self, data_files):
        self.data = []  # To store all sequences and HiC vectors
        
        # Load and process the data files
        for file in data_files:
            file_data = torch.load(file, weights_only=True)
            
            for data in file_data:
                ohe_sequence, hic_vector = data

                # Process the OHE sequence
                ohe_sequence = ohe_sequence.squeeze(0)  # Remove singleton dimension
                
                # Ensure the sequence has the correct shape
                assert ohe_sequence.shape[0] == 4, f"Expected 4 channels, but got {ohe_sequence.shape[0]}"
                assert len(ohe_sequence.shape) == 2, f"Expected 2D shape for sequence, but got {ohe_sequence.shape}"
                
                # Add processed pair to the data list
                self.data.append((ohe_sequence, hic_vector))
        
        print(f"Total sequences loaded: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the preprocessed (ohe_sequence, hic_vector) pair from memory
        ohe_sequence, hic_vector = self.data[idx]
        return ohe_sequence, hic_vector


def train(args, model, device, train_loader, val_loader, optimizer, epoch, best_val_loss, epochs_no_improve, weight_clip_value, save_model=False, save_model_path=None, scaler=None):
    model.train()
    optimizer.train()
    
    train_loss = 0  # Initialize training loss
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
            
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = F.mse_loss(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        # Apply weight clipping
        for param in model.parameters():
            param.data.clamp_(-weight_clip_value, weight_clip_value)
        
        train_loss += loss.item()  # SUM of batch losses
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break  

    # Normalize to get the mean loss
    train_loss /= num_batches    
    
    # # Apply Precise BatchNorm Updates
    print("Updating BatchNorm statistics with preciseBP...")
    # # update_bn_stats(model, (data.to(device) for data, _ in train_loader), num_iters=len(train_loader))
    # update_bn_stats(model, (data.to(device) for data, _ in train_loader), num_iters=200)
    # with multiple GPUs
    update_bn_stats(model.module, (data.to(device) for data, _ in train_loader), num_iters=200)
    
    model.eval()
    val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.mse_loss(output, target).item()
            num_batches += 1

    val_loss /= num_batches  # Normalize by number of batches
    print(f'\nValidation set: Average MSE loss: {val_loss:.4f}\n')

    # Early Stopping Logic
    if val_loss < best_val_loss:
        print(f"Validation loss improved: {best_val_loss:.6f} -> {val_loss:.6f}")
        best_val_loss = val_loss
        epochs_no_improve = 0
        
        if save_model:
            # saving each model separately
            # Save the model with epoch number in the filename
            # model_filename = f"{save_model_path}/model_epoch_{epoch}.pth"
            # torch.save(model.state_dict(), model_filename)
            # print(f"Model saved to {model_filename}")
            
            # overwritting the model
            torch.save(model.state_dict(), save_model_path)
        
    else:
        epochs_no_improve += 1
    
    return train_loss, val_loss, best_val_loss, epochs_no_improve  # Return training and validation losses        


def test(model, optimizer, device, test_loader):
    model.eval()
    optimizer.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()  # Sum up batch loss

    test_loss /= len(test_loader)
    print('\nTest set: Average MSE loss: {:.4f}\n'.format(test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Toy Akita Example')
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to dataset directory containing .pt files.")
    parser.add_argument("--test_fold", type=str, required=True, 
                        help="Fold to use as test data (e.g., 'fold0').")
    parser.add_argument("--val_fold", type=str, required=True, 
                        help="Fold to use as validation data (e.g., 'fold1').")
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: 0.002)')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"],
                        help='Choose optimizer: "adam" (default) or "sgd"')
    parser.add_argument('--momentum', type=float, default=0.98,
                        help='Momentum value for SGD optimizer (default: 0.99575). Ignored if using Adam.')
    parser.add_argument('--l2-scale', type=float, default=1.5e-5,
                        help='L2 regularization weight (default: 1.5e-5).')
    parser.add_argument('--weight-clipping', type=float, default=20.0,
                        help='')
    parser.add_argument('--early-stop-patience', type=int, default=8,
                    help='Stop training if validation loss does not improve for this many epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-path', type=str, default="./",
                        help='Path for Saving the current Model')
    parser.add_argument('--save-losses', type=str, default="./losses.csv",
                    help='Path for Saving train and valid losses')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # Gather data files
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pt")]
    test_files = [f for f in all_files if args.test_fold in f]
    val_files = [f for f in all_files if args.val_fold in f]
    train_files = [f for f in all_files if args.test_fold not in f and args.val_fold not in f]
    
    # for quick testing
    # train_files = [f"/scratch1/smaruj/train_pytorch_akita/mouse_data/Hsieh2019_mESC_data_local/fold2_{i}.pt" for i in range(4)]
    
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}, Test files: {len(test_files)}")
    print("Loading data")
    
    train_dataset = HiCDataset(train_files)
    val_dataset = HiCDataset(val_files)
    # test_dataset = HiCDataset(test_files)
    
    # differece: Akita.v2 has a parameter shuffle_buffer=128
    # Here, the entire train set is shuffled randomly at the beginning of each epoch, so it's more randomized than using shuffle_buffer
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("len train_loader", len(train_loader))
    print("len valid_loader", len(valid_loader))
    
    # model = SeqNN()
    # starting from loaded TF-weights
    # parameters only
    # model = torch.load("/home1/smaruj/pytorch_akita/model_v2_mouse_model0_target0.pth").to(device)
    # full model
    # model.load_state_dict(torch.load("/scratch1/smaruj/train_pytorch_akita/mouse_models/model_0_v2_finetuned.pt", map_location=device))
    
    # multiple GPUs
    model = SeqNN()
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    print(f"Selected optimizer: {args.optimizer}")
    # Select optimizer
    if args.optimizer == "adam":
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.l2_scale)
    elif args.optimizer == "sgd":
        optimizer = schedulefree.SGDScheduleFree(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_scale)
    
    # scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Open the file in append mode and write header only if it doesn't exist
    file_exists = os.path.isfile(args.save_losses)
    with open(args.save_losses, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        
        for epoch in range(1, args.epochs + 1):
            train_loss, val_loss, best_val_loss, epochs_no_improve = train(
                args, model, device, train_loader, valid_loader,
                optimizer, epoch, best_val_loss, epochs_no_improve,
                weight_clip_value=args.weight_clipping, 
                save_model=args.save_model, save_model_path=args.save_model_path,
                scaler=scaler
            )

            # Append training and validation losses for the current epoch
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()  # Ensure the data is written to the file immediately
            
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                break
        
if __name__ == '__main__':
    main()

