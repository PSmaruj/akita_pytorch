import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fvcore.nn.precise_bn import update_bn_stats
from akita_model.model import SeqNN
import schedulefree
from data_processing.dataset import HiCDataset

# =====================
# Utility Functions
# =====================
def data_loader_for_precise_bn(loader, device):
    for data, _ in loader:
        yield data.to(device)


# =====================
# Training / Validation
# =====================
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
                
                # Create mask for valid (non-NaN) target entries
                valid_mask = ~torch.isnan(target)
                
                # Compute MSE only on valid entries
                loss = F.mse_loss(output[valid_mask], target[valid_mask])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            valid_mask = ~torch.isnan(target)
            loss = F.mse_loss(output[valid_mask], target[valid_mask])
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
    update_bn_stats(model, data_loader_for_precise_bn(train_loader, device), num_iters=min(len(train_loader), 200))
    
    model.eval()
    val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Mask NaN values
            valid_mask = ~torch.isnan(target)
            
            # If there are any valid values, compute MSE only on them
            if valid_mask.any():
                loss = F.mse_loss(output[valid_mask], target[valid_mask])
                val_loss += loss.item()
                num_batches += 1

    val_loss /= num_batches  # Normalize by number of batches
    print(f'\nValidation set: Average MSE loss: {val_loss:.4f}\n')

    # Early Stopping Logic
    if val_loss < best_val_loss:
        print(f"Validation loss improved: {best_val_loss:.6f} -> {val_loss:.6f}")
        best_val_loss = val_loss
        epochs_no_improve = 0
        
        if save_model:            
            # overwritting the model
            torch.save(model.state_dict(), save_model_path)
        
    else:
        epochs_no_improve += 1
    
    return train_loss, val_loss, best_val_loss, epochs_no_improve  # Return training and validation losses        


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser(description='Toy Akita Example')
    
    # --- Data ---
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to dataset directory containing .pt files.")
    parser.add_argument("--test_fold", type=str, required=True, 
                        help="Fold to use as test data (e.g., 'fold0').")
    parser.add_argument("--data_name", type=str, required=True, 
                        help="Data name used (e.g., 'Bonev2017_CN').")
    parser.add_argument("--val_fold", type=str, required=True, 
                        help="Fold to use as validation data (e.g., 'fold1').")
    parser.add_argument("--organism", type=str, required=True, choices=["mouse", "human"],
                        help="Organism name, used for selecting pretrained model.")
    parser.add_argument("--data-split", type=int, required=True,
                        help="Model split index (e.g., 0, 1, 2...).")
    
    # --- Training ---
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # --- Misc ---
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    
    # --- Setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
    
    # --- Load data ---
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pt")]
    test_files = [f for f in all_files if args.test_fold in f]
    val_files = [f for f in all_files if args.val_fold in f]
    train_files = [f for f in all_files if args.test_fold not in f and args.val_fold not in f]
    
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}, Test files: {len(test_files)}")
    print("Loading data")
    
    train_dataset = HiCDataset(train_files)
    val_dataset = HiCDataset(val_files)
    
    # differece: Akita.v2 has a parameter shuffle_buffer=128
    # Here, the entire train set is shuffled randomly at the beginning of each epoch, so it's more randomized than using shuffle_buffer
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("len train_loader", len(train_loader))
    print("len valid_loader", len(valid_loader))
    
    torch.serialization.add_safe_globals([SeqNN])
    
    # --- Model ---
    print(f"Loading pretrained model for {args.organism}, {args.data_name}, split {args.data_split}")
    model_path = f"/scratch1/smaruj/Akita_pytorch_models/tf_transferred/{args.organism}_models/" \
                 f"{args.data_name}/Akita_v2_{args.organism}_{args.data_name}_model{args.data_split}.pth"

    # randomly initialized top layer
    # model_path = f"/scratch1/smaruj/Akita_pytorch_models/tf_transferred/random_dense_layer/Akita_v2_random_dense_model{args.data_split}.pth"

    # model = torch.load(model_path, map_location=device)  # loads the entire SeqNN
    # model.to(device)
    
    model = torch.load(model_path, map_location=device, weights_only=False)  
    model.to(device) 
    model.train()

    # Select optimizer
    print(f"Selected optimizer: {args.optimizer}")
    if args.optimizer == "adam":
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.l2_scale)
    elif args.optimizer == "sgd":
        optimizer = schedulefree.SGDScheduleFree(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_scale)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    save_model_path = f"/scratch1/smaruj/Akita_pytorch_models/finetuned/{args.organism}_models/{args.data_name}/models/Akita_v2_{args.organism}_{args.data_name}_model{args.data_split}_finetuned.pth"
    save_losses = f"/scratch1/smaruj/Akita_pytorch_models/finetuned/{args.organism}_models/{args.data_name}/losses/Akita_v2_{args.organism}_{args.data_name}_model{args.data_split}_finetuned.csv"
    
    # --- Training Loop ---
    file_exists = os.path.isfile(save_losses)
    with open(save_losses, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        
        # --- Compute initial losses before training ---
        print("Computing initial losses before training...")
        model.eval()
        init_train_loss, init_val_loss = 0.0, 0.0
        num_train_batches, num_val_batches = 0, 0

        with torch.no_grad():
            # Compute initial train loss
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_mask = ~torch.isnan(target)
                if valid_mask.any():
                    loss = F.mse_loss(output[valid_mask], target[valid_mask])
                    init_train_loss += loss.item()
                    num_train_batches += 1

            init_train_loss /= max(num_train_batches, 1)

            # Compute initial validation loss
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_mask = ~torch.isnan(target)
                if valid_mask.any():
                    loss = F.mse_loss(output[valid_mask], target[valid_mask])
                    init_val_loss += loss.item()
                    num_val_batches += 1

            init_val_loss /= max(num_val_batches, 1)

        print(f"Initial Train Loss: {init_train_loss:.4f}, Initial Validation Loss: {init_val_loss:.4f}")
        writer.writerow([0, init_train_loss, init_val_loss])
        f.flush()
        
        for epoch in range(1, args.epochs + 1):
            train_loss, val_loss, best_val_loss, epochs_no_improve = train(
                args, model, device, train_loader, valid_loader,
                optimizer, epoch, best_val_loss, epochs_no_improve,
                weight_clip_value=args.weight_clipping, 
                save_model=args.save_model, save_model_path=save_model_path,
                scaler=None
            )

            # Append training and validation losses for the current epoch
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()  # Ensure the data is written to the file immediately
            
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                break
        
if __name__ == '__main__':
    main()

