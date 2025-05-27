import argparse
import torch
import optuna
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
from fvcore.nn.precise_bn import update_bn_stats

import csv
import os

# from model import SeqNN
from model_v2_compatible import SeqNN
# from model_v2_horiz_checkpoint import SeqNN

import schedulefree


def load_shuffled_sequences(shuffled_dir, num_needed):
    all_files = sorted([
        os.path.join(shuffled_dir, f)
        for f in os.listdir(shuffled_dir)
        if f.endswith(".pt")
    ])

    selected = []
    total = 0

    for f in all_files:
        data = torch.load(f, weights_only=True)
        remaining = num_needed - total
        if remaining <= 0:
            break
        selected.extend(data[:remaining])
        total += min(len(data), remaining)
    
    return selected[:num_needed]


class HiCDataset(Dataset):
    def __init__(self, data_files):
        self.data = []  # To store all sequences and HiC vectors
        
        # Load and process the data files
        for file in data_files:
            print("Loading file:", file)
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


def data_loader_for_precise_bn(loader, device):
    for data, _ in loader:
        yield data.to(device)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ohe_sequence, hic_vector = self.data[idx]
        
        # Fix shape: if it's [1, 4, L], squeeze to [4, L]
        if ohe_sequence.ndim == 3 and ohe_sequence.shape[0] == 1:
            ohe_sequence = ohe_sequence.squeeze(0)
        
        return ohe_sequence, hic_vector


def train(model, device, train_loader, val_loader, optimizer, epoch, best_val_loss, epochs_no_improve, weight_clip_value, save_model=False, save_model_path=None, scaler=None):
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

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # Normalize to get the mean loss
    train_loss /= num_batches    
    
    # # Apply Precise BatchNorm Updates
    print("Updating BatchNorm statistics with preciseBP...")
    # # update_bn_stats(model, (data.to(device) for data, _ in train_loader), num_iters=len(train_loader))
    update_bn_stats(model, data_loader_for_precise_bn(train_loader, device), num_iters=min(len(train_loader), 200))
    # with multiple GPUs
    # update_bn_stats(model.module, (data.to(device) for data, _ in train_loader), num_iters=200)
    
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


def objective(trial):
    # === Tunable hyperparameter ===
    shuffled_ratio = trial.suggest_float("shuffled_ratio", 0.0, 0.1)

    # === Config ===
    data_dir = "/scratch1/smaruj/train_pytorch_akita/mouse_data/Hsieh2019_mESC_data_local"
    shuffled_dir = "/scratch1/smaruj/background_generation/training_shuffled_pt_files/"
    val_fold = "fold1"
    test_fold = "fold0"
    batch_size = 4
    num_epochs = 15
    early_stop_patience = 5
    
    torch.manual_seed(42 + trial.number)
    
    # === Get real data ===
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
    real_files = [f for f in all_files if test_fold not in f and val_fold not in f]
    val_files = [f for f in all_files if val_fold in f]

    real_dataset = HiCDataset(real_files)
    N_real = len(real_dataset)
    N_shuffled = int(shuffled_ratio * N_real)

    shuffled_data = load_shuffled_sequences(shuffled_dir, N_shuffled)
    shuffled_dataset = ListDataset(shuffled_data)
    
    train_dataset = ConcatDataset([real_dataset, shuffled_dataset])
    val_dataset = HiCDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === Model & Optimizer ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SeqNN().to(device)
    # fine-tuning
    model = torch.load("/home1/smaruj/pytorch_akita/model_v2_mouse_model0_target0.pth", map_location=device)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.001, weight_decay=1.5e-5)

    # === Training Loop ===
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        _, val_loss, best_val_loss, epochs_no_improve = train(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            epochs_no_improve=epochs_no_improve,
            weight_clip_value=10.0,
            save_model=False,
            save_model_path=None,
            scaler=None
        )

        trial.report(val_loss, epoch)
        if trial.should_prune() or epochs_no_improve >= early_stop_patience:
            raise optuna.TrialPruned()

    return best_val_loss

def main():
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize",  # MSE loss
                                pruner=pruner,
                                study_name="shuffled_ratio_tuning",
                                storage="sqlite:///optuna_shuffled_ratio.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=30, timeout=60*60*120)
    
    print("Best shuffled_ratio:", study.best_params["shuffled_ratio"])
    print("Best validation loss:", study.best_value)
    
    study.trials_dataframe().to_csv("./optuna_shuffled_ratio_tuning.csv", index=False)
        
if __name__ == '__main__':
    main()

