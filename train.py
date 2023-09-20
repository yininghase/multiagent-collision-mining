import os
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt 
from tqdm import tqdm
from argparse import ArgumentParser

from gnn import IterativeGNNModel
from loss import WeightedMeanSquaredLoss
from data_process import (load_data, load_yaml, split_train_valid, 
                          GNN_DataLoader, GNN_Dataset)


def train_supervised(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available now:', device)
    
    if not os.path.exists(config["model folder"]):
        os.makedirs(config["model folder"])
    
    model_name = config["model name"]
    
    os.makedirs(os.path.join(config["model folder"], model_name), exist_ok=True)
    model_path = os.path.join(config["model folder"], model_name, f"{model_name}.pth")
    log_path = os.path.join(config["model folder"], model_name, f"{model_name}_logs.txt")
    plot_path = os.path.join(config["model folder"], model_name, f"{model_name}_learning_plot.png")
    

    data = load_data(num_vehicles = config["number of vehicles"], 
                    num_obstacles = config["num of obstacles"],
                    folders = config["data folder"],
                    load_all_simpler = True,
                    horizon = 1)
    
    
    if config["hard data mining"] is not None:
        
        data_extend = load_data(num_vehicles = config["number of vehicles"], 
                                num_obstacles = config["num of obstacles"],
                                folders = config["hard data mining"]["data folder extend"],
                                load_all_simpler = True,
                                load_trajectory = True,
                                horizon = 1)
        
        folders = config["hard data mining"]["data folder extend"]
        
        collision = {}
        success = {}
        travel_distance = {}
        
        for folder in folders:
            metrics = torch.load((os.path.join(folder, 'metrics.pt')))
            collision_extend = metrics['collision times']
            success_extend = metrics['success to goal']
            travel_distance_extend = metrics['travel distance']
            
            for problem in collision_extend.keys():
                
                if problem in collision.keys():
                    collision[problem] = torch.cat((collision[problem], collision_extend[problem]), dim=0)
                    success[problem] = torch.cat((success[problem], success_extend[problem]), dim=0)
                    travel_distance[problem] = torch.cat((travel_distance[problem], travel_distance_extend[problem]), dim=0)
                
                else:
                    collision[problem] = collision_extend[problem]
                    success[problem] = success_extend[problem]
                    travel_distance[problem] = travel_distance_extend[problem]
        
        for problem in data_extend.keys():
            len_batch = problem[0]
            num_vehicle = problem[1]

            X_extend, y_extend, batches_extend, traj_extend = data_extend[problem]
            traj_extend = torch.div(traj_extend, len_batch, rounding_mode='floor') 
            
            num_batch = int(config["hard data mining"]["percent data used"]*(len(traj_extend)-1))

            ### hard data extension ###
            collision_extend = torch.sum(collision[problem], dim=-1)/torch.sum(travel_distance[problem], dim=-1)
            # success_extend = torch.sum(success[problem].long(), dim=-1)
            collision_rank = torch.argsort(collision_extend, dim=0, descending=True)
            # success_rank = torch.argsort(success_extend, dim=0)
            
            traj_idx = torch.repeat_interleave(torch.arange(len(traj_extend)-1, dtype=torch.int64), traj_extend[1:]-traj_extend[:-1])
            clip_idx = torch.isin(traj_idx, collision_rank[:num_batch])
            
            X_extend = (X_extend.reshape(len(batches_extend), len_batch, 8)[clip_idx]).reshape(-1, 8)
            y_extend = (y_extend.reshape(len(batches_extend), num_vehicle, 2)[clip_idx]).reshape(-1, 2)
            batches_extend = batches_extend[clip_idx]
            ### hard data extension ###
            

            ### random data extension ###
            # clip_idx = traj_extend[num_batch].item()
            
            # X_extend = X_extend[:clip_idx*len_batch]
            # y_extend = y_extend[:clip_idx*num_vehicle]
            # batches_extend = batches_extend[:clip_idx]
            ### random data extension ###
            

            if problem in data.keys():
                X , y, batches = data[problem]
                X = torch.cat((X, X_extend), dim=0)
                y = torch.cat((y, y_extend), dim=0)
                batches = torch.cat((batches, batches_extend), dim=0)
                
                data[problem] = [X, y, batches]
            
            else:
                data[problem] = [X_extend, y_extend, batches_extend]
            
        del data_extend, X_extend, y_extend, batches_extend, traj_extend, metrics, collision, \
            collision_extend, success, success_extend, travel_distance, travel_distance_extend
        

    train_data, valid_data = split_train_valid(data)

    train_dataset = GNN_Dataset(train_data, augmentation = config["augmentation"])
    valid_dataset = GNN_Dataset(valid_data, augmentation = config["augmentation"])
    
    train_data_num = len(train_dataset)
    print(f"Training Data: {train_data_num}")
    val_data_num = len(valid_dataset)
    print(f"Validation Data: {val_data_num}")

    train_loader = GNN_DataLoader(train_dataset, batch_size=config["batch size"], shuffle=True)
    valid_loader = GNN_DataLoader(valid_dataset, batch_size=config["batch size"], shuffle=False)

    model = IterativeGNNModel(horizon = 1,  
                            max_num_vehicles = config["number of vehicles"], 
                            max_num_obstacles = config["num of obstacles"],
                            mode = "training",
                            device = device,
                            )

    
    if config["pretrained model"] is not None: 
        model.load_state_dict(torch.load(config["pretrained model"]))

    model.to(device)
    print(model)

    optimizer = Adam(model.parameters(), 
                     lr = config["initial learning rate"], 
                     weight_decay = config["weight decay"])
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', verbose = True, 
                                  patience = config["learning rate patience"],
                                  factor = config["learning rate factor"], 
                                  min_lr = config["min learining rate"])
    
    
    criterion = WeightedMeanSquaredLoss(horizon = 1, device = device, no_weight = True)

    best_loss = np.inf

    LOGS = {
        "training_loss": [],
        "validation_loss": [],
    }

    
    f = open(log_path, 'w+')
    
    print("Start training")

    for epoch in tqdm(range(config["train max epochs"])):
        # === TRAIN ===
        # Sets the model in evaluation mode
        train_loss = 0
        valid_loss = 0
        model.train()

        for (inputs, targets, batches) in train_loader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)
            batches = batches.to(device)
            
            y_hat, y_static = model(inputs, batches)                

            loss = criterion(y_hat, targets, y_static)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(targets)
        
        train_loss /= train_data_num
        LOGS["training_loss"].append(train_loss)

        # === EVAL ===
        # Sets the model in evaluation mode
        model.eval()

        with torch.no_grad():
            for (inputs, targets, batches) in valid_loader:
                
                inputs = inputs.to(device)    
                targets = targets.to(device)
                batches = batches.to(device)

                y_hat, y_static = model(inputs, batches)
                
                loss = criterion(y_hat, targets, y_static)
                valid_loss += loss.item()*len(targets)
            
            valid_loss /= val_data_num
            LOGS["validation_loss"].append(valid_loss)
        
        scheduler.step(valid_loss)
            
        msg = f'Epoch: {epoch+1}/{config["train max epochs"]} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        with open(log_path, 'a+') as f:
            print(msg,file=f)
            print(msg)

        if valid_loss < best_loss:
    
            best_loss = valid_loss
            # Reset patience (because we have improvement)
            patience_f = config["train patience"]
            torch.save(model.state_dict(), model_path)               
            print('Model Saved!')
                
        else:
            # Decrease patience (no improvement in ROC)
            patience_f -= 1
            if patience_f == 0:
                print(f'Early stopping (no improvement since {config["train patience"]} models) | Best Valid Loss: {valid_loss:.6f}')
                break
    
    del train_dataset, valid_dataset, train_loader, valid_loader, inputs, targets, batches
    
    # plot training and validation loss
    plt.plot(LOGS["training_loss"], label='Training Loss')
    plt.plot(LOGS["validation_loss"], label='Validation Loss')
    plt.text(x = 0, y = 0, s=f"best valid loss is {best_loss:6f}")
    plt.legend()
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/train.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    train_supervised(config)