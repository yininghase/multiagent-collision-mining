import os
import copy
import time
import warnings
import torch
import numpy as np
from tqdm import tqdm
from simulation import sim_run
from data_process import get_problem, load_yaml, load_data

from argparse import ArgumentParser

warnings.filterwarnings("ignore")
    
def length(data):
    
    length = 0
    
    for value in data.values():
        batch = value["batches_data"]
        length += len(batch)
    
    return length
        

def generate_trainval_data(config):
    '''function to collect training data for supervised learning'''
    
    if not os.path.exists(config["data folder"]):
        os.makedirs(config["data folder"])
    
    # problem with different number of vehicles and obstacles
    # [[num_vehicles, num_obstacles], ...]
   
    problem_collection = np.array(config['problem collection'])
    assert problem_collection.shape[1] == 2 and \
            len(problem_collection.shape) == 2 and \
            np.amin(problem_collection[:,0]) >= 1 and \
            np.amin(problem_collection[:,1]) >= 0, \
            "Invalid input of problem_collection!"
    
    horizon = config['horizon'] 
    control_init = copy.copy(config['control init'])
    save_plot = copy.copy(config["save plot"])
    
    data = {}
    data_index = np.empty((0,3), dtype=int)
    control_init_dict = {}
    
    if control_init is not None:
        config["simulation time"] = 1
        config["collect trajectory"] = False
    
    for problem in problem_collection:
        
        num_vehicles, num_obstacles = problem
        
               
        data[(num_vehicles, num_obstacles)] = {
                "X_data": torch.tensor([]), 
                "y_GT_data": torch.tensor([]), 
                "batches_data": torch.tensor([]),
                "X_data_path": os.path.join(config["data folder"], 
                                            f"X_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                "y_GT_data_path": os.path.join(config["data folder"], 
                                            f"y_GT_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                "batches_data_path": os.path.join(config["data folder"], 
                                                f"batches_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"), 
            }
        
        if config["collect trajectory"]:
            
            data[(num_vehicles, num_obstacles)].update({
                    "trajectory_data": torch.tensor([0]),
                    "trajectory_data_path": os.path.join(config["data folder"], 
                                                    f"trajectory_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt"),
                })

        if control_init is not None:
            assert os.path.exists(config['control init']), "'control init' does not exists!"
            data_read = load_data(num_vehicles, num_obstacles, load_all_simpler=False, folders=config['control init'], 
                                  load_trajectory=True, load_model_prediction=True)
            
            key = list(data_read.keys())
            assert len(key) == 1, 'Error in data_read.keys()!'
            key = key[0]
            
            X_data, y_data, batches_data, trajectory_data = data_read[key]
            
            X_data = X_data.reshape(-1, key[0], 8)
            y_data = y_data.reshape(-1, key[1], 2)
            
            control_init_dict[key] = [X_data, y_data, batches_data, trajectory_data]
            data_index = np.concatenate((data_index, np.concatenate((batches_data, np.arange(len(batches_data))[:, None]), axis=1).astype(int)), axis=0)
            
            trajectory_data_path = os.path.join(config["data folder"], f"trajectory_data_vehicle={num_vehicles}_obstalce={num_obstacles}.pt")
            torch.save(trajectory_data, trajectory_data_path)   
            

    if control_init is not None:
        config["simutaion runs"] = len(data_index)
    
    # time_list = []
    # num_step_list = []    
    
    for i in tqdm(range(config["simutaion runs"])):
        
        # show first 5 trajectories to check the quality of ground truth data
        # if i >= 5:
        #     config["save plot"] = False
        
        config["control init trajectory"] = None
        
        if control_init is not None:
            
            len_batch, num_vehicles, idx = data_index[i]
            num_obstacles = len_batch - num_vehicles
            
            X = control_init_dict[(len_batch, num_vehicles)][0][idx]
            traj_indices = (control_init_dict[(len_batch, num_vehicles)][3]//len_batch).long()
            
            start = X[:num_vehicles, :4].numpy()
            target = X[:num_vehicles, 4:7].numpy()
            obstacles = X[num_vehicles:,4:7].numpy()
            
            #####
            # if idx not in traj_indices:
            #     continue
            #####
            
            traj_idx = torch.bucketize(idx, traj_indices, right=True)
            y = control_init_dict[(len_batch, num_vehicles)][1][idx:min(idx+horizon, traj_indices[traj_idx])].numpy()
            
            if len(y) < horizon:
                y = np.concatenate((y, np.ones((horizon-len(y), 1, 1))*y[-1]), axis=0)
            
            config["start"] = start
            config["target"] = target
            config["obstacles"] = obstacles
            config["name"] = f"generate_data_{i}"
            config["num of vehicles"] = num_vehicles
            config["num of obstacles"] = num_obstacles
            config["control init"] = y # None
            
            config["save plot"] = (save_plot and i%5 ==0)
            
            if config["save plot"] == True:
                X = control_init_dict[(len_batch, num_vehicles)][0][idx:min(idx+horizon, traj_indices[traj_idx]), :4, :].numpy()
                
                if len(X) < horizon:
                    X = np.concatenate((X, np.ones((horizon-len(X), 1, 1))*X[-1]), axis=0)
                    
                config["control init trajectory"] = X
            
        else:
            num_vehicles, num_obstacles = problem_collection[int(i%len(problem_collection))]
            start, target, obstacles = get_problem(num_vehicles, num_obstacles, 
                                                    collision = config["collision mode"],
                                                    parking = config["parking mode"],
                                                    mode = "generate train trajectory")
            
            config["start"] = start
            config["target"] = target
            config["obstacles"] = obstacles
            config["name"] = f"generate_data_{i}"
            config["num of vehicles"] = num_vehicles
            config["num of obstacles"] = num_obstacles

        # start_time = time.time()
        X_data, batches_data, y_GT_data, _, success, num_step = sim_run(config) 
        # end_time = time.time()
        
        # time_list.append(end_time-start_time)
        # num_step_list.append(num_step)
    
        
        if (not success) and (not control_init):
            print(f"The current problem of {num_vehicles} vehicle(s) and {num_obstacles} obstacle(s) failed!")
            if (not config["save failed samples"]):
                print("The failed data is not saved!")
                continue
        
        data[(num_vehicles, num_obstacles)]["X_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["X_data"], X_data))
        data[(num_vehicles, num_obstacles)]["y_GT_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["y_GT_data"], y_GT_data))
        data[(num_vehicles, num_obstacles)]["batches_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["batches_data"], batches_data))
        
        if config["collect trajectory"]:
            data[(num_vehicles, num_obstacles)]["trajectory_data"] = torch.cat((data[(num_vehicles, num_obstacles)]["trajectory_data"], 
                                                                                torch.tensor([data[(num_vehicles, num_obstacles)]["trajectory_data"][-1]+len(X_data)])))
        
        # save the data and the labels
        print(f"Saving data at step {i+1}")
        print(f"Datapoints collected: {length(data)}")
        
        torch.save(data[(num_vehicles, num_obstacles)]["X_data"], data[(num_vehicles, num_obstacles)]["X_data_path"])
        torch.save(data[(num_vehicles, num_obstacles)]["y_GT_data"], data[(num_vehicles, num_obstacles)]["y_GT_data_path"])
        torch.save(data[(num_vehicles, num_obstacles)]["batches_data"], data[(num_vehicles, num_obstacles)]["batches_data_path"])
        if config["collect trajectory"]:
            torch.save(data[(num_vehicles, num_obstacles)]["trajectory_data"], data[(num_vehicles, num_obstacles)]["trajectory_data_path"])
        
    for problem in problem_collection:
        
        num_vehicles, num_obstacles = problem
        if len(data[(num_vehicles, num_obstacles)]["X_data"]) > 0:
            torch.save(data[(num_vehicles, num_obstacles)]["X_data"], data[(num_vehicles, num_obstacles)]["X_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["y_GT_data"], data[(num_vehicles, num_obstacles)]["y_GT_data_path"])
            torch.save(data[(num_vehicles, num_obstacles)]["batches_data"], data[(num_vehicles, num_obstacles)]["batches_data_path"])
            if config["collect trajectory"]:
                torch.save(data[(num_vehicles, num_obstacles)]["trajectory_data"], data[(num_vehicles, num_obstacles)]["trajectory_data_path"])   

    # time_list = np.array(time_list)
    # num_step_list = np.array(num_step_list)
    
    # average_run_time_per_task = np.mean(time_list)
    # print(f"Average run time per task: {average_run_time_per_task}")
    # average_run_time_per_step = np.sum(time_list)/np.sum(num_step_list)
    # print(f"Average run time per step: {average_run_time_per_step}")

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/generate_trainval_data.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    generate_trainval_data(config)
   



    
    