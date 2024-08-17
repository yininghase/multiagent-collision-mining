import os

import torch
import numpy as np
from argparse import ArgumentParser

from data_process import load_data, load_yaml 


def calculate_metrics(config): 
    
    assert os.path.exists(config["data folder"]), \
        f"The given folder of '{config['data folder']}' does not exist!"        
    
    data = load_data(num_vehicles = config["number of vehicles"], 
                     num_obstacles = config["num of obstacles"],
                     folders = config["data folder"],
                     load_all_simpler = True,
                     horizon = 1,
                     load_trajectory=True,
                     load_model_prediction=True)
    
    metrics= {"success to goal":{},
              "collision states": {},
              "collision times": {},
              "travel distance":{},
              "success to goal rate":{},
              "collision rate": {},
              "number of trajectories": {},
              "trajectory efficiency": {},
              }
    
    for key, value in data.items():
        
        len_batch = key[0]
        len_vehicle = key[1]
        len_obstacle = key[0]-key[1]
        
        states, _, _, trajectory_idx = value
        metrics["number of trajectories"].update({key:len(trajectory_idx)-1})
        
        states = states.reshape(-1, len_batch, 8)
        trajectory_idx = (trajectory_idx/len_batch).type(torch.long)
        
        trajectory = states[:,:len_vehicle,:2]
        
        travel_distance = torch.norm(torch.diff(trajectory, dim=0, prepend=trajectory[0,:,:][None,...]), p=2, dim=-1)
        travel_distance = torch.cumsum(travel_distance, dim=0)
        travel_distance = travel_distance[trajectory_idx[1:]-1]-travel_distance[trajectory_idx[:-1]]
        metrics["travel distance"].update({key:travel_distance})
        
        collision_states, vehicle_with_collision = calculate_collision_times(states, vehicle_size=config["car size"], 
                                                    len_batch=len_batch, len_vehicle=len_vehicle, len_obstacle=len_obstacle)
        
        metrics["collision states"].update({key:collision_states})
        
        if key == (1,1):
            
            collision_times = torch.zeros((len(trajectory_idx)-1,1))
            metrics["collision times"].update({key:collision_times})
            metrics["collision rate"].update({key:0})
        
        else:
            collision_states = collision_states.clone().type(torch.int)
            collision_states = torch.clip(torch.diff(collision_states, dim=0, prepend=torch.zeros((1,collision_states.shape[1]))), min=0, max=None)
            collision_times = torch.cumsum(collision_states, dim=0)
            
            collision_times = torch.cat((torch.zeros((1,collision_times.shape[1])),collision_times))
            collision_times = collision_times[trajectory_idx[1:]] - collision_times[trajectory_idx[:-1]]
            metrics["collision times"].update({key:collision_times})
            collision_rate = torch.sum(collision_times)/torch.sum(travel_distance)
            metrics["collision rate"].update({key:collision_rate})
        
        vehicle_with_collision = torch.cat((torch.zeros((1,vehicle_with_collision.shape[1]), dtype=torch.bool), 
                                            vehicle_with_collision))
        vehicle_with_collision = torch.cumsum(vehicle_with_collision, dim=0)
        vehicle_with_collision = (vehicle_with_collision[trajectory_idx[1:]] - vehicle_with_collision[trajectory_idx[:-1]]) > 0
        
        final_states = states[trajectory_idx[1:]-1,:len_vehicle,:]
        reach_goal = calculate_reach_goal(final_states, 
                                            position_tolerance = config["position tolerance"], 
                                            angle_tolerance = config["angle tolerance"])
        
        success_to_goal = (~vehicle_with_collision) * reach_goal
        
        metrics["success to goal"].update({key:success_to_goal})
        
        success_to_goal_rate = torch.sum(success_to_goal)/(len_vehicle*len(success_to_goal))
        metrics["success to goal rate"].update({key:success_to_goal_rate})
        
        trajectory_efficiency = calculate_trajectory_efficiency(travel_distance, 
                                    states[trajectory_idx[:-1],:len_vehicle,:2], 
                                    states[trajectory_idx[:-1],:len_vehicle,4:6], 
                                    success_to_goal)
        
        metrics["trajectory efficiency"].update({key:trajectory_efficiency})
                
        print(f"number of vehicle: {len_vehicle}, number of obstacle: {len_obstacle}")
        print(f"number of trajectories: {metrics['number of trajectories'][key]}")
        print(f"success to goal rate: {metrics['success to goal rate'][key]}")
        print(f"collision rate: {metrics['collision rate'][key]}")
        print(f"trajectory efficiency: {metrics['trajectory efficiency'][key]}")
        print()
    
    if config['save metrics']:
        torch.save(metrics, os.path.join(config["data folder"], 'metrics.pt'))
               

def calculate_collision_times(states, vehicle_size, len_batch, len_vehicle, len_obstacle):
    
    vehicle_with_collision = torch.zeros((len(states),len_vehicle))
    
    collisions = torch.empty((len(states),0), dtype=bool)
    
    if len_vehicle > 1:
        for i in range(len_vehicle-1):
            for j in range(i+1,len_vehicle):
                
                collisions_ij = check_collision_rectangular_rectangular(states[:,i,:], states[:,j,:], vehicle_size)
                vehicle_with_collision[:, i] += collisions_ij
                vehicle_with_collision[:, j] += collisions_ij
                collisions = torch.cat((collisions, collisions_ij[:,None]), dim=-1)
                
    
    if len_obstacle > 0:
        
        for i in range(len_vehicle):
            for j in range(len_vehicle, len_batch):
                
                collisions_ij = check_collision_rectangular_circle(states[:,i,:], states[:,j,:], vehicle_size)
                vehicle_with_collision[:, i] += collisions_ij
                collisions = torch.cat((collisions, collisions_ij[:,None]), dim=-1)
    
    return collisions, (vehicle_with_collision>0)


def check_collision_rectangular_circle(state_rect, state_cir, vehicle_size):
    
    assert len(state_rect) == len(state_cir) or len(state_cir) == 1, \
        "Mismatch of data length in collision check of two vehicles!"
    
    vehicle_corners_left_top = torch.tensor([vehicle_size[1]/2, vehicle_size[0]/2])
    
    inv_rotation = torch.empty((len(state_rect),2,2))
    
    inv_rotation[:,0,0] = torch.cos(state_rect[:,2])
    inv_rotation[:,0,1] = torch.sin(state_rect[:,2])
    inv_rotation[:,1,0] = -torch.sin(state_rect[:,2])
    inv_rotation[:,1,1] = torch.cos(state_rect[:,2])
    
    vect_rect_to_cir = (inv_rotation @ (state_cir[:,4:6,None]-state_rect[:,:2,None])).squeeze(-1)
    
    min_dist = torch.norm(torch.clip((torch.abs(vect_rect_to_cir) - vehicle_corners_left_top), min=0, max=None),p=2,dim=-1)
    
    collision = (min_dist - state_cir[:,6]) <= 0    
    
    return collision

def check_collision_rectangular_rectangular(state_1, state_2, vehicle_size):
    
    assert len(state_1) == len(state_2), \
        "Mismatch of data length in collision check of two vehicles!"
    
    not_collision = torch.zeros(len(state_1), dtype=bool)
    
    proj_vect = []
    
    vehicle_corners = torch.tensor([[vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, vehicle_size[0]/2],
                                    [-vehicle_size[1]/2, -vehicle_size[0]/2],
                                    [vehicle_size[1]/2, -vehicle_size[0]/2],
                                    ])
    
    rotation_1 = torch.empty((len(state_1),2,2))
    
    rotation_1[:,0,0] = torch.cos(state_1[:,2])
    rotation_1[:,0,1] = -torch.sin(state_1[:,2])
    rotation_1[:,1,0] = torch.sin(state_1[:,2])
    rotation_1[:,1,1] = torch.cos(state_1[:,2])
    
    vehicle_corners_1 = (rotation_1[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_1 = vehicle_corners_1 + state_1[:,None,:2]
    
    proj_vect.append(rotation_1[:,:,0])
    proj_vect.append(rotation_1[:,:,1])
    
    rotation_2 = torch.empty((len(state_2),2,2))
    
    rotation_2[:,0,0] = torch.cos(state_2[:,2])
    rotation_2[:,0,1] = -torch.sin(state_2[:,2])
    rotation_2[:,1,0] = torch.sin(state_2[:,2])
    rotation_2[:,1,1] = torch.cos(state_2[:,2])
    
    vehicle_corners_2 = (rotation_2[:,None,:,:] @ vehicle_corners[None,:,:,None]).squeeze(-1)
    vehicle_corners_2 = vehicle_corners_2+ state_2[:,None,:2]
    
    proj_vect.append(rotation_2[:,:,0])
    proj_vect.append(rotation_2[:,:,1])
    
    for i in range(4):
        
        if torch.all(not_collision):
            break
            
        proj_vehicle_1 = (vehicle_corners_1[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        proj_vehicle_2 = (vehicle_corners_2[(not_collision==False),:,:] @ proj_vect[i][(not_collision==False),:,None]).squeeze(-1)
        
        not_collision[(not_collision==False)] |= (torch.amax(proj_vehicle_1, dim=-1)<torch.amin(proj_vehicle_2, dim=-1)) | \
                                                 (torch.amax(proj_vehicle_2, dim=-1)<torch.amin(proj_vehicle_1, dim=-1))
    
    return ~not_collision


def calculate_reach_goal(final_states, position_tolerance, angle_tolerance):
    
    pos_diff = torch.norm(final_states[:,:,:2]-final_states[:,:,4:6],p=2,dim=-1)
    
    angle_diff_1 = (final_states[:,:,2] - final_states[:,:,6])[...,None]%(2*np.pi)
    angle_diff_2 = 2*np.pi - angle_diff_1
    angle_diff = torch.amin(torch.concat((angle_diff_1,angle_diff_2), dim=-1), dim=-1)
    
    reach_goal = (pos_diff<=position_tolerance) & (angle_diff<=angle_tolerance)   
    
    return reach_goal


def calculate_trajectory_efficiency(trajectory_distance, start, goal, success_index):
        
    start_goal_distance = torch.norm(start-goal, p=2, dim=-1)[success_index]
    trajectory_distance = trajectory_distance[success_index]
    trajectory_efficiency = torch.sum(start_goal_distance)/torch.sum(trajectory_distance)
    
    return trajectory_efficiency


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/calculate_metrics.yaml", help='specify configuration path')
    args = parser.parse_args()
    
    config_path= args.config_path
    config = load_yaml(config_path)
    
    if not isinstance(config["data folder"], str):
        data_folders = config["data folder"].copy()
        
        for data_folder in data_folders:
            print(f"Currently evaluating {data_folder}: ")
            config["data folder"] = data_folder
            calculate_metrics(config)
            print("*"*10)
        
    else:
        print(f"Currently evaluating {config['data folder']}: ")
        calculate_metrics(config)
        print("*"*10)
    