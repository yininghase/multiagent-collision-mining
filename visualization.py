import os
import torch

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from calculate_metrics import (check_collision_rectangular_circle, 
                               check_collision_rectangular_rectangular)


class Visualize_Trajectory:
    def __init__(self, simulation_options, show_attention=False):
        
        self.simulation_options = simulation_options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5),
                     ]
        self.show_attention = show_attention

    # start: [num_vehicle, [x, y, psi]]
    # target: [num_vehicle, [x, y, psi]]
    
    def base_plot(self, is_trajectory):
        
        if self.show_attention:
            self.fig = plt.figure(figsize=(2*self.simulation_options["figure size"], 
                                        self.simulation_options["figure size"]))
            self.ax = self.fig.add_subplot(1,2,1)
            self.ax_ = self.fig.add_subplot(1,2,2)
            
        else:
            self.fig = plt.figure(figsize=(self.simulation_options["figure size"], 
                                        self.simulation_options["figure size"]))
            self.ax = self.fig.add_subplot()
            
        self.ax.set_xlim(-self.simulation_options["figure limit"], 
                 self.simulation_options["figure limit"])
        self.ax.set_ylim([-self.simulation_options["figure limit"], 
                          self.simulation_options["figure limit"]])
        self.ax.set_xticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))
        
        self.ax.set_yticks(np.arange(-self.simulation_options["figure limit"], 
                             self.simulation_options["figure limit"], 
                             step = self.simulation_options["ticks step"]))

        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.patch_target = []
        self.patch_target_arrow = []
        self.predicts_opt = []
        self.predicts_model = []
        self.predicts_init = []

        patch_obs = []
        
        start = self.simulation_options["start"]
        target = self.simulation_options["target"]
        
        start_new = self.car_patch_pos(self.simulation_options["start"])
        target_new = self.car_patch_pos(self.simulation_options["target"])

        for i in range(self.simulation_options["num of vehicles"]):
            # cars
            
            patch_car = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i])
            patch_car.set_xy(start_new[i,:2])
            patch_car.angle = np.rad2deg(start_new[i,2])-90
            self.patch_vehicles.append(patch_car)
            
            patch_car_arrow = mpatches.FancyArrow(start[i,0]-0.9*np.cos(start[i,2]), 
                                                  start[i,1]-0.9*np.sin(start[i,2]), 
                                                  1.5*np.cos(start[i,2]), 
                                                  1.5*np.sin(start[i,2]), 
                                                  width=0.1, color='w')
            self.patch_vehicles_arrow.append(patch_car_arrow)

            patch_goal = mpatches.Rectangle([0,0], 
                                            self.simulation_options["car size"][0], 
                                            self.simulation_options["car size"][1], 
                                            color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            
            patch_goal.set_xy(target_new[i,:2])
            patch_goal.angle = np.rad2deg(target_new[i,2])-90
            self.patch_target.append(patch_goal)
            
            patch_goal_arrow = mpatches.FancyArrow(target[i,0]-0.9*np.cos(target[i,2]), 
                                                   target[i,1]-0.9*np.sin(target[i,2]), 
                                                   1.5*np.cos(target[i,2]), 
                                                   1.5*np.sin(target[i,2]), 
                                                   width=0.1, 
                                                   color=self.cmap[i])
            self.patch_target_arrow.append(patch_goal_arrow)

            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)
            self.ax.add_patch(patch_car_arrow)
            self.ax.add_patch(patch_goal_arrow)

            self.frame = plt.text(12, 12, "", fontsize=15)

            # trajectories
            if self.simulation_options["show optimization"]:
                if is_trajectory:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1)
                elif i == 0:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="Optimization")
                else:
                    predict_opt, = self.ax.plot([], [], 'r--', linewidth=1, label="_Optimization")
                self.predicts_opt.append(predict_opt)
            
                if self.simulation_options["control init"] is not None:
                    if is_trajectory:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1)
                    elif i == 0:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1, label="Initialization")
                    else:
                        predict_init, = self.ax.plot([], [], 'b--', linewidth=1, label="_Initialization")
                    self.predicts_init.append(predict_init)
            
            # if self.simulation_options["is model"]:
            #     if is_trajectory:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1)
            #     elif i == 0:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="Model Prediction")
            #     else:
            #         predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="_Model Prediction")
            #     self.predicts_model.append(predict_model)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"vehicle {i+1}")
        
        for i, obs in enumerate(self.simulation_options["obstacles"]):
            patch_obs.append(mpatches.Circle(obs[:2], obs[2], color=self.cmap[i], fill=True))
            self.ax.add_patch(patch_obs[-1])
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"obstacle {i+1}")
        
        if not is_trajectory:   
            self.ax.legend(loc='upper left', fontsize=12)
    
    
    def create_video(self, data, predict_opt, predict_model, attention=None):
        self.base_plot(is_trajectory=False)
        self.data = data
        
        if self.simulation_options["is model"]:
            self.predict_model = predict_model
            
        if self.simulation_options["show optimization"]:
            self.predict_opt = predict_opt 
            
        self.attention = attention
            
        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(data)-1), interval=100, repeat=True, blit=False)
        
        if self.simulation_options["save plot"]:
            if not os.path.exists(self.simulation_options["plot folder"]):
                os.makedirs(self.simulation_options["plot folder"])
                
            car_animation.save(os.path.join(self.simulation_options["plot folder"], 
                                            self.simulation_options["name"] + ".gif"))
            
        if self.simulation_options["show plot"]:
            plt.show()

    def update_plot(self, num):
        
        data = self.data[num,...]               
        
        # self.frame.set_text("Frame: " + str(num))
        for i in range(self.simulation_options["num of vehicles"]):
            # vehicle
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))
            
            
            if self.simulation_options["show optimization"]:
                self.predicts_opt[i].set_data(self.predict_opt[num, :, i, 0], self.predict_opt[num, :, i, 1])
                
                
            # if self.simulation_options["is model"]:
            #     self.predicts_model[i].set_data(self.predict_model[num, :, i, 0], self.predict_model[num, :, i, 1])
        
            # self.velocities[i].set_text("Velocity: " + str(round(self.data[num, i, 3], 2)))
        
        
        if self.show_attention and self.attention is not None:
            self.ax_.imshow(self.attention[num], vmin=-2.5, vmax=2.5, cmap="gray")
            self.ax_.set_xticks(ticks=[i for i in range(self.simulation_options["num of vehicles"]+self.simulation_options["num of obstacles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])] + \
                                         [f"obstacle {i+1}" for i in range(self.simulation_options["num of obstacles"])])
            self.ax_.set_yticks(ticks=[i for i in range(self.simulation_options["num of vehicles"])],
                                labels = [f"vehicle {i+1}" for i in range(self.simulation_options["num of vehicles"])])
            
    def plot_trajectory(self, points):
        self.base_plot(is_trajectory=True)
        max_time = points.shape[0]
        
        for i in range(self.simulation_options["num of vehicles"]):
            veh_points = points[:, i, :2][:,None,:]
            segments = np.concatenate([veh_points[:-1], veh_points[1:]], axis=1)
            norm = plt.Normalize(0, max_time)
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            # lc = LineCollection(segments, colors=self.cmap[i])
            lc.set_array(range(points.shape[0]))
            lc.set_linewidth(2)
            line = self.ax.add_collection(lc)
        
        collision = np.zeros((points.shape[0], points.shape[1]), dtype=bool)
        
        for i in range(self.simulation_options["num of vehicles"]-1):
            for j in range(i+1, self.simulation_options["num of vehicles"]):
                collisions_ij = check_collision_rectangular_rectangular(torch.from_numpy(points[:,i,:]).type(torch.float32), 
                                                                        torch.from_numpy(points[:,j,:]).type(torch.float32), 
                                                                        vehicle_size=self.simulation_options["car size"]).numpy()
                collision[collisions_ij,i]=True
                collision[collisions_ij,j]=True
        
        for i in range(self.simulation_options["num of vehicles"]):
            for j in range(self.simulation_options["num of obstacles"]):
                obstacle_j = self.simulation_options["obstacles"][j]
                obstacle_j = np.concatenate((np.zeros(4),obstacle_j,np.ones(1)))[None,:]
                
                collisions_ij = check_collision_rectangular_circle(torch.from_numpy(points[:,i,:]).type(torch.float32), 
                                                                   torch.from_numpy(obstacle_j).type(torch.float32), 
                                                                   vehicle_size=self.simulation_options["car size"]).numpy()
                
                collision[collisions_ij,i]=True
               
        cbar = self.fig.colorbar(line, ax=self.ax)
        cbar.ax.set_ylabel("Timestep", fontsize=15)
        
        if np.sum(collision)>0:
            self.ax.scatter(points[collision][:,0], points[collision][:,1], s=15, c="r", marker="o")
        
        if self.simulation_options["save plot"]:
            if not os.path.exists(self.simulation_options["plot folder"]):
                os.makedirs(self.simulation_options["plot folder"])
            plt.savefig(os.path.join(self.simulation_options["plot folder"], 
                                     self.simulation_options["name"]+".png"), bbox_inches='tight')
        
        if self.simulation_options["show plot"]:
            plt.show()

    
    def plot_initialization_optimization(self, data, predict_opt, predict_init):
        
        self.base_plot(is_trajectory=False)
        self.data = data
        
        if self.simulation_options["control init"] is not None:
            self.predict_init = predict_init
            
        if self.simulation_options["show optimization"]:
            self.predict_opt = predict_opt 
        
        
        data = self.data[0,...]               
        
        # self.frame.set_text("Frame: " + str(num))
        for i in range(self.simulation_options["num of vehicles"]):
            # vehicle
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))
            
            
            if self.simulation_options["show optimization"]:
                self.predicts_opt[i].set_data(self.predict_opt[0, :, i, 0], self.predict_opt[0, :, i, 1])
                
                
            if self.simulation_options["control init"] is not None:
                self.predicts_init[i].set_data(self.predict_init[0, :, i, 0], self.predict_init[0, :, i, 1])
                
        
        if self.simulation_options["save plot"]:
            if not os.path.exists(self.simulation_options["plot folder"]):
                os.makedirs(self.simulation_options["plot folder"])
            plt.savefig(os.path.join(self.simulation_options["plot folder"], 
                                     self.simulation_options["name"]+".png"), bbox_inches='tight')
        
        if self.simulation_options["show plot"]:
            plt.show()
           
    
    def car_patch_pos(self, posture):
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.cos(posture[...,2])*(self.simulation_options["car size"][1]/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.simulation_options["car size"][0]/2) \
                                        - np.sin(posture[...,2])*(self.simulation_options["car size"][1]/2)
        
        return posture_new
    
    """
    Creates a heatmap
    state_data: dims (simulation_length, num_vehicles, 4)
    """
    def calculate_cost(self, coordinates, targets):
        
        dist_cost = self.simulation_options["distance_cost"]
        obst_cost = self.simulation_options["obstacle_cost"]
        obs_radius = self.simulation_options["obstacle_radius"]
        obstacles = self.simulation_options["obstacles"]
        num_obstacles = self.simulation_options["num of obstacles"]
        
        loss = np.linalg.norm(coordinates - targets[None, None, :2], ord=2, axis=-1)*dist_cost
        
        if  obst_cost > 0 and num_obstacles > 0:
            
            # dist = torch.norm(preds[:,1:,:,:2]-obs[:,None,None,:2], dim=-1, p=2)-obs[:,None,None,2]
            # dist1 = torch.clip(dist, min=0, max=None) + 1e-8
            # loss += torch.sum((1/dist1 - 1/self.obs_radius) * (dist1 < self.obs_radius), dim=(-1,-2)) * self.obs_cost
            # # dist2 = (torch.clip(-dist, min=0, max=None) + 100)**4 - 1e8
            # dist2 = torch.exp(torch.clip(-dist, min=0, max=None) + 10) - np.exp(10)
            # loss += torch.sum(dist2, dim=(-1,-2)) * self.obs_cost
            dist = np.linalg.norm(coordinates[:,:,None,:]-obstacles[None,None,:,:2], ord=2, axis=-1)-obstacles[None,None,:,2]-obs_radius
            dist = (np.clip(-dist, a_min=0, a_max=None))**2
            loss += np.sum(dist, axis=-1) * obst_cost
        
        return loss
