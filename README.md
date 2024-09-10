# Enhancing the Performance of Multi-Vehicle Navigation in Unstructured Environments using Hard Sample Mining

**Yining Ma, Ang Li, Qadeer Khan and Daniel Cremers**

[Project](https://yininghase.github.io/multiagent-collision-mining/) | [ArXiv](http://arxiv.org/abs/2409.05119)

This repository contains code for the paper **Enhancing the Performance of Multi-Vehicle Navigation in Unstructured Environments using Hard Sample Mining**. 

Contemporary research in autonomous driving has demonstrated tremendous potential in emulating the traits of human driving. However, they primarily cater to areas with well built road infrastructure and appropriate traffic management systems. Therefore, in the absence of traffic signals or in unstructured environments, these self-driving algorithms are expected to fail. This paper proposes a strategy for autonomously navigating multiple vehicles in close proximity to their desired destinations without traffic rules in unstructured environments. 

Graphical Neural Networks (GNNs) have demonstrated good utility for this task of multi-vehicle control. Among the different alternatives of training GNNs, supervised methods have proven to be most data-efficient, albeit requiring ground truth labels. However, these labels may not always be available, particularly in unstructured environments without traffic regulations. Therefore, a tedious optimization process may be required to determine them while ensuring that the vehicles reach their desired destination and do not collide with each other or any obstacles. Therefore, in order to expedite the training process, it is essential to reduce the optimization time and select only those samples for labeling that add most value to the training.  

In this paper, we propose a warm start method that first uses a pre-trained model trained on a simpler subset of data. Inference is then done on more complicated scenarios, to determine the hard samples wherein the model faces the greatest predicament. This is measured by the difficulty vehicles encounter in reaching their desired destination without collision. Experimental results demonstrate that mining for hard samples in this manner reduces the requirement for supervised training data by 10 fold. Moreover, we also use the predictions of this simpler pre-trained model to initialize the optimization process, resulting in a further speedup of up to 1.8 times.
<br />

![image](./images/overview.png)
<br />
<br />

## Comparison of Baseline Model and Improved Models

To show the effectiveness of hard data mining, we choose the pre-trained model as baseline model[1] and additionally train the baseline with additional random data. Here we show the comparison of our model (**Baseline with additional Hard Data**) with **Baseline Model** and **Baseline with additional Random Data**  for two different scenarios.

As can be seen, only our model is capable of simultaneously driving all the vehicles to their desired destinations without collision. For all other models, the vehicles collide with each other. 

 
**Scenario 1**:

<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
          <strong>Our Model (Baseline with additional Hard Data)</strong>      
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/improved_model_with_hard_data_1.gif">
        </figure>
    </td>
  </tr>
</table>
<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
        Baseline Model         
        </text> 
    </td>
    <td width="50%">
        <text>
        Baseline with additional Random Data
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/baseline_model_1.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/improved_model_with_random_data_1.gif">
        </figure>
    </td>
  </tr>
</table>


**Scenario 2**:

<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
          <strong>Our Model (Baseline with additional Hard Data)</strong>      
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/improved_model_with_hard_data_2.gif">
        </figure>
    </td>
  </tr>
</table>
<table style="table-layout: fixed; word-break: break-all; word-wrap: break-word;" width="100%">
  <tr>
    <td width="50%">
        <text>
        Baseline Model         
        </text> 
    </td>
    <td width="50%">
        <text>
        Baseline with additional Random Data
        </text>
    </td>
  </tr>
  <tr>
    <td width="50%">
        <figure>
            <img src="./images/baseline_model_2.gif">
        </figure>
    </td>
    <td width="50%">
        <figure>
            <img src="./images/improved_model_with_random_data_2.gif">
        </figure>
    </td>
  </tr>
</table>

More scenarios are shown in the [project page](https://yininghase.github.io/multiagent-collision-mining/).

For the interested reader, the project page also contains **Probability Density Function of Collision Rate** and **Robustness Analysis for Steering Angle Noise and Position Noise**.

[1]: Y. Ma, Q. Khan and D. Cremers, "Multi Agent Navigation in Unconstrained Environments using a Centralized Attention based Graphical Neural Network Controller," 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), Bilbao, Spain, 2023, pp. 2893-2900, doi: 10.1109/ITSC57777.2023.10422072.


## GNN Model Inference Runtime

Here we show the results of the average runtime per inference step of our model for 8 different vehicle/obstacle configurations. The tests were done both on a GeForce RTX2070 GPU and an Intel Core i7-10750H CPU.

| Num. of Vehicle | Num. of Obstacle | Runtime on GPU (s) | Runtime on CPU (s) |
| --------------- | ---------------- | ------------------ | ------------------ |
| 8               | 0                | 0.00775            | 0.00823            |
| 8               | 1                | 0.00788            | 0.00874            |
| 10              | 0                | 0.00773            | 0.01027            |
| 10              | 1                | 0.00804            | 0.01144            |
| 12              | 0                | 0.00801            | 0.01314            |
| 12              | 1                | 0.00807            | 0.01391            |
| 15              | 0                | 0.00805            | 0.01604            |
| 20              | 0                | 0.00806            | 0.02440            |


## Environment

Clone the repo and build the conda environment:
```
conda create -n <env_name> python=3.7 
conda activate <env_name>
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index torch-scatter --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install scipy
pip install --no-index torch-sparse --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install --no-index torch-cluster --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install --no-index torch-spline-conv --no-cache-dir -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-geometric==2.0.4
pip install pyyaml
pip install matplotlib
```


## Pipeline

The pipeline is based on the pipeline in [this repository](https://github.com/yininghase/multi-agent-control) adding with the collision mining part.

The extended test dataset containing 8 vehicles or more can be downloaded [here](https://cvg.cit.tum.de/webshare/g/papers/khamuham/multiagent-data/test_dataset_extension_8-20_vehicles.zip)

To run collision mining:

1. Run inference.py to get the predicted trajectories from previous GNN models.

2. Run calculate_metrics.py to check the collisions for each trajectories. The collision rates are saved in the file metrics.pt. 

3. Rank the collision rates and pick up the trajectories with high collision rates.

4. Run mpc.py to to get the ground truth trajectories and add them to the training dataset by modifying the setting [hard data mining](./configs/train.yaml/#L31) in [the config of training](./configs/train.yaml)

5. To save the runtime of MPC, the predict results of GNN can serve as initialization of MPC by modifying the setting [control init](./configs/configs/generate_trainval_data.yaml/#L31) in [the config of generate trainval data](./configs/configs/generate_trainval_data.yaml)
