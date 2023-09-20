# Is Autonomous Driving Ready for the Developing World? A Multi-Vehicle Control Network with Minimal Supervision

**Yining Ma, Qadeer Khan and Daniel Cremers**

This repository contains code for the paper **Is Autonomous Driving Ready for the Developing World? A Multi-Vehicle Control Network with Minimal Supervision** 

Although autonomous driving in recent years has shown tremendous potential in emulating the traits of human driving, it currently caters to areas with well built road infrastructure and clearly defined traffic regulations. These self-driving algorithms are expected to fail where this is not the case. This is particularly true of many places in the developing world where the infrastructure and regulations may not be standardized leading to haphazard driving behaviour among multiple vehicles. This paper proposes a strategy for autonomously navigating multiple vehicles in close proximity to their desired destinations without traffic rules or necessarily even any road infrastructure. Graphical Neural Networks (GNN) have demonstrated good utility for this task of multi-vehicle control. Among the different alternatives of training GNN's, supervised methods have proven to be most data-efficient, albeit requiring ground truth labels. However, these labels may not always be available and therefore may require a tedious optimization process to determine them. In order to expedite the training process, it is essential to reduce the optimization time and select only those samples that add most value to the training. 

In this paper, we propose a warm start method that first uses a pre-trained model trained on a simpler subset. Inference is then done on more complicated scenarios, to determine the hard samples wherein the model faces the greatest predicament. This is measured by the difficulty vehicles encounter in reaching their desired destination without collision. Experimental results demonstrate that mining for hard samples in this manner reduces the requirement for supervised training data by 10 fold. Moreover, we also use the predictions of this simpler pre-trained model to initialize the optimization process, resulting in a further speedup of up to 1.8 times.

![image](./images/overview.png)


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

[1]: Y. Ma, Q. Khan, and D. Cremers, “Multi agent navigation in unconstrained environments using a centralized attention based graphicalneural network controller,” in IEEE 26th International Conference on Intelligent Transportation Systems, 2023.


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

To be announced.