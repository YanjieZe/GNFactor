<p align="center">

  <h1 align="center"><a href="https://yanjieze.com/GNFactor/">GNFactor</a>:
Multi-Task Real Robot Learning with <a>G</a>eneralizable <a>N</a>eural Feature <a>F</a>ields</h1>
<h2 align="center">CoRL 2023 <a href="https://openreview.net/forum?id=b1tl3aOt2R2">Oral</a></h2>
  <p align="center">
    <a><strong>Yanjie Ze*</strong></a>
    ·
    <a><strong>Ge Yan*</strong></a>
    ·
    <a><strong>Yueh-Hua Wu*</strong></a>
    ·
    <a><strong>Annabella Macaluso</strong></a>
    ·
    <a><strong>Yuying Ge</strong></a>
    ·
    <a><strong>Jianglong Ye</strong></a>
    ·
    <a><strong>Nicklas Hansen</strong></a>
    ·
    <a><strong>Li Erran Li</strong></a>
    ·
    <a><strong>Xiaolong Wang</strong></a>
  </p>

</p>

<h3 align="center">
  <a href="https://yanjieze.com/GNFactor/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2308.16891"><strong>arXiv</strong></a>
  |
  <a href="https://twitter.com/xiaolonw/status/1697654370492469414?s=20"><strong>Twitter</strong></a>
</h3>

<div align="center">
  <img src="./docs/gnfactor_method.png" alt="Logo" width="100%">
</div>

https://github.com/YanjieZe/GNFactor/assets/59699800/0ad86126-ad24-4291-a4ba-6bb591d9934a



# 💻 Installation

See [INSTALL.md](docs/INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](docs/ERROR_CATCH.md) for error catching I personally encountered during installation.

**Note**: Due to the usage of `RLBench` and its physics engine `PyRep`, a lof of potential bugs during installing the environment could occur. These could be hard to address for people that are not familiar with the environment. If you encounter any problems during installation, please feel free to open an issue.

# 🧾 Brief Introduction to our Pipeline
Our setting is multi-task behavior cloning. The pipeline is generally as:
1. **Generate demonstrations**.
  - We study *10* RLBench tasks and each task uses *20* demonstrations.
  - We do not provide demonstration files on the cloud server, because of large file size (~92GB). 
  - We directly provide the scripts to generate demonstrations.
2. **Train a policy**.
  - We use *2* gpus (~24G each) and train *100k* iterations for one policy. 
  - It is ok to use only *1* gpu, if you only have one. But the results would be worse.
3. **Evaluate the policy**. 
  - For solid results, we evaluete *25* episodes for each task across each saved checkpoint.
  - In our paper, we report the results of the final checkpoint across *3* seeds.
  - But, we also find that our method could get a better number if we take the best checkpoint (see our paper appendix for results).

Don't worry about these complex illustrations and parameters! we provide all scripts to use just with one click.

# 🛠️ Usage
We provide scripts to generate demonstrations, train and evaluate policies, and draw results based on the evaluation results. 

We also provide the results (*.csv files, 3 seeds) of our paper in `GNFactor_results/`. Researchers could directly compare their own algorithm with these results under the similar setting.


## 🦉 Generate Demonstrations
In this work, we select 10 challenging tasks from RLBench:
```bash
tasks=[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]
```
We also encourage users to explore more other tasks.


For each task, we generate *20* demonstrations for training and *25* demonstrations for evaluation. To explain more, the demonstrations for evaluation are only to make sure the trajectory is feasible (see [this issue](https://github.com/peract/peract/issues/3)).


To generate demonstrations for a task, run:
```bash
bash scripts/gen_demonstrations.sh YOUR_TASK_NAME
```
For example, running `bash scripts/gen_demonstrations.sh close_jar` will generate demonstrations for the task `close_jar` under the folder `data/`.

To generate demonstrations for all 10 tasks, just simply run this command for different tasks.


## 📈 Training and Evaluation
We use wandb to log some curves and visualizations. Login to wandb before running the scripts.
```bash
wandb login
```
Then, to run GNFactor, use the following command:
```bash
bash scripts/train_and_eval.sh GNFACTOR_BC
```
We also provide the baseline PerAct in our repo. To run PerAct, use the following command:
```bash
bash scripts/train_and_eval.sh PERACT_BC
```
By default we use two gpus to train. 


Considering the possible gpu resource limitations, if you only have one gpu, you could use the following command to run with only one gpu:
```bash
bash scripts/train_and_eval_with_one_gpu.sh GNFACTOR_BC
bash scripts/train_and_eval_with_one_gpu.sh PERACT_BC
```
But this means the batch size is 1 and the results would be worse.


Configurations for training and evaluation are in `GNFactor/conf/`. You could change the parameters in these files for your own setting.


## 📊 Analyze Evaluation Results
For the users' convenience, we provide the results (`*.csv` files, 3 seeds) of our paper in `GNFactor_results/`.

We also provide a python script `scripts/compute_results.py` to compute the average success rates from the `*.csv` files (generated during the evaluation). An example to compute success rates given our provided csv files:

```
python scripts/compute_results.py --file_paths GNFactor_results/0.csv GNFactor_results/1.csv GNFactor_results/2.csv --method last
```
In our main paper, we only report the results of the **last** checkpoint for fairness. We also find that our method could get a better number if we take the **best** checkpoint. You can use `method=best` to view the results:
```
python scripts/compute_results.py --file_paths GNFactor_results/0.csv GNFactor_results/1.csv GNFactor_results/2.csv --method best
```

## 🆕 Pre-Trained Checkpoints
We provide the pre-trained checkpoints for users' convenience. You could download from the google drive [here](https://drive.google.com/file/d/1UOP9Zftk7zsfQTBVQvbm7GKz_V3-DaSx/view?usp=sharing). Then, unzip this file into `REPO_DIR/GNFactor/logs/`. The overall file structure should be like this:
```
docs/
GNFactor/
    logs/
        GNFACTOR_BC_released/ # our provided files
GNFactor_results/
```
After this, use our script to evaluate the checkpoints (remember to generate the evaluation demonstrations first):
```bash
bash scripts/eval.sh GNFACTOR_BC released
```

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 🙏 Acknowledgement

Our code is built upon [PerAct](https://github.com/peract/peract), [RLBench](https://github.com/stepjam/RLBench), [pixelNeRF](https://github.com/sxyu/pixel-nerf), [ODISE](https://github.com/NVlabs/ODISE), and [CLIP](https://github.com/openai/CLIP). We thank all these authors for their nicely open sourced code and their great contributions to the community.

# 📝 Citation

If you find our work useful, please consider citing:
```
@article{Ze2023GNFactor,
  title={Multi-Task Real Robot Learning with Generalizable Neural Feature Fields},
  author={Yanjie Ze and Ge Yan and Yueh-Hua Wu and Annabella Macaluso and Yuying Ge and Jianglong Ye and Nicklas Hansen and Li Erran Li and Xiaolong Wang},
  journal={CoRL}, 
  year={2023},
}
```