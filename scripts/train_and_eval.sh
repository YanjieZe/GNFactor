# this script is for training and evaluating multi-task BC agents.
# example to run our GNFactor:
#       bash scripts/train_and_eval.sh GNFACTOR_BC
# example to run baseline PerAct:
#       bash scripts/train_and_eval.sh PERACT_BC


# some params specified by user
method=${1}

######
# the following params could be also specified by user. we set them here for convenience.
# for more params, please refer to `conf/` folder.
#######
# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu, but need to set `num_devices=1`.
train_gpu="0,1"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu="0"
# set the port for ddp training.
port="12345"
# you could disable wandb by this.
use_wandb=True


cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"


exp_name="${method}_${addition_info}"
replay_dir="${cur_dir}/replay/${exp_name}"


cd GNFactor

# training stage
CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${train_demo_path} \
    replay.path=${replay_dir} \
    framework.start_seed=${seed} \
    framework.use_wandb=${use_wandb} \
    method.use_wandb=${use_wandb} \
    framework.wandb_group=${exp_name} \
    ddp.num_devices=2 \
    ddp.master_port=${port}

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

# eval stage
CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
    method.name=$method \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.start_seed=${seed} \
