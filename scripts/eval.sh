# this script is for evaluating a given checkpoint.
# example to evaluate our GNFactor:
#       bash scripts/eval.sh GNFACTOR_BC released


# some params specified by user
method_name=$1
addition_info=$2
exp_name="${method_name}_${addition_info}"

# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu="0"


# cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"


cd GNFactor

# eval stage
CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
    method.name=$method \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.start_seed=${seed} \

