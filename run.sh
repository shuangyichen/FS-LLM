#!/bin/sh
#SBATCH --job-name=rolora
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --qos=normal
#SBATCH --time=9:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
date;hostname;pwd
. ~/condaenvs/fs-llm-1

# ~/condaenvs/fs-llm-1/bin/python3.9 -c "import numpy;print(numpy.__version__);print(numpy.__file__)";
# ~/condaenvs/fs-llm-1/bin/python3.9 federatedscope/main.py --cfg federatedscope/llm/baseline/exp_yaml/gsm/gsm_federate.yaml
# ~/condaenvs/fs-llm-1/bin/python3.9 federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg federatedscope/llm/baseline/exp_yaml/gsm/gsm_federate.yaml
~/condaenvs/fs-llm-1/bin/python3.9 federatedscope/main.py --cfg federatedscope/llm/baseline/exp_yaml/alpaca/alpaca_federate.yaml
# ~/condaenvs/fs-llm-1/bin/python3.9 federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg federatedscope/llm/baseline/exp_yaml/alpaca/alpaca_federate.yaml