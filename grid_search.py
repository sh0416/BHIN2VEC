import os
import subprocess

alpha_list = [0.05, 0.1, 0.2]
lr2_list = [0.0025, 0.025]
gpu_idx = 2
gpu_num = 4

for alpha in alpha_list:
    for lr2 in lr2_list:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        #subprocess.Popen(['nohup', 'python', 'train.py', '--dataset', 'dblp', '--batch_size', '128', '--alpha', str(alpha), '--lr2', str(lr2)], env=env, close_fds=True)
        subprocess.Popen(['nohup', 'python', 'evaluate_node_classification.py', '--dataset', 'dblp-expert-knowledge', '--batch_size', '128', '--alpha', str(alpha), '--lr2', str(lr2)], env=env, close_fds=True)
        gpu_idx = (gpu_idx + 1) % gpu_num
