import os
import subprocess

alpha_list = [0.1, 0.2]
lr2_list = [0.25]
gpu_idx = 0
gpu_num = 4

for alpha in alpha_list:
    for lr2 in lr2_list:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        """
        subprocess.Popen(['nohup', 'python', 'train.py',
                         '--dataset', 'blog-catalog',
                         '--batch_size', '32',
                         '--k', '5',
                          '--d', '128',
                          '--epoch', '10',
                          '--alpha', str(alpha),
                          '--lr', '0.0025',
                          '--lr2', str(lr2)], env=env, close_fds=True)
        subprocess.Popen(['nohup', 'python', 'evaluate_node_classification.py',
                          '--dataset', 'dblp',
                          '--batch_size', '128',
                          '--d', '128',
                          '--alpha', str(alpha),
                          '--lr', '0.0025',
                          '--lr2', str(lr2)], env=env, close_fds=True)
        """
        subprocess.Popen(['nohup', 'python', 'evaluate_link_prediction.py',
                          '--dataset', 'blog-catalog',
                          '--batch_size', '32',
                          '--k', '5',
                          '--epoch', '10',
                          '--d', '128',
                          '--alpha', str(alpha),
                          '--lr', '0.0025',
                          '--lr2', str(lr2)], env=env, close_fds=True)

        gpu_idx = (gpu_idx + 1) % gpu_num
