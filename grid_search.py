import os
import itertools
import functools
import subprocess
import multiprocessing
import tqdm

def execute(d, k, m, device):
    """
    command_args = ['python',
                    'train_deepwalk.py',
                    '--dataset', 'yelp',
                    '--d', str(d),
                    '--k', str(k),
                    '--m', str(m)]
    """
    command_args = ['python',
                    'train_metapath2vec.py',
                    '--dataset', 'yelp',
                    '--metapath', 'URWRBRWR',
                    '--d', str(d),
                    '--k', str(k),
                    '--m', str(m)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    proc = subprocess.Popen(command_args, env=env)
    proc.wait()


if __name__=='__main__':
    dlist = [8, 16, 32, 64, 128, 256]
    klist = [1, 3, 5, 7, 9]
    mlist = [1, 3, 5, 7, 9]
    params = [dlist, klist, mlist]
    params = list(map(lambda x: (*x[0], x[1]),
                      zip(itertools.product(*params),
                          itertools.cycle(range(4)))))

    with multiprocessing.Pool(processes=8) as pool:
        total = len(params)
        for _ in pool.starmap(execute, params):
            print('%d / %d' % (cnt, total))
            cnt += 1

    """
    procs = []
    for embedding in os.listdir('output'):
        command_args = ['python', 'eval.py', '--embedding', os.path.join('output', embedding), '--result', os.path.join('result', embedding+'.csv')]
        procs.append(subprocess.Popen(command_args, stdout=subprocess.PIPE))

    exit_codes = [p.wait() for p in procs]
    """
