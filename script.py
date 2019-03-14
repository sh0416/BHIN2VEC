import os
import subprocess

embeddings = os.listdir('output')

with open('result.log', 'w') as f:
    for _ in range(5):
        for embedding in embeddings:
            p = subprocess.Popen(['python',
                                  'evaluate_link_prediction.py',
                                  '--dataset',
                                  'douban_movie',
                                  '--embedding',
                                  os.path.join('output', embedding)],
                                 stdout=subprocess.PIPE)
            out, err = p.communicate()
            f.write(embedding+out.decode('utf-8'))

