import subprocess

if __name__ == "__main__":
    file_names = ['deepwalk_douban_movie.npy']#, 'hin2vec_douban_movie.npy', 'LINE_douban_movie.npy', 'experimental_douban_movie_128_100_5_5_1.npy', 'metapaht2vec_douban_movie.npy']
    for file_name in file_names:
        for i in range(2):#5):
            print("!--- run({}) : {} ---!".format(i, file_name))
            file_path = 'output/' + file_name
            proc = subprocess.Popen(['python', 'evaluate_link_prediction.py', 
                '--dataset', 'douban_movie', '--embedding', file_path])
            stdout, stderr = proc.communicate()
            print(stdout)
