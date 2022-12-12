import pickle
from experiment_eval import eval_adv, eval_ben
from art.defences.preprocessor.mp3_compression_pytorch import Mp3CompressionPyTorch

mp3compress = Mp3CompressionPyTorch(16000)

with open ('../datasets/gt.pkl', 'rb') as fp:
    labels = pickle.load(fp)

bb_results = []
for eps in [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]:
    bb_results.append(eval_adv('../datasets/', eps, mp3compress, labels, 30))

    print(eps, bb_results)