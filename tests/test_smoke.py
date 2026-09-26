"""Duman testi: ağ kurulur, eğitim ağırlıkları değiştirir, sorgu çalışır."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dnn  # noqa: E402


def test_training_updates_weights_and_query_runs():
    nn = dnn.neuralNetwork(inputnodes=5, hiddennodes=[4, 10, 2], outputnodes=3, learningrate=0.5)
    before = copy.deepcopy(nn.allweights)
    x, y = [[0, 0, 0, 0, 1], [1, 1, 1, 1, 0]], [[0, 1, 0], [1, 1, 1]]
    for _ in range(5):
        nn.train(inputs_list=x, targets_list=y)
    assert any((before[k] != v).any() for k, v in nn.allweights.items())
    nn.query(x)
