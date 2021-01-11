import unittest
import torch
from multimodalattentivepooling.model.attentivepooling import CorpusAttentivePool
from multimodalattentivepooling.model.encoder import WordEncoder

class TestStringMethods(unittest.TestCase):

    def test_attentive_pool(self):
        # create visual words, textual words
        N,C,T,H,W = 4, 128, 21, 55, 33 # batch size, input channels, time, height, width
        K = 12 # num words
        vw = torch.randint(0,255, size=(N,C,T,H,W)).float()
        w = torch.randn(N,K,C).float()
        # create pooling layer
        kernel_size = (1, 3, 3) # T H W
        stride = (1, 1, 1)
        padding = (0, 1, 1)
        dilation = (1, 1, 1)
        op = "max"
        pool = CorpusAttentivePool(kernel_size, stride, padding, dilation, op)
        # forward
        rslt = pool(vw, w)
        self.assertEqual(rslt.shape, vw.shape)

    def test_word_encoder(self):
        S, N, E = 12, 4, 128
        words = torch.randn(S, N, E)
        we = WordEncoder(d_model=E)
        rslt = we(words)
        self.assertEqual(rslt.shape, words.shape)

if __name__ == '__main__':
    unittest.main()
