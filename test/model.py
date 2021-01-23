import unittest
import torch
from multimodalattentivepooling.model.modchunks import ModulatedChunks
from multimodalattentivepooling.model.encoder import SeqEncoder
from pathlib import Path

class TestDatasets(unittest.TestCase):

    def test_word_phrase_module(self):
        # visual features
        B, C, L = 4, 32, 62
        vis_feats = torch.arange(B*C*L).view(B, L, C).float()
        # words
        NW, WD = 9, 32
        words = torch.arange(B*NW*WD).view(B, NW, WD).float()
        # phrases
        NP, PD = 25, 32
        phrases = torch.arange(B*NP*PD).view(B, NP, PD).float()
        # encoders
        window_size = 12
        num_chunks = 3
        word_encoder = SeqEncoder(d_model=WD, d_out=C)
        phrase_encoder = SeqEncoder(d_model=PD, d_out=C)
        wpa = ModulatedChunks(window_size, num_chunks, word_encoder, phrase_encoder)
        out = wpa({"vis_feats":vis_feats,"query":words,"phrases":phrases})
        B1, NW, NCH, C1 = out.shape
        self.assertEqual(B, B1)
        self.assertEqual(C, C1)
        self.assertEqual(NCH,num_chunks)
    def test_seq_encoder(self):
        B, T, D = 12, 4, 128
        words = torch.randn(B, T, D)
        se = SeqEncoder(d_model=D)
        rslt = se(words)
        self.assertEqual(rslt.shape, words.shape)


if __name__ == '__main__':
    unittest.main()