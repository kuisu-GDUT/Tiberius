import os
import sys
import unittest
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda:2"

    def test_tf_model(self):
        # test_model.TestModel.test_tf_model
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        import tensorflow.keras as keras


        print(tf.__version__)
        print(tf.config.list_physical_devices('GPU'))
        logger.info("test tf model")

    def test_learnMSA(self):
        # test_model.TestModel.test_learnMSA
        import learnMSA.mas_hmm.Viterbi
        import learnMSA.msa_hmm.Viterbi
