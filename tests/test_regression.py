import json
import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cdsd'))

from train_classifier import train as train_cls
from train_separator import train as train_sep
from train_fully_supervised_separator import train as train_sup_sep


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRAIN_CONFIG_DIR = os.path.join(DATA_DIR, "train_configs")
REGRESSION_DIR = os.path.join(DATA_DIR, "regression_data")

CLASSIFIER_CLIP_CONFIG_PATH = os.path.join(TRAIN_CONFIG_DIR, "crnn_classifier_clip.json")
CLASSIFIER_FRAME_CONFIG_PATH = os.path.join(TRAIN_CONFIG_DIR, "crnn_classifier_frame.json")
SEPARATOR_CLIP_CONFIG_PATH = os.path.join(TRAIN_CONFIG_DIR, "rnn_separator_crnn_classifier_clip_frozen.json")
SEPARATOR_FRAME_CONFIG_PATH = os.path.join(TRAIN_CONFIG_DIR, "rnn_separator_crnn_classifier_frame_frozen.json")
SEPARATOR_SUPERVISED_CONFIG_PATH = os.path.join(TRAIN_CONFIG_DIR, "rnn_separator_supervised.json")

CLASSIFIER_CLIP_REGRESSION_DIR = os.path.join(REGRESSION_DIR, "crnn_classifier_clip")
CLASSIFIER_FRAME_REGRESSION_DIR = os.path.join(REGRESSION_DIR, "crnn_classifier_frame")
SEPARATOR_CLIP_REGRESSION_DIR = os.path.join(REGRESSION_DIR, "rnn_separator_crnn_classifier_clip_frozen")
SEPARATOR_FRAME_REGRESSION_DIR = os.path.join(REGRESSION_DIR, "rnn_separator_crnn_classifier_frame_frozen")
SEPARATOR_SUPERVISED_REGRESSION_DIR = os.path.join(REGRESSION_DIR, "rnn_separator_supervised")


CONFIG_FNAME = "config.json"
CLASSIFIER_CHECKPOINT_FNAME = "classifier_epoch-0.pt"
CLASSIFIER_BEST_FNAME = "classifier_best.pt"
CLASSIFIER_LATEST_FNAME = "classifier_latest.pt"
CLASSIFIER_EARLYSTOPPING_FNAME = "classifier_earlystopping.pt"
SEPARATOR_CHECKPOINT_FNAME = "separator_epoch-0.pt"
SEPARATOR_BEST_FNAME = "separator_best.pt"
SEPARATOR_LATEST_FNAME = "separator_latest.pt"
SEPARATOR_EARLYSTOPPING_FNAME = "separator_earlystopping.pt"
OPTIMIZER_LATEST_FNAME = "optimizer_latest.pt"
HISTORY_FNAME = "history.csv"
MASK_DEBUG_FNAME = "mask_debug_0.npz"


def ordered(obj):
    # https://stackoverflow.com/a/25851972
    if isinstance(obj, dict):
        try:
            return sorted((k, ordered(v)) for k, v in obj.items())
        except TypeError:
            return [(k, ordered(v)) for k, v in obj.items()]
    elif isinstance(obj, list):
        try:
            return sorted(ordered(x) for x in obj)
        except TypeError:
            return [ordered(x) for x in obj]
    else:
        return obj


def assert_json_equal(a_path, b_path):
    with open(a_path, 'r') as f:
        a = json.load(f)
    with open(b_path, 'r') as f:
        b = json.load(f)
    return ordered(a) == ordered(b)


def assert_torch_weights_equal(a_path, b_path):
    device = torch.device('cpu')
    a_obj = torch.load(a_path, map_location=device)
    b_obj = torch.load(b_path, map_location=device)
    assert set(a_obj.keys()) == set(b_obj.keys())
    for k in a_obj.keys():
        a_weights = a_obj[k]
        b_weights = b_obj[k]
        assert torch.allclose(a_weights, b_weights)


def assert_csv_equal(a_path, b_path):
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    pd.testing.assert_frame_equal(a_df, b_df)


def assert_npz_equal(a_path, b_path):
    a_obj = np.load(a_path)
    b_obj = np.load(b_path)
    assert set(a_obj.keys()) == set(b_obj.keys())
    for k in a_obj.keys():
        a_arr = a_obj[k]
        b_arr = b_obj[k]
        assert np.allclose(a_arr, b_arr)


def test_regression():
    # Create output in /tmp
    output_dir = "/tmp/cdsd-test-output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(CLASSIFIER_CLIP_CONFIG_PATH, 'r') as f:
            cls_clip_config = json.load(f)
        cls_clip_output_dir = os.path.join(output_dir, "crnn_classifier_clip")
        train_cls(root_data_dir=AUDIO_DIR,
                  train_config=cls_clip_config,
                  output_dir=cls_clip_output_dir,
                  num_data_workers=1,
                  checkpoint_interval=1)

        cls_clip_history_path = os.path.join(cls_clip_output_dir, HISTORY_FNAME)
        cls_clip_clsmodel_best_path = os.path.join(cls_clip_output_dir, CLASSIFIER_BEST_FNAME)

        cls_clip_history_reg_path = os.path.join(CLASSIFIER_CLIP_REGRESSION_DIR, HISTORY_FNAME)
        cls_clip_clsmodel_best_reg_path = os.path.join(CLASSIFIER_CLIP_REGRESSION_DIR, CLASSIFIER_BEST_FNAME)

        # Make sure outputs are the same
        assert_csv_equal(cls_clip_history_path, cls_clip_history_reg_path)
        assert_torch_weights_equal(cls_clip_clsmodel_best_path, cls_clip_clsmodel_best_reg_path)

        with open(CLASSIFIER_FRAME_CONFIG_PATH, 'r') as f:
            cls_frame_config = json.load(f)
        cls_frame_output_dir = os.path.join(output_dir, "crnn_classifier_frame")
        train_cls(root_data_dir=AUDIO_DIR,
                  train_config=cls_frame_config,
                  output_dir=cls_frame_output_dir,
                  num_data_workers=1,
                  checkpoint_interval=1)

        cls_frame_history_path = os.path.join(cls_frame_output_dir, HISTORY_FNAME)
        cls_frame_clsmodel_best_path = os.path.join(cls_frame_output_dir, CLASSIFIER_BEST_FNAME)

        cls_frame_history_reg_path = os.path.join(CLASSIFIER_FRAME_REGRESSION_DIR, HISTORY_FNAME)
        cls_frame_clsmodel_best_reg_path = os.path.join(CLASSIFIER_FRAME_REGRESSION_DIR, CLASSIFIER_BEST_FNAME)

        # Make sure outputs are the same
        assert_csv_equal(cls_frame_history_path, cls_frame_history_reg_path)
        assert_torch_weights_equal(cls_frame_clsmodel_best_path, cls_frame_clsmodel_best_reg_path)

        with open(SEPARATOR_CLIP_CONFIG_PATH, 'r') as f:
            sep_clip_config = json.load(f)
        sep_clip_output_dir = os.path.join(output_dir, "rnn_separator_crnn_classifier_clip_frozen")
        train_sep(root_data_dir=AUDIO_DIR,
                  train_config=sep_clip_config,
                  output_dir=sep_clip_output_dir,
                  num_data_workers=1,
                  checkpoint_interval=1,
                  num_debug_examples=1,
                  save_debug_interval=1)

        sep_clip_history_path = os.path.join(sep_clip_output_dir, HISTORY_FNAME)
        sep_clip_sepmodel_best_path = os.path.join(sep_clip_output_dir, SEPARATOR_BEST_FNAME)
        sep_clip_clsmodel_best_path = os.path.join(sep_clip_output_dir, CLASSIFIER_BEST_FNAME)
        sep_clip_mask_debug_path = os.path.join(sep_clip_output_dir, MASK_DEBUG_FNAME)

        sep_clip_history_reg_path = os.path.join(SEPARATOR_CLIP_REGRESSION_DIR, HISTORY_FNAME)
        sep_clip_sepmodel_best_reg_path = os.path.join(SEPARATOR_CLIP_REGRESSION_DIR, SEPARATOR_BEST_FNAME)
        sep_clip_clsmodel_best_reg_path = os.path.join(SEPARATOR_CLIP_REGRESSION_DIR, CLASSIFIER_BEST_FNAME)
        sep_clip_mask_debug_reg_path = os.path.join(SEPARATOR_CLIP_REGRESSION_DIR, MASK_DEBUG_FNAME)

        # Make sure outputs are the same
        assert_csv_equal(sep_clip_history_path, sep_clip_history_reg_path)
        assert_npz_equal(sep_clip_mask_debug_path, sep_clip_mask_debug_reg_path)
        assert_torch_weights_equal(sep_clip_sepmodel_best_path, sep_clip_sepmodel_best_reg_path)
        assert_torch_weights_equal(sep_clip_clsmodel_best_path, sep_clip_clsmodel_best_reg_path)

        with open(SEPARATOR_FRAME_CONFIG_PATH, 'r') as f:
            sep_frame_config = json.load(f)
        sep_frame_output_dir = os.path.join(output_dir, "rnn_separator_crnn_classifier_frame_frozen")
        train_sep(root_data_dir=AUDIO_DIR,
                  train_config=sep_frame_config,
                  output_dir=sep_frame_output_dir,
                  num_data_workers=1,
                  checkpoint_interval=1,
                  num_debug_examples=1,
                  save_debug_interval=1)

        sep_frame_history_path = os.path.join(sep_frame_output_dir, HISTORY_FNAME)
        sep_frame_sepmodel_best_path = os.path.join(sep_frame_output_dir, SEPARATOR_BEST_FNAME)
        sep_frame_clsmodel_best_path = os.path.join(sep_frame_output_dir, CLASSIFIER_BEST_FNAME)
        sep_frame_mask_debug_path = os.path.join(sep_frame_output_dir, MASK_DEBUG_FNAME)

        sep_frame_history_reg_path = os.path.join(SEPARATOR_FRAME_REGRESSION_DIR, HISTORY_FNAME)
        sep_frame_sepmodel_best_reg_path = os.path.join(SEPARATOR_FRAME_REGRESSION_DIR, SEPARATOR_BEST_FNAME)
        sep_frame_clsmodel_best_reg_path = os.path.join(SEPARATOR_FRAME_REGRESSION_DIR, CLASSIFIER_BEST_FNAME)
        sep_frame_mask_debug_reg_path = os.path.join(SEPARATOR_FRAME_REGRESSION_DIR, MASK_DEBUG_FNAME)

        # Make sure outputs are the same
        assert_csv_equal(sep_frame_history_path, sep_frame_history_reg_path)
        assert_npz_equal(sep_frame_mask_debug_path, sep_frame_mask_debug_reg_path)
        assert_torch_weights_equal(sep_frame_sepmodel_best_path, sep_frame_sepmodel_best_reg_path)
        assert_torch_weights_equal(sep_frame_clsmodel_best_path, sep_frame_clsmodel_best_reg_path)

        with open(SEPARATOR_SUPERVISED_CONFIG_PATH, 'r') as f:
            sup_sep_config = json.load(f)
        sup_sep_output_dir = os.path.join(output_dir, "rnn_separator_supervised")
        train_sup_sep(root_data_dir=AUDIO_DIR,
                      train_config=sup_sep_config,
                      output_dir=sup_sep_output_dir,
                      num_data_workers=1,
                      checkpoint_interval=1,
                      num_debug_examples=1,
                      save_debug_interval=1)

        sup_sep_history_path = os.path.join(sup_sep_output_dir, HISTORY_FNAME)
        sup_sep_sepmodel_best_path = os.path.join(sup_sep_output_dir, SEPARATOR_BEST_FNAME)
        sup_sep_mask_debug_path = os.path.join(sup_sep_output_dir, MASK_DEBUG_FNAME)

        sup_sep_history_reg_path = os.path.join(SEPARATOR_SUPERVISED_REGRESSION_DIR, HISTORY_FNAME)
        sup_sep_sepmodel_best_reg_path = os.path.join(SEPARATOR_SUPERVISED_REGRESSION_DIR, SEPARATOR_BEST_FNAME)
        sup_sep_mask_debug_reg_path = os.path.join(SEPARATOR_SUPERVISED_REGRESSION_DIR, MASK_DEBUG_FNAME)

        # Make sure outputs are the same
        assert_csv_equal(sup_sep_history_path, sup_sep_history_reg_path)
        assert_npz_equal(sup_sep_mask_debug_path, sup_sep_mask_debug_reg_path)
        assert_torch_weights_equal(sup_sep_sepmodel_best_path, sup_sep_sepmodel_best_reg_path)
    finally:
        # Clean up all output
        shutil.rmtree(output_dir)
