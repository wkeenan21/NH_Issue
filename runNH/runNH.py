import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
sys.path.append(r"T:\ProjectWorkspace\EPA\SSWR_EPA_project\Gage_time_series\Will_Keenan\NeuralHydrology\neuralhydrology-master")
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.datasetzoo.willsdataset import WillsDataset

# by default we assume that you have at least one CUDA-capable NVIDIA GPU

start_run(config_file=Path(r"T:\ProjectWorkspace\EPA\SSWR_EPA_project\Gage_time_series\Will_Keenan\NeuralHydrology\runNH\will.yml"), gpu=-1)


