
batch_size      = 128
in_seq_length   = 4 * 24
out_seq_length  = 4 * 2
num_neighbour   = 10
dim_hidden      = 128
dim_features_info = 131
dim_features_time = 6
dim_features    = dim_features_info + dim_features_time

full_length     = 61 * 24 * 4
valid_length    = 2900

start_id        = 100
pad_id          = 0
end_id          = 101

epoch           = 100
save_p_epoch    = 5
test_p_epoch    = 10

data_path       = "../../data/"
result_path     = "../results/"

model_path      = "../models/"
logs_path       = "../logs/"
figs_path       = "../figs/"

import utils
global_start_time = utils.now2string()

import numpy as np
np.set_printoptions(
    linewidth=150,
    formatter={'float_kind': lambda x: "%.4f" % x}
)

impact_k        = 50

all_model_stage_epoch = 100

gpu_memory      = 0.5
