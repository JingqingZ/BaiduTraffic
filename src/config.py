
batch_size      = 128
in_seq_length   = 4 * 24
out_seq_length  = 4 * 2
num_neighbour   = 10

# TODO update
dim_hidden      = 128
query_dim_hidden = 128
# dim_hidden      = 64
# query_dim_hidden = 32

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
test_p_epoch    = 5

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

impact_k        = 150
# 150 epoch 20 for query_comb
# 300 epoch 30 for query_comb

# all_model_stage_epoch = [100 + 1, 150 + 1]
all_model_stage_epoch = [100 + 1, 130 + 1]
# all_model_stage_epoch = [30 + 1, 40 + 1]

gpu_memory      = 1

