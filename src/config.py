
batch_size      = 128
in_seq_length   = 4 * 24
out_seq_length  = 4 * 2
num_neighbour   = 6
dim_hidden      = 512

valid_length    = 1000

start_id        = 100
pad_id          = 0
end_id          = 101

epoch           = 20

data_path       = "../../data/"
result_path     = "../results/"

model_path      = "../models/"
logs_path       = "../logs/"
figs_path       = "../figs/"

import utils
global_start_time = utils.now2string()
