
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import sklearn
import random
import sklearn.utils
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import threading_data
from datetime import datetime, timedelta

import log
import model
import utils
import config
import dataloader


class Controller():

    def __init__(self, model, base_epoch):
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess = tf.Session()
        tl.layers.initialize_global_variables(self.sess)

        self.base_epoch = base_epoch
        self.__init_path__()
        self.__init_mkdir__()

    def save_model(self, path, global_step=None):
        save_path = self.saver.save(self.sess, path, global_step=global_step)
        print("[S] Model saved in ckpt %s" % save_path)
        return save_path

    def restore_model(self, path, global_step=None):
        model_path = "%s-%s" % (path, global_step)
        self.saver.restore(self.sess, model_path)
        print("[R] Model restored from ckpt %s" % model_path)
        return True

    def __init_path__(self):
        self.model_save_dir = config.model_path + self.model.model_name + "/"
        self.log_save_dir = config.logs_path + self.model.model_name + "/"
        self.figs_save_dir = config.figs_path + self.model.model_name + "/"

    def __init_mkdir__(self):
        dirlist = [
            self.model_save_dir,
            self.log_save_dir,
            self.figs_save_dir
        ]
        utils.make_dirlist(dirlist)

    def __train__(self, epoch, root_data, neighbour_data, logger):
        # shuffle data batch
        root_data, neighbour_data = sklearn.utils.shuffle(
            root_data,
            neighbour_data
        )

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))
        random.shuffle(train_order)

        all_loss = np.zeros(11)

        start_time = time.time()
        step_time = time.time()
        train_steps = total_batch_size // config.batch_size

        for cstep in range(train_steps):

            x_root, x_neight, decode_seq, target_seq = dataloader.get_minibatch(
                root_data,
                neighbour_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            global_step = cstep + epoch * train_steps

            results = self.sess.run([
                self.model.mae_copy,
                self.model.train_loss,
                self.model.nmse_train_loss,
                self.model.nmse_train_noend,
                self.model.mse_train_noend,
                self.model.mae_train_noend,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.mse_test_noend,
                self.model.mae_test_noend,
                self.model.learning_rate,
                self.model.optim],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.x_neighbour: x_neight,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                    self.model.global_step: global_step,
                })

            all_loss += np.array(results[:11])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()
                logger.add_log(global_step, all_loss / (cstep + 1))

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )
        logger.add_log(global_step, all_loss / train_steps)

        return all_loss


    def __valid__(self, epoch, root_data, neighbour_data, logger):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))

        all_loss = np.zeros(6)

        start_time = time.time()
        step_time = time.time()
        valid_steps = total_batch_size // config.batch_size

        for cstep in range(valid_steps):

            x_root, x_neight, decode_seq, target_seq = dataloader.get_minibatch(
                root_data,
                neighbour_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            results = self.sess.run([
                self.model.mae_copy,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.mse_test_noend,
                self.model.mae_test_noend],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.x_neighbour: x_neight,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                })

            all_loss += np.array(results[:6])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Valid] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s" %
                    (epoch, cstep, valid_steps, time.time() - step_time, all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[Valid Sum] Epoch: [%3d] time: %.4f, loss: %s" %
            (epoch, time.time() - start_time, all_loss / valid_steps)
        )
        logger.add_log(epoch, all_loss / valid_steps)

        return all_loss


    def controller_train(self, tepoch=config.epoch):
        root_data, neighbour_data = dataloader.load_data(3, 3)

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        logger_train = log.Logger(columns=["mae_copy", "loss", "nmse_train", "nmse", "mse", "mae", "lossv", "nmse_test", "nmsev", "msev", "maev"])
        logger_valid = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev"])

        for epoch in range(tepoch + 1):

            self.__train__(global_epoch, root_data[:, :-config.valid_length, :], neighbour_data[:, :-config.valid_length, :], logger_train)

            if epoch % 1 == 0:
                self.__valid__(global_epoch, root_data[:, -config.valid_length:, :], neighbour_data[:, -config.valid_length:, :], logger_valid)

            if global_epoch > self.base_epoch and global_epoch % 1 == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            logger_train.save(self.log_save_dir + config.global_start_time + "_train.csv")
            logger_valid.save(self.log_save_dir + config.global_start_time + "_valid.csv")

            global_epoch += 1

class Seq2Seq_Controller(Controller):

    def __train__(self, epoch, root_data, logger):
        # shuffle data batch
        root_data = sklearn.utils.shuffle(
            root_data
        )

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))
        random.shuffle(train_order)

        all_loss = np.zeros(11)

        start_time = time.time()
        step_time = time.time()
        train_steps = total_batch_size // config.batch_size

        for cstep in range(train_steps):

            x_root, decode_seq, target_seq = dataloader.get_minibatch_all(
                root_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            global_step = cstep + epoch * train_steps

            results = self.sess.run([
                self.model.mae_copy,
                self.model.train_loss,
                self.model.nmse_train_loss,
                self.model.nmse_train_noend,
                self.model.mse_train_noend,
                self.model.mae_train_noend,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.mse_test_noend,
                self.model.mae_test_noend,
                self.model.learning_rate,
                self.model.optim],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                    self.model.global_step: global_step,
                })

            all_loss += np.array(results[:11])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()
                logger.add_log(global_step, all_loss / (cstep + 1))

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )
        logger.add_log(global_step, all_loss / train_steps)

        return all_loss


    def __valid__(self, epoch, root_data, logger):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))

        all_loss = np.zeros(6)

        start_time = time.time()
        step_time = time.time()
        valid_steps = total_batch_size // config.batch_size

        for cstep in range(valid_steps):

            x_root, decode_seq, target_seq = dataloader.get_minibatch_all(
                root_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            results = self.sess.run([
                self.model.mae_copy,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.mse_test_noend,
                self.model.mae_test_noend],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                })

            all_loss += np.array(results[:6])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Valid] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s" %
                    (epoch, cstep, valid_steps, time.time() - step_time, all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[Valid Sum] Epoch: [%3d] time: %.4f, loss: %s" %
            (epoch, time.time() - start_time, all_loss / valid_steps)
        )
        logger.add_log(epoch, all_loss / valid_steps)

        return all_loss


    def controller_train(self, tepoch=config.epoch):
        root_data  = dataloader.load_data_all()

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        logger_train = log.Logger(columns=["mae_copy", "loss", "nmse_train", "nmse", "mse", "mae", "lossv", "nmse_test", "nmsev", "msev", "maev"])
        logger_valid = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev"])

        for epoch in range(tepoch + 1):

            self.__train__(global_epoch, root_data[:, :-config.valid_length, :], logger_train)

            if epoch % 1 == 0:
                self.__valid__(global_epoch, root_data[:, -config.valid_length:, :], logger_valid)

            if global_epoch > self.base_epoch and global_epoch % 1 == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            logger_train.save(self.log_save_dir + config.global_start_time + "_train.csv")
            logger_valid.save(self.log_save_dir + config.global_start_time + "_valid.csv")

            global_epoch += 1

if __name__ == "__main__":
    np.set_printoptions(
        linewidth=150,
        formatter={'float_kind': lambda x: "%.4f" % x}
    )
    '''
    model = model.Spacial_Model(
        model_name="spacial_model",
        start_learning_rate=0.001,
        decay_steps=2e4,
        decay_rate=0.5,
    )
    ctl = Controller(model=model, base_epoch=0)
    '''
    model = model.Seq2Seq_Model(
        model_name="seq2seq_model",
        start_learning_rate=0.001,
        decay_steps=2e5,
        decay_rate=0.5,
    )
    ctl = Seq2Seq_Controller(model=model, base_epoch=-1)
    ctl.controller_train()
    # ctl.controller_test()
