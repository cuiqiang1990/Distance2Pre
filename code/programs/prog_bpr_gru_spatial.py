#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   纯RNN、GRU程序
# Create Date:  2017-05-19 12:00:00
# Modify Date:  2010-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
from collections import OrderedDict     # 按输入的顺序构建字典
import time
import datetime
import numpy as np
import os
import random
from public.BPR import OboBpr
from public.GRU import OboGru
from public.GRU_Spatial import OboDist2PreLinear, OboDist2PreNonLinear, softmax
from public.Global_Best import GlobalBest
from public.Load_Data_by_length import load_data, fun_data_buys_masks, cal_dis
from public.Load_Data_by_length import fun_random_neg_masks_tra, fun_random_neg_masks_tes, fun_compute_dist_neg
from public.Load_Data_by_length import fun_compute_distance, fun_acquire_prob
from public.Valuate import fun_predict_auc_recall_map_ndcg, fun_save_best_and_losses, fun_predict_auc_recall_map_ndcg_dist
__docformat__ = 'restructedtext en'

WHOLE = './poidata/'
PATH_f = os.path.join(WHOLE, './Foursquare')
PATH_g = os.path.join(WHOLE, './Gowalla')
PATH = PATH_f


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back
    return new_func


class Params(object):
    def __init__(self, p=None):
        """
        构建模型参数，加载数据
            把前90%分为8:1用作train和valid，来选择超参数, 不用去管剩下的10%.
            把前90%作为train，剩下的是test，把valid时学到的参数拿过来跑程序.
            valid和test部分，程序是一样的，区别在于送入的数据而已。
        :param p: 一个标示符，没啥用
        :return:
        """
        # 1. 建立各参数。要调整的地方都在 p 这了，其它函数都给写死。
        if not p:
            t = 't'                       # 写1就是valid, 写0就是test
            assert 't' == t or 'v' == t   # no other case
            p = OrderedDict(
                [
                    ('dataset',             'sub_users5_items5.txt'),
                    ('mode',                'test' if 't' == t else 'valid'),

                    ('split',               -1 if 't' == t else -2),   # test预测最后一个。
                    ('at_nums',             [5, 10, 15, 20]),
                    ('epochs',              100),    # 调程序用50，test时用100.

                    ('latent_size',         20),
                    ('alpha',               0.01),
                    ('lambda',              0.001),     # dist2pre: 两个数据集都用0.001，并且用SGD。
                    ('loss_weight',         [0.5, 0.5]),

                    # foursquare: 已是最佳值
                    ('dd',                  150 / 1000.0),    # 150m
                    ('UD',                  5),    # 截断距离5km，lambda_s的维度。
                    # gowalla: 已是最佳值
                    # ('dd',                  100 / 1000.0),    # 200m
                    # ('UD',                  20),    # 截断距离40km，lambda_s的维度。

                    ('mini_batch',          0),     # 0:one_by_one, 1:mini_batch. 全都用逐条。
                    ('gru',                 3),     # 0:bpr, 1:gru, 2:dist2pre-linear
                                                    # 3:dist2pre-nonlinear

                    ('batch_size_train',    1),     #
                    ('batch_size_test',     128),   # user * item 矩阵太大了，分成多次计算。 768
                ])
            for i in p.items():
                print(i)
        dist_num = int(p['UD'] / p['dd'])    # 1520 = 38*1000/25。idx=[0, 1519+1]

        # 2. 加载数据
        # 因为train/set里每项的长度不等，无法转换为完全的(n, m)矩阵样式，所以shared会报错.
        [(user_num, item_num), pois_cordis, (tra_buys, tes_buys), (tra_dist, tes_dist)] = \
            load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'], p['dd'], dist_num)
        # 正样本加masks
        tra_buys_masks, tra_masks = fun_data_buys_masks(tra_buys, tail=[item_num])          # 预测时算用户表达用
        tes_buys_masks, tes_masks = fun_data_buys_masks(tes_buys, tail=[item_num])          # 预测时用
        tra_dist_masks, _ = fun_data_buys_masks(tra_dist, tail=[dist_num])
        tes_dist_masks, _ = fun_data_buys_masks(tes_dist, tail=[dist_num])
        # 负样本加masks
        tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)   # 训练时用（逐条、mini-batch均可）
        tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)   # 预测时用
        # 计算负样本与上一个正样本的距离间隔，并加masks
        tra_dist_neg_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, p['dd'], dist_num)
        # 每个user训练序列里最后一个poi和all pois的距离落在哪个区间里。
        usrs_last_poi_to_all_intervals = fun_compute_distance(tra_buys_masks, tra_masks, pois_cordis, p['dd'], dist_num)

        # print(tra_dist[0][:5])        # [1520, 274, 0, 428, 142], 38km/25m=1520
        # print(tra_dist_masks[1020][:sum(tra_masks[1020])])

        # 3. 创建类变量
        self.p = p
        self.user_num, self.item_num, self.dist_num = user_num, item_num, dist_num
        self.pois_cordis = pois_cordis
        self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks = tra_buys_masks, tra_masks, tra_buys_neg_masks
        self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks = tes_buys_masks, tes_masks, tes_buys_neg_masks
        self.tra_dist_masks = tra_dist_masks
        self.tes_dist_masks = tes_dist_masks
        self.tra_dist_neg_masks = tra_dist_neg_masks
        self.ulptai = usrs_last_poi_to_all_intervals

    def build_model_one_by_one(self, flag=0):
        """
        建立模型对象
        :param flag: 参数变量、数据
        :return:
        """
        print('Building the model one_by_one ...')      # mask只是test计算用户表达时用。
        p = self.p
        size = p['latent_size']
        if 0 == flag:
            model = OboBpr(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_in=size,
                n_hidden=size)
        elif 1 == flag:
            model = OboGru(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_in=size,
                n_hidden=size)
        elif 2 == flag:
            model = OboDist2PreLinear(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                dist= [self.tra_dist_masks, self.tes_dist_masks, self.tra_dist_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_dists=[self.dist_num, p['dd']],
                n_in=size,
                n_hidden=size)
        else:
            model = OboDist2PreNonLinear(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                dist= [self.tra_dist_masks, self.tes_dist_masks, self.tra_dist_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_dists=[self.dist_num, p['dd']],
                n_in=size,
                n_hidden=size)
        model_name = model.__class__.__name__
        print('\t the current Class name is: {val}'.format(val=model_name))
        return model, model_name

    def compute_start_end(self, flag):
        """
        获取mini-batch的各个start_end(np.array类型，一组连续的数值)
        :param flag: 'train', 'test'
        :return: 各个start_end组成的list
        """
        assert flag in ['train', 'test', 'test_auc']
        if 'train' == flag:
            size = self.p['batch_size_train']
        elif 'test' == flag:
            size = self.p['batch_size_test']        # test: top-k and acquire user vector
        else:
            size = self.p['batch_size_test'] * 10   # test: auc
        user_num = self.user_num
        rest = (user_num % size) > 0   # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)   # 限制标号索引不能超过user_num
            start_end = np.arange(start, end, dtype=np.int32)
            starts_ends.append(start_end)
        return batch_idxs, starts_ends


def train_valid_or_test():
    """
    主程序
    :return:
    """
    # 建立参数、数据、模型、模型最佳值
    pas = Params()
    p = pas.p
    model, model_name = pas.build_model_one_by_one(flag=p['gru'])
    best = GlobalBest(at_nums=p['at_nums'])   # 存放最优数据
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    # 直接取出来部分变量，后边就不用加'pas.'了。
    user_num, item_num, dist_num = pas.user_num, pas.item_num, pas.dist_num
    tra_buys_masks, tra_masks, tra_buys_neg_masks = pas.tra_buys_masks, pas.tra_masks, pas.tra_buys_neg_masks
    tes_buys_masks, tes_masks, tes_buys_neg_masks = pas.tes_buys_masks, pas.tes_masks, pas.tes_buys_neg_masks
    dd = p['dd']
    pois_cordis = pas.pois_cordis
    ulptai = pas.ulptai
    del pas

    # 主循环
    losses = []
    wds = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        # 每次epoch，都要重新选择负样本。都要把数据打乱重排，这样会以随机方式选择样本计算梯度，可得到精确结果
        if epoch > 0:       # epoch=0的负样本已在循环前生成，且已用于类的初始化
            tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)
            tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)
            if p['gru'] in [0, 1]:
                model.update_neg_masks(tra_buys_neg_masks, tes_buys_neg_masks)
            else:
                tra_dist_neg_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, dd, dist_num)
                model.s_update_neg_masks(tra_buys_neg_masks, tes_buys_neg_masks, tra_dist_neg_masks)

        # ----------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)       # 每个epoch都打乱user_id输入顺序
        if 0 == p['gru']:
            for uidx in user_idxs_tra:
                tra = tra_buys_masks[uidx]
                neg = tra_buys_neg_masks[uidx]
                for i in np.arange(sum(tra_masks[uidx])):
                    loss += model.train(uidx, [tra[i], neg[i]])
        else:
            for uidx in user_idxs_tra:
                loss += model.train(uidx)

        rnn_l2_sqr = model.l2.eval()            # model.l2是'TensorVariable'，无法直接显示其值
        # 把loss及loss_weight保存下来.
        wd = model.wd.eval()
        print('\t\twd = {v1}'.format(v1=wd))
        print('\t\tsum_loss = {val} = {v1} + {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('{v1}'.format(v1=int(loss + rnn_l2_sqr)))
        wds.append('{v1}'.format(v1=wd))
        t1 = time.time()
        times0.append(t1 - t0)

        # ----------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        # 计算：所有用户、商品的表达
        if 0 == p['gru']:
            model.update_trained_items()
            model.update_trained_users()
        elif 1 == p['gru']:
            model.update_trained_items()    # 对于MV-GRU，这里会先算出来图文融合特征。
            all_hus = []
            for start_end in starts_ends_tes:
                sub_all_hus = model.predict(start_end)
                all_hus.extend(sub_all_hus)
            model.update_trained_users(all_hus)
        else:
            model.update_trained_items()
            model.update_trained_dists()
            all_hus = []
            all_sus = []
            for start_end in starts_ends_tes:
                [sub_all_hus, sub_all_sus] = model.predict(start_end)
                all_hus.extend(sub_all_hus)
                all_sus.extend(sub_all_sus)
            probs = fun_acquire_prob(all_sus, ulptai, dist_num)   # 输入shape=(user_num, dist_num), (user_num, item_num)
            model.update_trained_users(all_hus)
            model.update_prob(probs)
        t2 = time.time()
        times1.append(t2 - t1)

        # 计算各种指标，并输出当前最优值。
        fun_predict_auc_recall_map_ndcg(
            p, model, best, epoch, starts_ends_auc, starts_ends_tes, tes_buys_masks, tes_masks)
        best.fun_print_best(epoch)                  # 每次都只输出当前最优的结果
        t3 = time.time()
        times2.append(t3-t2)
        print('\tavg. time (train, user, test): %0.0fs,' % np.average(times0),
              '%0.0fs,' % np.average(times1), '%0.0fs' % np.average(times2),
              '| alpha, lam: {v1}'.format(v1=', '.join([str(lam) for lam in [p['alpha'], p['lambda']]])),
              '| model: {v1}'.format(v1=model_name))

        # ----------------------------------------------------------------------------------------------------------
        if epoch in [p['epochs'] - 1, 99]:
            # 保存最优值、所有的损失值。
            print("\tBest and losses saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_best_and_losses', PATH.split('/')[-2])
            fun_save_best_and_losses(path, model_name, epoch, p, best, [losses, wds])

    for i in p.items():
        print(i)
    print('\t the current Class name is: {val}'.format(val=model_name))


@exe_time
def main():
    train_valid_or_test()


if '__main__' == __name__:
    main()
