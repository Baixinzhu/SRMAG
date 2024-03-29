import time
import tensorflow as tf

from SRMAG_model import Model

from six.moves import cPickle

import argparse
from util import *
import csv

if __name__ == '__main__':


    tf.compat.v1.reset_default_graph()
    parser = argparse.ArgumentParser()
    # 调整的参数
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--hidden_units', default=100, type=int)

    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    # 数据库的绝对路径
    parser.add_argument('--root', default='data/')
    parser.add_argument('--dataset', default='gowalla')

    parser.add_argument('--maxlen_l', default=100, type=int)
    parser.add_argument('--maxlen_s', default=10, type=int)
    # 训练循环次数
    parser.add_argument('--num_epochs', default=201, type=int)
    # 注意力网络层数和头数
    parser.add_argument('--num_lblocks', default=2, type=int)
    parser.add_argument('--num_sblocks', default=2, type=int)
    parser.add_argument('--num_lheads', default=2, type=int)
    parser.add_argument('--num_sheads', default=2, type=int)

    args = parser.parse_args()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)

    # gawalla
    item_num = 11364
    # Tmall
    # item_num = 48804
    # movielens
    # item_num = 7287
    model = Model(item_num, args, args.maxlen_l, args.maxlen_s)
    sess.run(tf.compat.v1.global_variables_initializer())

    T = 0.0
    t0 = time.time()
    with open(args.root + args.dataset + '/train.pkl', 'rb') as tf1:
        (train_users, train_long_seq, train_short_seq, train_long_neg, train_short_neg) = cPickle.load(
            tf1)

    num_batch = int(len(train_long_seq) / args.batch_size)

    with open(args.root + args.dataset + '/test.pkl', 'rb') as tf2:
        test_user, test_long_seq, test_short_seq, test_item = cPickle.load(tf2)



        for epoch in range(1, args.num_epochs + 1):
            print('training--------------', epoch)
            sum_loss = 0
            for b in range(num_batch + 1):
                if b < num_batch:
                    low = b * args.batch_size
                    high = (b + 1) * args.batch_size
                else:
                    low = b * args.batch_size
                    high = len(train_long_seq)

                inputs_long = train_long_seq[low:high]
                inputs_current = train_short_seq[low:high]

                long_seq = [seq[:-1] for seq in inputs_long]
                long_pos = [seq[1:] for seq in inputs_long]
                long_neg = train_long_neg[low: high]

                short_seq = [seq[:-1] for seq in inputs_current]
                short_pos = [seq[1:] for seq in inputs_current]
                short_neg = train_short_neg[low:high]

                long_seq = zero_padding_pre(long_seq, args.maxlen_l)
                long_pos = zero_padding_pre(long_pos, args.maxlen_l)
                long_neg = zero_padding_pre(long_neg, args.maxlen_l)

                short_seq = zero_padding_pre(short_seq, args.maxlen_s)
                short_pos = zero_padding_pre(short_pos, args.maxlen_s)
                short_neg = zero_padding_pre(short_neg, args.maxlen_s)

                loss, _ = sess.run([model.loss, model.train_op],
                                   {model.long_seq: long_seq, model.long_pos: long_pos,
                                    model.long_neg: long_neg,
                                    model.short_seq: short_seq, model.short_pos: short_pos,
                                    model.short_neg: short_neg,
                                    model.is_training: True})
                sum_loss += loss

            if epoch >= 20:
                t1 = time.time() - t0
                T += t1
                print('Evaluating------------', epoch)
                top_K = [10, 20]
                hit_result = {}
                ndcg_result = {}
                apks = []
                for k in top_K:
                    hit_result[k] = []
                    ndcg_result[k] = []
                for i in range(len(test_user)):
                    inputs = test_long_seq[i]
                    current = test_short_seq[i]
                    real_item = [test_item[i] - 1]
                    inputs = zero_padding_pre([inputs], args.maxlen_l)
                    current = zero_padding_pre([current], args.maxlen_s)

                    top_index = model.predict(sess, inputs, current)
                    apks.append(compute_apk(real_item, top_index[0], k=np.inf))

                    for k in top_K:
                        prec, rec, ndcg = precision_k(top_index[0], real_item, k)
                        hit_result[k].append(rec)
                        ndcg_result[k].append(ndcg)
                HR = []
                NDCG = []
                for k in top_K:
                    HR.append(str(np.mean(hit_result[k])))
                    NDCG.append(str(np.mean(ndcg_result[k])))
                    print('HR@' + str(k) + ' = ' + str(np.mean(hit_result[k])) + ',  ' +
                          'NDCG@' + str(k) + ' = ' + str(np.mean(ndcg_result[k])))
                print('MAP' + ' = ' + str(np.mean(apks)))


                t0 = time.time()

        print("Done")




