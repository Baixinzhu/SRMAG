from modules import *

from recurrent_GSNP import *
class Model():
    def __init__(self, itemnum, args, maxlen_long, maxlen_short, reuse=None):
        tf.compat.v1.disable_eager_execution()
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        self.long_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_long))
        self.long_pos = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_long))
        self.long_neg = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_long))

        self.short_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_short))
        self.short_pos = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_short))
        self.short_neg = tf.compat.v1.placeholder(tf.int32, shape=(None, maxlen_short))

        long_pos = self.long_pos
        long_neg = self.long_neg
        short_pos = self.short_pos
        short_neg = self.short_neg


        mask_long = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.long_seq, 0)), -1)
        mask_short = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.short_seq, 0)), -1)
        mask_fusion_feature = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.long_seq, 0)), -1)
        mask_long_lstm = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.long_seq, 0)), -1)
        mask_short_lstm = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.short_seq, 0)), -1)
        # sequence embedding, item embedding table
        self.long, item_emb_table = embedding(self.long_seq, vocab_size=itemnum + 1, num_units=args.hidden_units,
                                              zero_pad=True, scale=True, l2_reg=args.l2_emb,
                                              scope="input_embeddings", with_t=True, reuse=reuse)

        self.short = tf.nn.embedding_lookup(item_emb_table, self.short_seq)

        with tf.compat.v1.variable_scope("lstm_long", reuse=reuse):
            lstm_snp = GSNPCell(units=32, dropout=0.2, return_sequences=True)
            self.long_lstm = lstm_snp(self.long)

            pad1 = self.long_lstm.shape[2] - mask_long_lstm.shape[2]
            pad2 = mask_long_lstm.shape[1] - self.long_lstm.shape[1]
            mask_long_lstm = tf.compat.v1.pad(mask_long_lstm, [[0, 0], [0, 0], [0, pad1]])
            self.long_lstm = tf.compat.v1.pad(self.long, [[0, 0], [0, pad2], [0, 0]])
            # Dropout
            self.long_lstm = tf.compat.v1.layers.dropout(self.long_lstm, rate=args.dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))

            pad5 = self.long_lstm.shape[2] - mask_long_lstm.shape[2]
            mask_long_lstm = tf.pad(mask_long_lstm, [[0, 0], [0, 0], [0, pad5]])
            self.long_lstm *= mask_long_lstm
            for i in range(1):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.long_lstm = multihead_attention(queries=self.long_lstm, keys=self.long_lstm,
                                                         num_units=args.hidden_units, num_heads=1,
                                                         dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                         causality=True, scope="self_attention")

                    # Feed forward
                    self.long_lstm = feedforward(normalize(self.long_lstm),
                                                 num_units=[args.hidden_units, args.hidden_units],
                                                 dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.long_lstm *= mask_long_lstm
        with tf.compat.v1.variable_scope("lstm_short", reuse=reuse):
            lstms_snp = GSNPCell(units=32, dropout=0.2, return_sequences=True)
            self.short_lstm = lstms_snp(self.short)

            pad_s = self.short_lstm.shape[2] - mask_short_lstm.shape[2]
            pad_ms = mask_short_lstm.shape[1] - self.short_lstm.shape[1]
            mask_short_lstm = tf.compat.v1.pad(mask_short_lstm, [[0, 0], [0, 0], [0, pad_s]])
            self.short_lstm = tf.compat.v1.pad(self.short_lstm, [[0, 0], [0, pad_ms], [0, 0]])
            # Dropout
            self.short_lstm = tf.compat.v1.layers.dropout(self.short_lstm, rate=args.dropout_rate,
                                                             training=tf.convert_to_tensor(self.is_training))

            self.short_lstm *= mask_short_lstm
            for i in range(1):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.short_lstm = multihead_attention(queries=self.short_lstm, keys=self.short_lstm,
                                                             num_units=args.hidden_units, num_heads=1,
                                                             dropout_rate=args.dropout_rate,
                                                             is_training=self.is_training,
                                                             causality=True, scope="self_attention")

                    # Feed forward
                    self.short_lstm = feedforward(normalize(self.short_lstm),
                                                     num_units=[args.hidden_units, args.hidden_units],
                                                     dropout_rate=args.dropout_rate, is_training=self.is_training)

                    pad6 = self.short_lstm.shape[2] - mask_short_lstm.shape[2]
                    mask_short_lstm = tf.pad(mask_short_lstm, [[0, 0], [0, 0], [0, pad6]])
                    self.short_lstm *= mask_short_lstm

            self.short_lstm = normalize(self.short_lstm)
        with tf.compat.v1.variable_scope("SASRec_long", reuse=reuse):

            # Dropout
            self.long = tf.compat.v1.layers.dropout(self.long, rate=args.dropout_rate,
                                          training=tf.convert_to_tensor(self.is_training))
            self.long *= mask_long

            for i in range(args.num_lblocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.long = multihead_attention(queries=self.long, keys=self.long,
                                                    num_units=args.hidden_units, num_heads=args.num_lheads,
                                                    dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                    causality=True, scope="self_attention")

                    # Feed forward
                    self.long = feedforward(normalize(self.long), num_units=[args.hidden_units, args.hidden_units],
                                            dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.long *= mask_long

            self.long = normalize(self.long)

        with tf.compat.v1.variable_scope("SASRec_short", reuse=reuse):

            # Dropout
            self.short = tf.compat.v1.layers.dropout(self.short, rate=args.dropout_rate,
                                           training=tf.convert_to_tensor(self.is_training))
            self.short *= mask_short

            for i in range(args.num_sblocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.short = multihead_attention(queries=self.short, keys=self.short,
                                                     num_units=args.hidden_units, num_heads=args.num_sheads,
                                                     dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                     causality=True, scope="self_attention")

                    # Feed forward
                    self.short = feedforward(normalize(self.short), num_units=[args.hidden_units, args.hidden_units],
                                             dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.short *= mask_short

            self.short = normalize(self.short)
        print('self-long = ', self.long)
        print('self-short = ', self.short)
        print('self-long_lstm=', self.long_lstm)
        print('self.short_lstm',self.short_lstm)


        long_pos = tf.reshape(long_pos, [tf.shape(self.long_seq)[0] * maxlen_long])
        long_neg = tf.reshape(long_neg, [tf.shape(self.long_seq)[0] * maxlen_long])
        lpos_emb = tf.nn.embedding_lookup(item_emb_table, long_pos)
        lneg_emb = tf.nn.embedding_lookup(item_emb_table, long_neg)

        short_pos = tf.reshape(short_pos, [tf.shape(self.short_seq)[0] * maxlen_short])
        short_neg = tf.reshape(short_neg, [tf.shape(self.short_seq)[0] * maxlen_short])
        spos_emb = tf.nn.embedding_lookup(item_emb_table, short_pos)
        sneg_emb = tf.nn.embedding_lookup(item_emb_table, short_neg)

        lseq_emb = tf.reshape(self.long, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])
        sseq_emb = tf.reshape(self.short, [tf.shape(self.short_seq)[0] * maxlen_short, args.hidden_units])


        expand = tf.zeros([tf.shape(self.long_seq)[0], (maxlen_long - maxlen_short),
                           args.hidden_units], dtype=tf.float32)
        expand_semb = tf.concat([expand, self.short], axis=1)
        expand_semb_lstm = tf.concat([expand,self.short_lstm],axis=1)
        print('expand_current_emb = ', expand_semb)
        l_expand = tf.expand_dims(self.long, axis=-1)
        s_expand = tf.expand_dims(expand_semb, axis=-1)
        lstm_expand = tf.expand_dims(self.long_lstm, axis=-1)
        lstms_expand = tf.expand_dims(expand_semb_lstm,axis=-1)
        print('l_expand = ', l_expand)
        print('s_expand = ', s_expand)
        print('lstm_expand=', lstm_expand)
        print('lstms_expand=',lstms_expand)

        seql_emb=tf.reduce_sum(tf.concat([l_expand,lstm_expand], axis=-1), axis=-1)
        seqs_emb=tf.reduce_sum(tf.concat([s_expand,lstms_expand], axis=-1), axis=-1)
        # seq_emb = tf.reduce_sum(tf.concat([l_expand, s_expand, lstm_expand,lstms_expand], axis=-1), axis=-1)
        # seq_emb = tf.reshape(seq_emb, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])
        # self.fusion_feature = tf.reshape(seq_emb, [tf.shape(self.long_seq)[0] * maxlen_long, maxlen_long, args.hidden_units])
        self.lfusion_feature = seql_emb
        self.sfusion_feature = seqs_emb

        for i in range(1):
            with tf.compat.v1.variable_scope("Dy_Attention_long_%d" % i):
            # with tf.compat.v1.variable_scope("Dy_Attention_long", reuse=reuse):
                # Dropout
                self.lfusion_feature = tf.compat.v1.layers.dropout(self.lfusion_feature, rate=args.dropout_rate,
                                              training=tf.convert_to_tensor(self.is_training))
                self.lfusion_feature *= mask_fusion_feature
                # Self-attention
                self.lfusion_feature = multihead_attention(queries=self.lfusion_feature, keys=self.lfusion_feature,
                                                num_units=args.hidden_units, num_heads=1,
                                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                causality=True, scope="self_attention")

                # Feed forward
                self.lfusion_feature = feedforward(normalize(self.lfusion_feature), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.lfusion_feature *= mask_long

            self.lfusion_feature = normalize(self.lfusion_feature)

            with tf.compat.v1.variable_scope("Dy_Attention_sfusion_%d"% i):
                # Dropout
                self.sfusion_feature = tf.compat.v1.layers.dropout(self.sfusion_feature, rate=args.dropout_rate,
                                                        training=tf.convert_to_tensor(self.is_training))
                self.sfusion_feature *= mask_fusion_feature

                # Self-attention
                self.sfusion_feature = multihead_attention(queries=self.sfusion_feature, keys=self.sfusion_feature,
                                                 num_units=args.hidden_units, num_heads=1,
                                                 dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                 causality=True, scope="self_attention")

                # Feed forward
                self.sfusion_feature = feedforward(normalize(self.sfusion_feature),
                                                  num_units=[args.hidden_units, args.hidden_units],
                                                  dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.sfusion_feature *= mask_fusion_feature

            self.sfusion_feature = normalize(self.sfusion_feature)


            # expand_femb = tf.concat([expand, self.fusion_feature], axis=1)
            # print('expand_current_emb = ', expand_femb)
            l_expand = tf.expand_dims(self.lfusion_feature, axis=-1)
            f_expand = tf.expand_dims(self.sfusion_feature, axis=-1)
            print('l_expand = ', l_expand)
            print('f_expand = ', f_expand)

            seq_femb = tf.reduce_sum(tf.concat([l_expand, f_expand], axis=-1), axis=-1)
            # seq_femb = tf.reshape(seq_femb, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])
            self.fusion_feature = seq_femb

        seq_femb = tf.reshape(self.fusion_feature, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])

        self.test_logits = tf.matmul(seq_femb, tf.transpose(item_emb_table[1:]))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.long_seq)[0], maxlen_long, itemnum])


        self.test_logits = self.test_logits[:, -1, :]
        self.top_value, self.top_index = tf.nn.top_k(self.test_logits, k=itemnum, sorted=True)

        # prediction layer
        self.lpos_logits = tf.reduce_sum(lpos_emb * lseq_emb, -1)
        self.lneg_logits = tf.reduce_sum(lneg_emb * lseq_emb, -1)
        self.spos_logits = tf.reduce_sum(spos_emb * sseq_emb, -1)
        self.sneg_logits = tf.reduce_sum(sneg_emb * sseq_emb, -1)

        # ignore padding items (0)
        istarget_long = tf.reshape(tf.compat.v1.to_float(tf.not_equal(long_pos, 0)), [tf.shape(self.long_seq)[0] * maxlen_long])
        istarget_short = tf.reshape(tf.compat.v1.to_float(tf.not_equal(short_pos, 0)), [tf.shape(self.short_seq)[0] * maxlen_short])

        self.loss = tf.reduce_mean(tf.compat.v1.log(1 + tf.exp(self.sneg_logits - self.spos_logits))) \
                    + tf.reduce_mean(tf.compat.v1.log(1 + tf.exp(self.lneg_logits - self.lpos_logits)))
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.lpos_logits - self.lneg_logits) + 1) / 2) * istarget_long
        ) / tf.reduce_sum(istarget_long)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged =tf.compat.v1.summary.merge_all()

    def predict(self, sess, long_seq, short_seq):
        return sess.run(self.top_index,
                        {self.long_seq: long_seq, self.short_seq: short_seq, self.is_training: False})
