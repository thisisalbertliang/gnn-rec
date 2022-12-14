# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.keras import initializers, layers
import time
import os
import sys
import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.python_utils import get_top_k_scored_items

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x

class GraphAttentionLayer(layers.Layer):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = self.add_weight(
            shape=(in_features, out_features),
            initializer=initializers.GlorotUniform(),
            name="attn_W"
        )
        self.a = self.add_weight(
            shape=(2 * out_features, 1),
            initializer=initializers.GlorotUniform(),
            name="attn_a"
        )

        self.leakyrelu = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, h, adj):
        Wh = tf.matmul(h, self.W)
        Wh1 = tf.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = tf.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + tf.transpose(Wh2) # broadcast add
        e = self.leakyrelu(e)

        zero_vec = tf.ones_like(e) * -9e15
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention)
        attention = tf.nn.dropout(attention, self.dropout)
        h_prime = tf.matmul(attention, Wh)

        return tf.nn.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SelfAttention(layers.Layer):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.WQ = self.add_weight(
            shape=(embedding_size, embedding_size),
            initializer=initializers.GlorotUniform(),
            name="self_attn_queries"
        )
        self.WK = self.add_weight(
            shape=(embedding_size, embedding_size),
            initializer=initializers.GlorotUniform(),
            name="self_attn_keys"
        )
        self.attn = layers.Attention()

    def call(self, embs): # embs.shape == (2625, 4, 64)
        queries = tf.matmul(embs, self.WQ)
        keys = tf.matmul(embs, self.WK)
        attn_res = self.attn([queries, embs])
        return attn_res


class GNN(object):
    """GNN model
    """

    def __init__(self, params, data, seed=None):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function.

        Args:
            params (dict): A dict, hold the entire set of hyperparameters.
            data (object): A recommenders.models.deeprec.DataModel.ImplicitCF object, load and process data.
            seed (int): Seed.

        """

        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.data = data
        self.epochs = params.epochs
        self.lr = params.learning_rate
        self.emb_dim = params.embed_size
        self.batch_size = params.batch_size
        self.n_layers = params.n_layers
        self.decay = params.decay
        self.eval_epoch = params.eval_epoch
        self.top_k = params.top_k
        self.save_model = params.save_model
        self.save_epoch = params.save_epoch
        self.metrics = params.metrics
        self.model_dir = params.model_dir
        self.params = params

        self.neighbor_aggregator = params.neighbor_aggregator
        self.info_updater = params.info_updater
        self.final_node_repr = params.final_node_repr
        assert self.neighbor_aggregator in ("degree_norm", "attention")
        assert self.info_updater in ("direct", "single_linear", "multi_linear")
        assert self.final_node_repr in ("mean", "concat", "weighted", "attention")

        metric_options = ["map", "ndcg", "precision", "recall"]
        for metric in self.metrics:
            if metric not in metric_options:
                raise ValueError(
                    "Wrong metric(s), please select one of this list: {}".format(
                        metric_options
                    )
                )

        self.norm_adj = data.get_norm_adj_mat()

        self.n_users = data.n_users
        self.n_items = data.n_items

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.weights, self.alphas = self._init_weights()
        self.self_attn = SelfAttention(self.emb_dim)
        self.info_updater = self._init_info_updater()

        self.gat = GraphAttentionLayer(self.emb_dim, self.emb_dim, dropout=0, alpha=0.2)

        self.ua_embeddings, self.ia_embeddings = self._create_embeddings()

        self.u_g_embeddings = tf.nn.embedding_lookup(
            params=self.ua_embeddings, ids=self.users
        )
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(
            params=self.ia_embeddings, ids=self.pos_items
        )
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(
            params=self.ia_embeddings, ids=self.neg_items
        )
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["user_embedding"], ids=self.users
        )
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["item_embedding"], ids=self.pos_items
        )
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["item_embedding"], ids=self.neg_items
        )

        self.batch_ratings = tf.matmul(
            self.u_g_embeddings,
            self.pos_i_g_embeddings,
            transpose_a=False,
            transpose_b=True,
        )

        self.mf_loss, self.emb_loss = self._create_bpr_loss(
            self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings
        )
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss
        )
        self.saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_weights(self):
        """Initialize user and item embeddings.

        Returns:
            dict: With keys `user_embedding` and `item_embedding`, embeddings of all users and items.

        """
        all_weights = dict()
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        all_weights["user_embedding"] = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name="user_embedding"
        )
        all_weights["item_embedding"] = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name="item_embedding"
        )
        alphas = tf.Variable(
            initializer([self.n_layers+1]), name="alphas"
        )
        print("Using xavier initialization.")

        return all_weights, alphas
    
    def _init_info_updater(self):
        if self.info_updater == "multi_linear":
            return [
                tf.compat.v1.layers.Dense(
                    units=self.emb_dim, activation=tf.nn.leaky_relu, use_bias=True
                )
                for _ in range(self.n_layers + 1)
            ]
        elif self.info_updater == "single_linear":
            return (self.n_layers + 1) * [tf.compat.v1.layers.Dense(
                units=self.emb_dim, activation=tf.nn.leaky_relu, use_bias=True
            )]
        else:
            return [
                tf.identity
                for _ in range(self.n_layers + 1)
            ]

    def _create_embeddings(self):
        """Calculate the average embeddings of users and items after every layer of the model.

        Returns:
            tf.Tensor, tf.Tensor: Average user embeddings. Average item embeddings.

        """
        A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj) # (2625, 2625)
        if self.neighbor_aggregator == "attention":
            A_hat = tf.sparse.to_dense(A_hat) # attention with sparse currently unimplemented

        ego_embeddings = tf.concat(
            [self.weights["user_embedding"], self.weights["item_embedding"]], axis=0
        )
        ego_embeddings = self.info_updater[0](ego_embeddings)
        all_embeddings = [ego_embeddings]

        for k in range(1, self.n_layers + 1):
            if self.neighbor_aggregator == "attention":
                ego_embeddings = self.gat(ego_embeddings, A_hat) # (2625, 64)
            elif self.neighbor_aggregator == "degree_norm":
                ego_embeddings = tf.sparse.sparse_dense_matmul(A_hat, ego_embeddings) # (2625, 64)
            else:
                raise NotImplementedError()

            ego_embeddings = self.info_updater[k](ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1) # (2625, 4, 64)
        # print(all_embeddings.shape, self.n_layers)

        if self.final_node_repr == "mean":
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1)
        elif self.final_node_repr == "weighted":
            all_embeddings = tf.einsum("nkd,k->nd", all_embeddings, tf.nn.softmax(self.alphas))
        elif self.final_node_repr == "concat":
            all_embeddings = tf.reshape(all_embeddings, shape=[-1, (self.n_layers+1)*self.emb_dim])
        elif self.final_node_repr == "attention":
            all_embeddings = self.self_attn(all_embeddings)[:,0,:]

        u_g_embeddings, i_g_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], 0
        )
        return u_g_embeddings, i_g_embeddings

    def _create_bpr_loss(self, users, pos_items, neg_items):
        """Calculate BPR loss.

        Args:
            users (tf.Tensor): User embeddings to calculate loss.
            pos_items (tf.Tensor): Positive item embeddings to calculate loss.
            neg_items (tf.Tensor): Negative item embeddings to calculate loss.

        Returns:
            tf.Tensor, tf.Tensor: Matrix factorization loss. Embedding regularization loss.

        """
        pos_scores = tf.reduce_sum(input_tensor=tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(input_tensor=tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(self.u_g_embeddings_pre)
            + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)
            + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        )
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_mean(
            input_tensor=tf.nn.softplus(-(pos_scores - neg_scores))
        )
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert a scipy sparse matrix to tf.SparseTensor.

        Returns:
            tf.SparseTensor: SparseTensor after conversion.

        """
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def fit(self):
        """Fit the model on self.data.train. If eval_epoch is not -1, evaluate the model on `self.data.test`
        every `eval_epoch` epoch to observe the training status.

        """
        for epoch in range(1, self.epochs + 1):
            train_start = time.time()
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.opt, self.loss, self.mf_loss, self.emb_loss],
                    feed_dict={
                        self.users: users,
                        self.pos_items: pos_items,
                        self.neg_items: neg_items,
                    },
                )
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            if np.isnan(loss):
                print("ERROR: loss is nan.")
                sys.exit()
            train_end = time.time()
            train_time = train_end - train_start

            if self.save_model and epoch % self.save_epoch == 0:
                save_path_str = os.path.join(self.model_dir, self.params.dataset_size, "checkpoints", self.params.model_name, "epoch_" + str(epoch))
                if not os.path.exists(os.path.dirname(save_path_str)):
                    os.makedirs(os.path.dirname(save_path_str))
                checkpoint_path = self.saver.save(  # noqa: F841
                    sess=self.sess, save_path=save_path_str
                )
                print("Saved model to path {0}".format(os.path.abspath(save_path_str)))

            if self.eval_epoch == -1 or epoch % self.eval_epoch != 0:
                print(
                    "Epoch %d (train)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f"
                    % (epoch, train_time, loss, mf_loss, emb_loss)
                )
            else:
                eval_start = time.time()
                ret = self.run_eval()
                eval_end = time.time()
                eval_time = eval_end - eval_start

                print(
                    "Epoch %d (train)%.1fs + (eval)%.1fs: train loss = %.5f = (mf)%.5f + (embed)%.5f, %s"
                    % (
                        epoch,
                        train_time,
                        eval_time,
                        loss,
                        mf_loss,
                        emb_loss,
                        ", ".join(
                            metric + " = %.5f" % (r)
                            for metric, r in zip(self.metrics, ret)
                        ),
                    )
                )

    def load(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: Model path.

        Raises:
            IOError: if the restore operation failed.

        """
        try:
            self.saver.restore(self.sess, model_path)
        except Exception:
            raise IOError(
                "Failed to find any matching files for {0}".format(model_path)
            )

    def run_eval(self):
        """Run evaluation on self.data.test.

        Returns:
            dict: Results of all metrics in `self.metrics`.
        """
        topk_scores = self.recommend_k_items(
            self.data.test, top_k=self.top_k, use_id=True
        )
        ret = []
        for metric in self.metrics:
            if metric == "map":
                ret.append(
                    map_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "ndcg":
                ret.append(
                    ndcg_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "precision":
                ret.append(
                    precision_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "recall":
                ret.append(
                    recall_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
        return ret

    def score(self, user_ids, remove_seen=True):
        """Score all items for test users.

        Args:
            user_ids (np.array): Users to test.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            numpy.ndarray: Value of interest of all items for the users.

        """
        if any(np.isnan(user_ids)):
            raise ValueError(
                "LightGCN cannot score users that are not in the training set"
            )
        u_batch_size = self.batch_size
        n_user_batchs = len(user_ids) // u_batch_size + 1
        test_scores = []
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start:end]
            item_batch = range(self.data.n_items)
            rate_batch = self.sess.run(
                self.batch_ratings, {self.users: user_batch, self.pos_items: item_batch}
            )
            test_scores.append(np.array(rate_batch))
        test_scores = np.concatenate(test_scores, axis=0)
        if remove_seen:
            test_scores += self.data.R.tocsr()[user_ids, :] * -np.inf
        return test_scores

    def recommend_k_items(
        self, test, top_k=10, sort_top_k=True, remove_seen=True, use_id=False
    ):
        """Recommend top K items for all users in the test set.

        Args:
            test (pandas.DataFrame): Test data.
            top_k (int): Number of top items to recommend.
            sort_top_k (bool): Flag to sort top k results.
            remove_seen (bool): Flag to remove items seen in training from recommendation.

        Returns:
            pandas.DataFrame: Top k recommendation items for each user.

        """
        data = self.data
        if not use_id:
            user_ids = np.array([data.user2id[x] for x in test[data.col_user].unique()])
        else:
            user_ids = np.array(test[data.col_user].unique())

        test_scores = self.score(user_ids, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                data.col_user: np.repeat(
                    test[data.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                data.col_item: top_items.flatten()
                if use_id
                else [data.id2item[item] for item in top_items.flatten()],
                data.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()

    def output_embeddings(self, idmapper, n, target, user_file):
        embeddings = list(target.eval(session=self.sess))
        with open(user_file, "w") as wt:
            for i in range(n):
                wt.write(
                    "{0}\t{1}\n".format(
                        idmapper[i], " ".join([str(a) for a in embeddings[i]])
                    )
                )

    def infer_embedding(self, user_file, item_file):
        """Export user and item embeddings to csv files.

        Args:
            user_file (str): Path of file to save user embeddings.
            item_file (str): Path of file to save item embeddings.

        """
        # create output directories if they do not exist
        dirs, _ = os.path.split(user_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        dirs, _ = os.path.split(item_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        data = self.data

        self.output_embeddings(
            data.id2user, self.n_users, self.ua_embeddings, user_file
        )
        self.output_embeddings(
            data.id2item, self.n_items, self.ia_embeddings, item_file
        )
