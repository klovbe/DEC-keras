
import os
# os.environ["LD_LIBRARY_PATH"] = "/home/xysmlx/anaconda3/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:" \
#                                 "/usr/local/cuda/lib64/libcublas.so.8.0"

print(os.environ.get("LD_LIBRARY_PATH"))

from time import time
import numpy as np
import pandas as pd
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Lambda, Flatten, Activation, normalization, Add
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
from DEC import ClusteringLayer
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], activation='linear', kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

def autoencoder_bn(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = normalization.BatchNormalization()(h)
        h = Activation(activation=act)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], kernel_initializer=init, name='decoder_%d' % i)(y)
        y = normalization.BatchNormalization()(y)
        y = Activation(activation=act)(y)

    # output
    y = Dense(dims[0], activation='linear', kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

class s_IDEC(object):
    def __init__(self,
                 dims,
                 gamma=0.1,
                 n_clusters=10,
                 alpha=1.0,
                 bn = False,
                 init='glorot_uniform'):

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        if bn:
            self.autoencoder, self.encoder = autoencoder_bn(self.dims, init=init)
        else:
            self.autoencoder, self.encoder = autoencoder(self.dims, init=init)
        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        # reconstruct = Add()(self.autoencoder.output)
        # clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(reconstruct)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        # self.model = Model(inputs=self.autoencoder.input,
        #                    outputs=self.autoencoder.output)
        # print(self.model)
    def pretrain(self, x, loss='mse', optimizer='adam', epochs=200, batch_size=256, gamma_s = 0.05):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: ', time() - t0)
        self.pretrained = True

    def compile(self, optimizer, loss, loss_weights):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x_train, x_val, x_test, model_name, outdir, df_columns, y = None, epoch=500,
            batch_size=256, update_interval=5, early_stopping=20, tol=0.01):

        print('Update interval', update_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        encoder_out = self.encoder.predict(x_train)
        y_pred = kmeans.fit_predict(encoder_out)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('kmeans : acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
            X_embedded = TSNE(n_components=2).fit_transform(encoder_out)
            plt.figure(figsize=(12, 10))
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
            plt.colorbar()
            plt.show()
        print(np.bincount(y_pred))

        # y_pred = kmeans.fit_predict(x_train)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # for ite in range(int(epoch)):
        #     if ite % update_interval == 0:
        #         q,_,_ = self.model.predict(x_train, verbose=0)
        #         p = self.target_distribution(q)  # update the auxiliary target distribution p
        #     y0 = np.zeros_like(x_train)
        #     self.model.fit(x=x_train, y=[p, y0, x_train], batch_size=batch_size)

        # Step 2: deep clustering
        index = 0
        index_array_train = np.arange(x_train.shape[0])
        index_array_val = np.arange(x_val.shape[0])
        cost_val = []
        cost_train = []
        for ite in range(int(epoch)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x_train, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                print("delta label:{}".format(delta_label))
                y_pred_last = np.copy(y_pred)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
                print(np.bincount(y_pred))
                if ite > update_interval and delta_label < tol:
                        # and np.mean(cost_val[-(early_stopping + 1):-1]) > \
                        # np.mean(cost_val[-(early_stopping*2 + 1):-(early_stopping + 1)])\
                        # and np.mean(cost_train[-(early_stopping + 1):-1]) < \
                        # np.mean(cost_train[-(early_stopping*2 + 1):-(early_stopping + 1)])\
                        # :
                    print("Early stopping...")
                    break

            # train on batch
            tot_train_loss = 0.
            tot_mse_loss = 0.
            tot_cluster_loss = 0.
            while True:
                if index == 0:
                    np.random.shuffle(index_array_train)
                idx = index_array_train[index * batch_size: min((index+1) * batch_size, x_train.shape[0])]
                # cluster_loss, sparse_loss, mse_loss = self.model.train_on_batch(x=x_train[idx], y=[p[idx], y0, x_train[idx]])
                loss, cluster_loss, mse_loss = self.model.train_on_batch(x=x_train[idx], y=[p[idx], x_train[idx]])
                index = index + 1 if (index + 2) * batch_size <= x_train.shape[0] else 0
                tot_train_loss += loss * len(idx)
                tot_cluster_loss += cluster_loss * len(idx)
                tot_mse_loss += mse_loss * len(idx)
                if index == 0:
                    break
            avg_train_loss = tot_train_loss / x_train.shape[0]
            avg_cluster_loss = tot_cluster_loss / x_train.shape[0]
            avg_mse_loss = tot_mse_loss / x_train.shape[0]
            print("epoch {}th train, train_loss :{:.6f}, cluster_loss: {:.6f}, mse_loss: {:.6f}\n".format(ite + 1,
                                                                                                 avg_train_loss, avg_cluster_loss,
                                                                                                 avg_mse_loss))
            cost_train.append(avg_train_loss)

            # tot_val_loss = 0.
            # tot_mse_loss = 0.
            # tot_cluster_loss = 0.
            # while True:
            #     if index == 0:
            #         np.random.shuffle(index_array_val)
            #     idx = index_array_val[index * batch_size: min((index+1) * batch_size, x_val.shape[0])]
            #     loss, cluster_loss, mse_loss = self.model.test_on_batch(x=x_val[idx], y=[p[idx], x_val[idx]])
            #     index = index + 1 if (index + 1) * batch_size <= x_val.shape[0] else 0
            #     tot_cluster_loss += cluster_loss *len(idx)
            #     tot_mse_loss += mse_loss *len(idx)
            #     tot_val_loss += loss * len(idx)
            #     if index==0:
            #         break
            # avg_val_loss = tot_val_loss / x_val.shape[0]
            # avg_cluster_loss = tot_cluster_loss / x_val.shape[0]
            # avg_mse_loss = tot_mse_loss / x_val.shape[0]
            # print("epoch {}th validate, loss: {:.6f}, cluster_loss: {:.6f}, mse_loss: {:.6f}\n".format(ite + 1,
            #                                                                                      avg_val_loss, avg_cluster_loss,
            #                                                                                      avg_mse_loss))
            # cost_val.append(avg_val_loss

        print('training time: ', time() - t1)
        # save the trained model

        print("saving predict data...")
        encoder_out = self.encoder.predict(x_test)
        q, decoder_out= self.model.predict(x_test)
        y_pred = q.argmax(1)
        if y is not None:
            print("orginal cluster proportion: {}".format(np.bincount(y)))
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
            X_embedded = TSNE(n_components=2).fit_transform(encoder_out)
            plt.figure(figsize=(12, 10))
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
            plt.colorbar()
            plt.show()
        print(np.bincount(y_pred))

        y_pred = kmeans.fit_predict(self.encoder.predict(x_train))
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            print('kmeans : acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
        print(np.bincount(y_pred))

        decoder_sub = decoder_out * (x_test==0) + x_test
        df = pd.DataFrame(decoder_out, columns=df_columns)
        df_replace = pd.DataFrame(decoder_sub, columns=df_columns)

        outDir = os.path.join(outdir, model_name)
        if os.path.exists(outDir) == False:
            os.makedirs(outDir)
        outPath = os.path.join(outDir, "{}.{}.complete".format(model_name, ite))

        df.to_csv(outPath, index=None, float_format='%.4f')
        df_replace.to_csv(outPath.replace(".complete", ".complete.sub"), index=None, float_format='%.4f')
        pd.DataFrame(encoder_out).to_csv(outPath.replace(".complete", ".encoder.out"), float_format='%.4f')
        print("saving done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--gene_select', default=None, type=int)
    parser.add_argument('--gamma', default=0.05,type=float)
    parser.add_argument('--gamma_s', default=0.05, type=float)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument("--early_stopping", default=20, type=int)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--train_datapath", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--labelpath", default=None, type=str)
    parser.add_argument("--outDir", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--model_name", default="data/drop80-0-1.train", type=str)
    parser.add_argument("--data_type", default="count", type=str)
    parser.add_argument("--loss", default='binary_crossentropy', type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--trans', dest='trans', action='store_true')
    feature_parser.add_argument('--no-trans', dest='trans', action='store_false')
    parser.set_defaults(trans=True)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--bn', dest='bn', action='store_true')
    feature_parser.add_argument('--no-bn', dest='bn', action='store_false')
    parser.set_defaults(bn=True)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--gene_scale', dest='gene_scale', action='store_true')
    feature_parser.add_argument('--no-gene_scale', dest='gene_scale', action='store_false')
    parser.set_defaults(gene_scale=True)
    # parser.add_argument('--trans', dest='feature', default=False, action='store_true')
    # parser.add_argument('--bn', dest='feature', default=True, action='store_true')
    # parser.add_argument('--gene_scale', dest='feature', default=True, action='store_true')
    args = parser.parse_args()
    print(args)

    # load dataset
    from datasets import load_newdata
    x_train, x_val, x_test, df_columns, df_index = load_newdata(args)
    n_clusters = args.n_clusters

    y = None
    if args.labelpath is not None:
        from sklearn.preprocessing import LabelEncoder
        labeldf = pd.read_csv(args.labelpath, header=0, index_col=0)
        y = labeldf.values
        y = y.transpose()
        y = np.squeeze(y)
        if not isinstance(y, (int, float)):
            y = LabelEncoder().fit_transform(y)
        n_clusters = len(np.unique(y))
        # labeldf = pd.read_csv(args.labelpath, header=None, index_col=None)
        # y = labeldf.values
        # y = y.transpose()
        # y = np.squeeze(y)
        # n_clusters = len(np.unique(y))
        print("has {} clusters:".format(n_clusters))
        print(np.bincount(y))

    init = 'glorot_uniform'
    optimizer1 = Adam(lr=0.0001, decay=0.)
    # optimizer = SGD(lr=0.01, momentum=0.0, decay=0.99, nesterov=False)
    # setting parameters
    # if args.dataset == 'mnist' or args.dataset == 'fmnist':
    #     update_interval = 140
    #     pretrain_epochs = 300
    #     init = VarianceScaling(scale=1. / 3., mode='fan_in',
    #                            distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    #     pretrain_optimizer = SGD(lr=1, momentum=0.9)
    # elif args.dataset == 'reuters10k':
    #     update_interval = 30
    #     pretrain_epochs = 50
    #     init = VarianceScaling(scale=1. / 3., mode='fan_in',
    #                            distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    #     pretrain_optimizer = SGD(lr=1, momentum=0.9)
    # elif args.dataset == 'usps':
    #     update_interval = 30
    #     pretrain_epochs = 50
    # elif args.dataset == 'stl':
    #     update_interval = 30
    #     pretrain_epochs = 10

    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    # # prepare the DEC model
    dims = [x_train.shape[-1], 300, 100, 30, 10]
    # dims = [x_train.shape[-1], 512, 128, 32]
    Idec = s_IDEC(dims=dims, n_clusters=n_clusters, gamma=args.gamma, bn=args.bn)
    #
    if args.ae_weights is None:
        Idec.pretrain(x=x_test, loss=args.loss, optimizer=optimizer1,
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     gamma_s=args.gamma_s)
    else:
        Idec.autoencoder.load_weights(args.ae_weights)
    #
    Idec.model.summary()
    t0 = time()
    Idec.compile(optimizer=optimizer1, loss={'clustering': 'kld', 'decoder_0': args.loss},
                 loss_weights=[args.gamma, 1])
    Idec.fit(x_test, x_test, x_test, y=y, model_name=args.model_name, outdir=args.outDir, df_columns=df_columns,
             epoch=args.epoch, batch_size=args.batch_size,
                     update_interval=update_interval, early_stopping=args.early_stopping, tol=args.tol)
