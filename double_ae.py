import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os

import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
from keras.models import Model, model_from_json, Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as K

from keras.callbacks import ModelCheckpoint

import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from time import time

def extend_graph(graph_df, new_column_list):
    print("extending graph")
    ppi_gene_names = list(graph_df.columns)
    print("original ppi length: {}".format(len(ppi_gene_names)))
    cell_gene_names = new_column_list
    print("original gene number: {}".format(len(cell_gene_names)))
    gene_names = []
    for col in ppi_gene_names:
        if col not in cell_gene_names:
            continue
        gene_names.append(col)
    print("overlap gene number of ppi: {}".format(len(gene_names)))
    graph_df = graph_df.ix[gene_names, gene_names]
    values = graph_df.values
    dim = len(new_column_list)
    column2id = dict(zip(new_column_list, range(dim)))
    rows, cols = np.where(values > 0.0)
    edge_indexs = list( zip(rows, cols) )
    new_edge_indexs = [ [column2id[gene_names[x]], column2id[gene_names[y]] ] \
                        for (x, y) in edge_indexs
                      ]
    new_index_array = np.array(new_edge_indexs, dtype=np.int32)
    row_index, col_index = new_index_array[:, 0], new_index_array[:, 1]
    dim = len(new_column_list)
    new_graph = np.zeros((dim, dim), dtype=np.int32)
    new_graph[ row_index, col_index ] = 1.0
    new_graph = new_graph + np.diag(np.ones(shape=new_graph.shape[0]))
    return new_graph

def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div


def load_newdata(train_datapath, metric='pearson', gene_scale=False, data_type='count', trans=True):
    print("make dataset from {}...".format(train_datapath))
    df = pd.read_csv(train_datapath, sep=",", index_col=0)
    if trans:
        df = df.transpose()
    print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
    if data_type == 'count':
        df = row_normal(df)
        # df = sizefactor(df)
    elif data_type == 'rpkm':
        df = np.log(df + 1)
    df = df.transpose()
    if gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns, index=df.index)
    return df


def batch_generator(X, graph, batch_size, shuffle, beta1=1, beta2=1):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter: batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :]
        X_batch_v_j = X.transpose()[col_indices[batch_index], :]
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)*beta1
        B_i[X_batch_v_i != 0] = beta2
        B_j = np.ones(X_batch_v_j.shape)*beta1
        B_j[X_batch_v_j != 0] = beta2
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X[row_indices[batch_index], :], 1).reshape((batch_size, 1))
        deg_j = np.sum(X.transpose()[col_indices[batch_index], :], 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def weighted_mse_x(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    return K.sum(
        K.square(y_pred) * y_true[:, :-1],
        axis=-1) / y_true[:, -1]

def weighted_mse_pre(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    return K.sum(
        K.square(y_pred)* y_true,
        axis=-1)

def weighted_mse_y(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
    y_pred: Contains y2 - y1
    y_true: Contains s12
    '''
    min_batch_size = K.shape(y_true)[0]
    return K.reshape(
        K.sum(K.square(y_pred), axis=-1),
        [min_batch_size, 1]
    ) * y_true


class Autoencoder():
    def __int__(self):
        pass

    def pretrain(self):
        X_train_tmp = train_set
        self.trained_encoders = []
        self.trained_decoders = []
        for i in range(len(dims) - 1):
            print('Pre-training the layer: Input {} -> {} -> Output {}'.format(dims[i], dims[i + 1], dims[i]))
            # Create AE and training
            ae = Sequential()
            if i == 0:
                print(i)
                if i == 0:
                    x = Input(shape=(dims[0],), name='input')
                    x_drop = Dropout(dr_rate)(x)
                    h = Dense(dims[i + 1], input_dim=dims[i], activation='relu',
                              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='encoder_%d' % i)(x_drop)
                    y = Dense(dims[i], input_dim=dims[i + 1], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='decoder_%d' % i)(h)
                    x_diff = Subtract()([x, y])
                    ae = Model(inputs=x, outputs=x_diff)
                    ae.compile(loss=weighted_mse_pre, optimizer='adam')
                    ae.fit(x = train_set, y= B, batch_size=batch_size, epochs=epochs_pretrain)
                    ae.summary()
                    # Store trainined weight
                    self.trained_encoders.append(ae.layers[2])
                    self.trained_decoders.append(ae.layers[3])
                    # Update training data
                    encoder = Model(ae.input, ae.layers[2].output)
                    X_train_tmp = encoder.predict(X_train_tmp)
            else:
                if i == len(dims) - 2:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                else:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                ae.compile(loss='mean_squared_error', optimizer='adam')
                ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=epochs_pretrain)
                ae.summary()
                # Store trainined weight
                self.trained_encoders.append(ae.layers[1])
                self.trained_decoders.append(ae.layers[2])
                # Update training data
                encoder = Model(ae.input, ae.layers[1].output)
                X_train_tmp = encoder.predict(X_train_tmp)


    def pretrain1(self):
        X_train_tmp = train_set
        self.trained_encoders = []
        self.trained_decoders = []
        for i in range(len(dims) - 1):
            print('Pre-training the layer: Input {} -> {} -> Output {}'.format(dims1[i], dims1[i + 1], dims1[i]))
            # Create AE and training
            ae = Sequential()
            if i == 0:
                print(i)
                if i == 0:
                    x = Input(shape=(dims[0],), name='input')
                    x_drop = Dropout(dr_rate)(x)
                    h = Dense(dims[i + 1], input_dim=dims[i], activation='relu',
                              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='encoder_%d' % i)(x_drop)
                    y = Dense(dims[i], input_dim=dims[i + 1], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='decoder_%d' % i)(h)
                    x_diff = Subtract()([x, y])
                    ae = Model(inputs=x, outputs=x_diff)
                    ae.compile(loss=weighted_mse_pre, optimizer='adam')
                    ae.fit(x = train_set, y= B, batch_size=batch_size, epochs=epochs_pretrain)
                    ae.summary()
                    # Store trainined weight
                    self.trained_encoders.append(ae.layers[2])
                    self.trained_decoders.append(ae.layers[3])
                    # Update training data
                    encoder = Model(ae.input, ae.layers[2].output)
                    X_train_tmp = encoder.predict(X_train_tmp)
            else:
                if i == len(dims) - 2:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                else:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                ae.compile(loss='mean_squared_error', optimizer='adam')
                ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=epochs_pretrain)
                ae.summary()
                # Store trainined weight
                self.trained_encoders.append(ae.layers[1])
                self.trained_decoders.append(ae.layers[2])
                # Update training data
                encoder = Model(ae.input, ae.layers[1].output)
                X_train_tmp = encoder.predict(X_train_tmp)


    def ae_build(self):
        print('Fine-tuning')
        self.decoders = Sequential()
        self.encoders = Sequential()
        # autoencoders.add(InputLayer(input_shape=(dims[0],)))
        # encoders.add(InputLayer(input_shape=(dims[0],)))
        for encoder in self.trained_encoders:
            #     autoencoders.add(encoder)
            self.encoders.add(encoder)
        for decoder in self.trained_decoders[::-1]:
            self.decoders.add(decoder)
        self.x = Input(shape=(dims[0],), name='input')
        self.h = self.encoders(self.x)
        self.y = self.decoders(self.h)
        self.autoencoders = Model(inputs=self.x, outputs=self.y)

    def graph_ae_buld(self):
        x_in = Input(shape=(2 * dims[0],), name='x_in')
        x1 = Lambda(lambda x: x[:, 0:dims[0]], output_shape=(dims[0],))(x_in)
        x2 = Lambda(lambda x: x[:, dims[0]:2 * dims[0]], output_shape=(dims[0],))(x_in)
        # Process inputs
        x_hat1 = self.autoencoders(x1)
        x_hat2 = self.autoencoders(x2)
        y1 = self.encoders(x1)
        y2 = self.encoders(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])
        self.graph_model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])

    def train_ae(self):
        self.autoencoders.compile(optimizer='adam', loss=weighted_mse_pre)
        checkpoint = ModelCheckpoint(filepath=outdir + 'best_ae.h5', monitor='loss', save_best_only=True,
                                     save_weights_only=True)
        self.history = self.autoencoders.fit(x = train_set, y= B, batch_size=batch_size,steps_per_epoch=steps_per_epoch,
                                             nb_epoch=epochs_pretrain, callbacks=[checkpoint])
        self.autoencoders.save_weights('{}/autoencoder_ae.h5'.format(outdir))


    def train_graph(self):
        self.graph_model.compile(optimizer='adam', loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
                                  loss_weights=[1, 1, alpha])
        checkpoint = ModelCheckpoint(filepath=outdir + 'best.h5', monitor='loss', save_best_only=True,
                                     save_weights_only=True)
        self.history = self.graph_model.fit_generator(batch_generator(train_set, graph, batch_size=batch_size
                                                                       , shuffle=True, beta1=beta1, beta2=beta2),
                                                       steps_per_epoch=steps_per_epoch,
                                                       epochs=epochs_ae, callbacks=[checkpoint])
        self.graph_model.save_weights('{}/autoencoder_graph.h5'.format(outdir))


    def load_pretrain_weights(self):
        self.autoencoders.load_weights('{}/autoencoder_pretrain.h5'.format(outdir))


    def save_imputation(self):
        mask_data = train_set == 0.0
        mask_data = np.float32(mask_data)
        decoder_out = self.autoencoders.predict(train_set)
        decoder_out_replace = mask_data * decoder_out + train_set
        df_raw = pd.DataFrame(decoder_out)
        df_raw.to_csv('{}/autoencoder.csv'.format(outdir), index=None, float_format='%.4f')
        df_replace = pd.DataFrame(decoder_out_replace)
        df_replace.to_csv('{}/autoencoder_r.csv'.format(outdir), index=None, float_format='%.4f')

    def plot_loss(self):
        f = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}.png".format(outdir, name), bbox_inches='tight')
        f = plt.figure()
        plt.plot(self.history.history['subtract_2_loss'])
        plt.title('mse_1 loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_mse1.png".format(outdir, name), bbox_inches='tight')
        f = plt.figure()
        plt.plot(self.history.history['subtract_3_loss'])
        plt.title('mse_2 loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_mse2.png".format(outdir, name), bbox_inches='tight')
        plt.plot(self.history.history['subtract_4_loss'])
        plt.title('mse_3 loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_mse3.png".format(outdir, name), bbox_inches='tight')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_iters_ae', default=2000, type=int)
    parser.add_argument('--n_iters_pretrain', default=800, type=int)
    parser.add_argument('--beta1', default=0.1, type=float)
    parser.add_argument('--beta2', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--dr_rate', default=0.2, type=float)
    parser.add_argument('--nu1', default=0.0, type=float)
    parser.add_argument('--nu2', default=0.0, type=float)
    parser.add_argument("--train_datapath", default="/home/xysmlx/data/filter_data/zeisel_count.csv", type=str)
    parser.add_argument("--data_type", default="count", type=str)
    parser.add_argument("--outDir", default="/home/xysmlx/data/", type=str)
    parser.add_argument("--name", default="zeisel", type=str)

    args = parser.parse_args()
    print(args)
    name = args.name
    train_datapath = args.train_datapath
    batch_size = args.batch_size
    n_iters_pretrain = args.n_iters_pretrain
    n_iters_ae = args.n_iters_ae
    dr_rate = args.dr_rate
    nu1 = args.nu1
    nu2 = args.nu2
    outdir = args.outDir
    beta1 = args.beta1
    beta2 = args.beta2
    alpha = args.alpha

    train_df= load_newdata(train_datapath, data_type=args.data_type)
    train_set = train_df.values
    nsamples = len(train_set)
    steps_per_epoch = nsamples // batch_size
    if nsamples < batch_size:
        steps_per_epoch = nsamples
    if name == 'zeisel' or name == 'zeisel_ercc':
        graph_df = pd.read_csv("/home/xysmlx/data/ppi_mouse.csv", sep=",", index_col=0)
    elif name == 'ziegenhain':
        graph_df = pd.read_csv("/home/xysmlx/data/ppi_mouse_geneid.csv", sep=",", index_col=0)
    else:
        graph_df = pd.read_csv("/home/xysmlx/data/ppi_human.csv", sep=",", index_col=0)
    graph = extend_graph(graph_df, list(train_df.index))
    B = np.ones(train_set.shape) * beta1
    B[train_set != 0] = beta2


    dims = [train_set.shape[1], 500, 500, 2000, 10]

    epochs_pretrain = max(n_iters_pretrain // steps_per_epoch, 1)
    epochs_ae = max(n_iters_ae // steps_per_epoch, 1)

    optimizer = SGD(lr=0.1, momentum=0.99)

    ae = Autoencoder()
    ae.pretrain()
    ae.ae_build()
    ae.graph_ae_buld()
    ae.train_graph()
    ae.save_imputation()
    ae.plot_loss()