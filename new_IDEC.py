from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
from DEC import autoencoder,ClusteringLayer,DEC

class IDEC(DEC):
    def __init__(self,
                 dims,
                 gamma=0.1,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        # super(IDEC,self).__init__(dims,
        #          n_clusters=10,
        #          alpha=1.0,
        #          init='glorot_uniform')
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        # self.model = Model(inputs=self.autoencoder.input,
        #                    outputs=self.autoencoder.output)
        # print(self.model)

    def compile(self, optimizer, loss, loss_weights):
        self.model.compile(optimizer=optimizer, loss=loss,loss_weights=loss_weights)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):
        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/Idec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q,_ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if index == 0:
                np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=[p[idx],x[idx]])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--model_name', default='results')
    parser.add_argument('--gene_select', default=1000,type=int)
    parser.add_argument('--gamma',default=0.1,type=float)
    args = parser.parse_args()
    print(args)
    save_dir = args.save_dir+'/'+args.model_name
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # load dataset
    # from datasets import load_data
    # x, y = load_data(args.dataset)
    # n_clusters = len(np.unique(y))

    # load dataset
    from datasets import load_mydata
    x, y = load_mydata(args.model_name,gene_select=args.gene_select)
    n_clusters = len(np.unique(y))



    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
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
    Idec = IDEC(dims=[x.shape[-1], 300, 100, 30, 10], n_clusters=n_clusters, gamma=args.gamma)
    #
    if args.ae_weights is None:
        Idec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     save_dir=save_dir)
    else:
        Idec.autoencoder.load_weights(args.ae_weights)
    #
    Idec.model.summary()
    t0 = time()
    Idec.compile(optimizer='adam', loss={'clustering': 'kld', 'decoder_0': 'mse'},
                 loss_weights=[args.gamma, 1])
    y_pred = Idec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=update_interval, save_dir=save_dir)
    print('acc:', metrics.acc(y, y_pred))
    print('clustering time: ', (time() - t0))


