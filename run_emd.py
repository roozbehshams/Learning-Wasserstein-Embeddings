#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:10:16 2018

@author: mducoffe, rflammary, ncourty
"""

"""
run emd on mnist
"""

import numpy as np
import scipy as sp
import scipy.io as spio
import ot
import os
from tqdm import tqdm
import parmap
from multiprocessing import Pool
import h5py

#from keras.datasets import mnist, fashion_mnist
#import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
import gryds




FMNIST = 'synth-fmnist'
MNIST='mnist'
MNIST_N2='mnistN2'
REPO='data'
CAT='cat'
CRAB='crab'
FACE='face'
REPO='data'
#%%

def run_emd_earthbender(dataset_name='fmnist', train=True, n_pairwise=1000000, n_iter=1, n_proc=None):
    assert dataset_name in [FMNIST]
    if n_proc is None:
        import multiprocessing
        n_proc = multiprocessing.cpu_count()

    print('number of processors used {}'.format(n_proc))

    if dataset_name==FMNIST:
        n = 32
        h5_file = h5py.File('synth-deform-fmnist.hd5', 'w')

        #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        h5_fmnist = h5py.File('fashion_mnist.h5')
        x_train = np.array(h5_fmnist['x_train'])
        y_train = np.array(h5_fmnist['y_train'])
        x_test = np.array(h5_fmnist['x_test'])
        y_test = np.array(h5_fmnist['y_test'])

        #padding to be 32x32:
        x_train = np.pad(x_train, [[0, 0], [2, 2], [2, 2]], 'constant')
        x_test = np.pad(x_test, [[0, 0], [2, 2], [2, 2]], 'constant')

        x_train_vec = (x_train.reshape((len(x_train), -1))).astype(float)
        x_test_vec = (x_train.reshape((len(x_test), -1))).astype(float)

        x_train_vec /= x_train_vec.sum(1).reshape((-1, 1))
        x_test_vec /= x_test_vec.sum(1).reshape((-1, 1))

        x_train_transformed=[]
        x_test_transformed=[]

        #for i in tqdm(range(len(x_train))):
        #    x_train_transformed.append(random_deformation_gen(x_train[i]))

        with Pool(n_proc) as pool:
            x_train_transformed = parmap.map(random_deformation_gen, x_train, pm_pbar=True, pm_pool=pool,
                                             pm_chunksize=50)
            x_test_transformed = parmap.map(random_deformation_gen, x_test, pm_pbar=True, pm_pool=pool,
                                            pm_chunksize=50 )

            x_train_moving = np.array([x[0] for x in x_train_transformed])
            x_train_flow = np.array([x[1] for x in x_train_transformed])

            x_test_moving = np.array([x[0] for x in x_test_transformed])
            x_test_flow = np.array([x[1] for x in x_test_transformed])

            x_train_transformed_vec = (x_train_moving.reshape((len(x_train), -1))).astype(float)
            x_test_transformed_vec = (x_test_moving.reshape((len(x_test), -1))).astype(float)

            x_train_transformed_vec /= x_train_transformed_vec.sum(1).reshape((-1, 1))
            x_test_transformed_vec /= x_test_transformed_vec.sum(1).reshape((-1, 1))

            xx, yy = np.meshgrid(np.arange(n), np.arange(n))
            xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

            M = ot.dist(xy, xy)

            ilist_train = range(len(x_train_vec))
            ilist_test = range(len(x_test_vec))


            emdist_train = parmap.map(compute_emd_earth_bender, ilist_train, x_train_vec, x_train_transformed_vec, M, pm_pbar=True, pm_pool=pool,
                                            pm_chunksize=50)
            emdist_test = parmap.map(compute_emd_earth_bender, ilist_test, x_test_vec, x_test_transformed_vec, M,
                                      pm_pbar=True, pm_pool=pool,
                                      pm_chunksize=50)

            train_gr = h5_file.create_group('train')
            train_gr.create_dataset('fixed',data=x_train, dtype=np.uint8, compression="gzip")
            train_gr.create_dataset('moving', data=x_train_moving, dtype=np.uint8, compression="gzip")
            train_gr.create_dataset('wass_distance', data=emdist_train, compression="gzip")

            test_gr = h5_file.create_group('test')
            test_gr.create_dataset('fixed', data=x_test, dtype=np.uint8, compression="gzip")
            test_gr.create_dataset('moving', data=x_test_moving, dtype=np.uint8, compression="gzip")
            test_gr.create_dataset('wass_distance', data=emdist_test, compression="gzip")

            h5_file.close()





            print('done')
            # D2=np.array(ot.utils.parmap(compute_emd,ilist,n_proc))
            # D2=np.array(list(map(compute_emd,ilist)))
            #D2 = np.array((parmap.map(compute_emd_earth_bender, ilist, x_train_vec, , M)))
            #spio.savemat('{}/{}_{}_{}.mat'.format(REPO, dataset_name, 'train' if train else 'test', i),
            #             {'is': isource, 'it': itarget, 'D': D2})




def run_emd(dataset_name='mnist', train=True, n_pairwise=1000000, n_iter=1, n_proc=None):
    assert dataset_name in [MNIST_N2, MNIST, CAT, CRAB, FACE], 'unknown dataset {}'.format(dataset_name)
    
    if n_proc is None:
        import multiprocessing
        n_proc = multiprocessing.cpu_count()
        
    print('number of processors used {}'.format(n_proc))
    
    if dataset_name==MNIST:
        n = 28
        if train:
            (x_train, _), _ = mnist.load_data()
            xapp=x_train.reshape((len(x_train),-1))*1.0
        else:
            _, (x_test, _) = mnist.load_data()
            xapp=x_test.reshape((len(x_test),-1))*1.0
    
    if dataset_name==MNIST_N2:
        n = 32
        if train:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train =  np.pad(x_train[y_train!=2], [[0,0],[2,2],[2,2]], 'constant')
            print(x_train.shape)
            xapp=x_train.reshape((len(x_train),-1))*1.0
        else:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_test =  np.pad(x_test[y_test!=2], [[0,0],[2,2],[2,2]], 'constant')
            xapp=x_test.reshape((len(x_test),-1))*1.0
            
    if dataset_name in [CAT, CRAB, FACE]:
        n=28
        url_path = "https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap"
        
        assert os.path.isfile(os.path.join(REPO, '{}.npy'.format(dataset_name))), \
            "file not found: please download it at '{}' and put it in './{}/{}.npy'".format(url_path, REPO, dataset_name)
            
        X = np.load(os.path.join(REPO, '{}.npy'.format(dataset_name)))
        X = X.reshape((len(X),-1))*1.0
        X /=X.sum(1).reshape((-1,1))
        X=X.reshape((-1,1,n,n))
        
        # split into train, and test
        N = len(X)
        n_test = (int)(0.2*N)
        n_train = N - 2*n_test
        x_train = X[:n_train]
        x_test = X[n_train:]

        if train:
            xapp=x_train.reshape((len(x_train), -1))
        else:
            xapp=x_test.reshape((len(x_test), -1))
    ###################################################################""
            
    N = len(xapp)
    print(N)
    print(xapp.shape)
    xapp/=xapp.sum(1).reshape((-1,1))
    xx,yy=np.meshgrid(np.arange(n),np.arange(n))
    xy=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
    
    M=ot.dist(xy, xy)
        
    for i in tqdm(range(n_iter)):

        isource=np.random.randint(0,N,n_pairwise)
        itarget=np.random.randint(0,N,n_pairwise)


        ilist=range(n_pairwise)
        #D2=np.array(ot.utils.parmap(compute_emd,ilist,n_proc))        
        #D2=np.array(list(map(compute_emd,ilist)))
        D2=np.array((parmap.map(compute_emd,ilist, xapp, isource, itarget, M)))        
        spio.savemat('{}/{}_{}_{}.mat'.format(REPO, dataset_name, 'train' if train else 'test', i),{'is':isource,'it':itarget,'D':D2})
        
def compute_emd(i,xapp, isource, itarget, M):
            return ot.emd2(xapp[isource[i],:],xapp[itarget[i],:],M)


def compute_emd_earth_bender(i, x_fixed, x_moving, M):
    return ot.emd2(x_fixed[i], x_moving[i], M)

def random_deformation_gen(image, bspline_shape=(3, 3), std=0.15):
    assert image.shape[0] == image.shape[1]
    bspline_grid_shape = (len(image.shape),) + bspline_shape
    bspline_grid = np.random.rand(*bspline_grid_shape) * std

    a_bspline_transformation = gryds.BSplineTransformation(bspline_grid)
    an_image_interpolator = gryds.Interpolator(image, order=1)

    an_image_grid = gryds.Grid(image.shape)  # makes a Grid the size of the image
    a_deformed_image_grid = an_image_grid.transform(a_bspline_transformation)
    a_deformed_image = an_image_interpolator.resample(a_deformed_image_grid)

    f = (a_deformed_image_grid.grid - an_image_grid.grid) * image.shape[0]
    flow = np.ndarray(image.shape + (len(image.shape),))

    for i in range(len(image.shape)):
        flow[...,i] = f[i,...]


    return a_deformed_image, flow


def example_gen_f_mnist(batch_size=1):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in
            npz files. default to 'vol_data'
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    fmnist_train = x_train

    while True:
        idx = np.random.randint(1, len(fmnist_train), size=batch_size)
        X_data = []

        # x_fixed = mnist_zero_fixed
        # x_fixed = np.pad(x_fixed,2,mode='constant')
        # x_fixed = x_fixed[..., np.newaxis]

        x_fixed = np.squeeze(fmnist_train[idx].astype(float))
        x_fixed = np.pad(x_fixed, [[2, 2], [2, 2]], mode='constant')
        [x_moving, x_flow] = random_deformation_gen(x_fixed)

        x_fixed = x_fixed[np.newaxis,..., np.newaxis]
        x_moving = x_moving[np.newaxis,..., np.newaxis]
        x_flow = x_flow[np.newaxis,...]

        # X = np.concatenate((x_moving,x_fixed), axis=2)
        X_data.append(x_fixed)
        X_data.append(x_moving)
        X_data.append(x_flow)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = X_data
        yield tuple(return_vals)
        #%%
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
		
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset_name', type=str, default='cat', help='dataset name')
    parser.add_argument('--n_pairwise', type=int, default=10000, help='number of pairwise emd')
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--train', type=str2bool, default=True, help='number of iterations')
    
    args = parser.parse_args()                                                                                                                                                                                                                             
    dataset_name=args.dataset_name
    n_pairwise=args.n_pairwise
    n_iter=args.n_iter
    train=args.train
    print(train)

    run_emd_earthbender(dataset_name, train, n_pairwise, n_iter)