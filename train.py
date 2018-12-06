from __future__ import print_function
import argparse
import os

import multiprocessing

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import model
import pickle
import preprocess

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Chainer Training:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs (at least 2)')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--no_snapshot', action='store_true',
                        help='Suppress storing snapshots.')
    args = parser.parse_args()

    print('# GPUs: {}'.format(args.n_gpus))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    return args

def prepare_extensions(trainer, evaluator, args):
    trainer.extend(evaluator)

    trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(20, 'epoch'))

    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    if not args.no_snapshot:
        trainer.extend(extensions.snapshot(), trigger=(25, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
         'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.ProgressBar(update_interval=100))

def train_using_gpu(args, model, x, t, valid_rate=0.1):
    if args.n_gpus == 1:
        print('Start a training script using single GPU.')
    else:
        multiprocessing.set_start_method('forkserver')
        print('Start a training script using multiple GPUs.')

    # Set up a dataset and prepare train/valid data iterator.
    threshold = int(len(t)*(1-valid_rate))
    train = datasets.tuple_dataset.TupleDataset(x[0:threshold], t[0:threshold])
    valid = datasets.tuple_dataset.TupleDataset(x[threshold:], t[threshold:])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Make a specified GPU current
    master_gpu_id = 0
    if args.n_gpus == 1:
        cuda.get_device_from_id(master_gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
    else:
        cuda.get_device_from_id(master_gpu_id).use()

    # Make optimizer.
    optimizer = chainer.optimizers.Adam(alpha=2e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-3))
    #optimizer.add_hook(chainer.optimizer.Lasso(1e-5))

    # Set up a trainer
    if args.n_gpus == 1:
        updater = training.StandardUpdater(train_iter, optimizer, device=0)
    else:
        devices_list = {'main': master_gpu_id}
        devices_list.update({'gpu{}'.format(i): i for i in range(1, args.n_gpus)})
        print(devices_list)
        updater = training.updaters.ParallelUpdater(train_iter, optimizer, devices=devices_list)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(valid_iter, model, device=master_gpu_id)

    # Set some extension modules to a trainer.
    prepare_extensions(trainer, evaluator, args)

    # Run the training
    trainer.run()

    # Show real throughput.
    datasize = len(train) * args.epoch
    throughput = datasize / trainer.elapsed_time
    print('Throughput: {} [images/sec.] ({} / {})'.format(
        throughput, datasize, trainer.elapsed_time))

    # Save trained model.
    model_filepath = os.path.join(args.out, 'trained.model')
    chainer.serializers.save_npz(model_filepath, model)


if __name__ == '__main__':
    args = parse_cmd_args()
    with open("train.pkl", "rb") as f:
        x, t = pickle.load(f)
    M = L.Classifier(model.CNN())
    if args.n_gpus > 0:
        train_using_gpu(args, M, x, t)
