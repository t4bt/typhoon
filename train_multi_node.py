from __future__ import print_function

import argparse
import os

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, serializers
from chainer.training import extensions
from mpi4py import MPI

import chainermn

import model
import pickle
import preprocess


def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Multi-node training script based on ChainerMN example.')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result/multi_node/',
                        help='Directory to output the result')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset filepath.')
    parser.add_argument('--no_snapshot', action='store_true',
                        help='Suppress storing snapshots.')
    args = parser.parse_args()
    assert args.communicator != 'naive', 'Error: "naive" communicator does not support GPU.'
    return args


def prepare_extensions(trainer, evaluator, args, comm):
    trainer.extend(evaluator)
    trainer.extend(extensions.ExponentialShift('lr', 0.98), trigger=(1, 'epoch'))

    if comm.rank != 0:
        return
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    if not args.no_snapshot:
        trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
         'epoch', file_name='accuracy.png'))
    return


def main(args, model, x, t, valid_rate=0.2):
    print('Start a training script using multiple nodes.')

    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank
    assert device >= 0, 'invalid device ID: {}'.format(device)

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
        print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    if comm.rank == 0:
        threshold = int(len(t)*(1-valid_rate))
        train = datasets.tuple_dataset.TupleDataset(x[0:threshold], t[0:threshold])
        valid = datasets.tuple_dataset.TupleDataset(x[threshold:], t[threshold:])
        datasize = len(train) * args.epoch
    else:
        train, valid = None, None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    valid = chainermn.scatter_dataset(valid, comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)

    if device >= 0:
        cuda.get_device_from_id(device).use()
        model.to_gpu()

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.SGD(lr=2e-4), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-2))

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = extensions.Evaluator(valid_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)

    prepare_extensions(trainer, evaluator, args, comm)

    trainer.run()

    if comm.rank == 0:
        throughput = datasize / trainer.elapsed_time
        print('Throughput: {} [images/sec.] ({} / {})'.format(
            throughput, datasize, trainer.elapsed_time))

        model_filepath = os.path.join(args.out, 'trained.model')
        chainer.serializers.save_npz(model_filepath, model)


if __name__ == '__main__':
    args = parse_cmd_args()
    with open("train.pkl", "rb") as f:
        x, t = pickle.load(f)
    M = L.Classifier(model.CNN3())
    serializers.load_npz("cnn3_multi_node_3_/trained.model", M)
    main(args, M, x, t)
