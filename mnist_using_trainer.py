import math
import numpy
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

###########################################prepare the dataset
#from chainer.datasets import mnist
#train, test = mnist.get_mnist()]
train, test = chainer.datasets.get__mnist()
###########################################prepare the dataset iterations
#Iterator can creates a mini-batch from the given dataset
batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, False)

###########################################prepare the model
from net import Net

#gpu_id = 0

model = Net()
#if gpu_id >= 0:
#    model.to_gpu(gpu_id)

##########################################prepare the updater

MAX_EPOCH = 10
# Wrap your model by Classifier and include the process of loss calculation within your model.
# Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
model = L.Classifier(model)
#Selection of your otimizing method
optimizer = optimizers.MomentumSGD()
#Give the optimizer a reference to your model
optimizer.setup(model)
#Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

##########################################Setup the Trainer
trainer = training.Trainer(updater, (MAX_EPOCH, 'epoch'),  out = 'mnist_result')

##########################################Add extentions to the Trainer object
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device = gpu_id))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.DumpGraph('main/loss'))

##########################################Start trainig
trainer.run()
