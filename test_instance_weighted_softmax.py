# PYLINT: SKIP-FILE
import os


# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
import sys


os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx
import numpy as np

import custom_iters


class InstanceWeightedSoftmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[2].asnumpy().ravel().astype(np.int)
        w = in_data[1].asnumpy().astype(np.float32)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        y *= w
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register('instance_weighted_softmax')
class InstanceWeightedSoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(InstanceWeightedSoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'weight', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        weight_shape = (in_shape[0][0], 1L)
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, weight_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return InstanceWeightedSoftmax()


# define mlp
def get_symbol():
    data = mx.symbol.Variable('data')
    weight = mx.symbol.Variable('weight')
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
    fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
    #mlp = mx.symbol.Softmax(data = fc3, name = 'softmax')
    mlp = mx.symbol.Custom(data=fc3, weight=weight, name='softmax', op_type='instance_weighted_softmax')
    return mlp

if __name__ == '__main__':
    mlp = get_symbol()

    # data
    batch_size = 128
    data_shape = (784, )
    train = mx.io.MNISTIter(
        image       = './data/train-images-idx3-ubyte',
        label       = './data/train-labels-idx1-ubyte',
        input_shape = data_shape,
        batch_size  = batch_size,
        shuffle     = False,
        flat        = True
    )
    val = mx.io.MNISTIter(
        image       = './data/t10k-images-idx3-ubyte',
        label       = './data/t10k-labels-idx1-ubyte',
        input_shape = data_shape,
        batch_size  = batch_size,
        flat        = True
    )

    weight = mx.io.CSVIter(data_csv='./data/weights.csv', data_shape=(1,), batch_size=batch_size)

    train_iter = custom_iters.DataAndWeightIter(iters=[train, weight])

    model = mx.model.FeedForward(
        ctx=mx.cpu(), symbol=mlp, num_epoch=2,
        learning_rate=0.1, momentum=0.9, wd=0.00001)

    model.fit(X=train_iter, eval_data=val,
              batch_end_callback=mx.callback.Speedometer(batch_size, 150))
