import model
import preprocess
import pickle
import numpy as np
import chainer
from chainer import serializers, Variable
import chainer.links as L
import csv

chainer.config.train=False

def load_model(model, trained_model="result/trained.model", gpu_id=0):
    model = L.Classifier(model)
    serializers.load_npz(trained_model, model)
    if gpu_id > -1:
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
    return model

def predict(model, x, gpu_id=0):
    batch_size = 100
    epoch = int(np.ceil(len(x)/batch_size))
    y = np.array([]).astype(np.int32)
    for i in range(epoch):
        data = Variable(x[i*batch_size:(i+1)*batch_size])
        if gpu_id > -1:
            data.to_gpu()
        pred = model.predictor(data)
        pred.to_cpu()
        y = np.r_[y, np.argmax(pred.data, axis=1)]
    return y

def output_file(pred, fname_list, fname="submit.tsv"):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\r\n')
        data = []
        for i in range(len(pred)):
            data.append([fname_list[i], pred[i]])
        writer.writerows(data)

if __name__=='__main__':
    m = load_model(model.CNN(), trained_model="cnn/trained.model", gpu_id=-1)
    x, t = preprocess.Load('test')
    pred = predict(m, x)
    output_file(pred, t, fname="submit.tsv")
