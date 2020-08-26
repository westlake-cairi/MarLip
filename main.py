import os
import csv
import json
import torch
import signal
import argparse
import datetime
import numpy as np
from multiprocessing import Process, Manager

# Some self-defined functions that need to be imported
import dataset
from model import MLDL_model
from loss import MLDL_Loss
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description="author: CAIRI")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Train(model, loss, epoch, train_data, train_label, index_generator, batch_size):

    """
    Train the model for one loop.

    Arguments:
        model {torch model} -- a model need to train
        loss {torch model} -- a model used to get the loss
        epoch {int} -- current epoch
        train_data {tensor} -- the train data
        train_label {tensor} -- the train label, for unsuprised method, it is only used in plot figs
        index_generator {class} -- an index generator used for training
        batch_size {int} -- batch size
    """

    model.train()
    loss.SetEpoch(epoch)
    index_generator.Reset()

    train_loss_sum = [0, 0, 0, 0]
    num_train_sample = train_data.shape[0]
    num_batch = (num_train_sample - 0.5) // batch_size + 1
    for batch_idx in torch.arange(num_batch):
        sample_index = index_generator.CalSampleIndex(batch_idx)
        data = train_data[sample_index].float()
        label = train_label[sample_index]

        if 'Spheres5500' in param['DATASET'] or 'Spheres10000' in param['DATASET']:

            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            train_info = model(data)
            loss_list = loss.CalLosses(train_info)

            for i, loss_item in enumerate(loss_list):
                loss_item.backward(retain_graph=True)
                train_loss_sum[i] += loss_item.item()

            optimizer.step()

        else:
            optimizer_enc.zero_grad()

            # Add data to device and forward
            data = data.to(device)
            label = label.to(device)
            train_info = model(data)
            loss_list = loss.CalLosses(train_info)

            for i, loss_item in enumerate(loss_list[1:]):
                loss_item.backward(retain_graph=True)
                train_loss_sum[i+1] += loss_item.item()

            optimizer_enc.step()

            optimizer_dec.zero_grad()
            for i, loss_item in enumerate(loss_list[0:1]):
                loss_item.backward(retain_graph=True)
                train_loss_sum[i] += loss_item.item()

            optimizer_dec.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {}'.format(
            epoch, 
            batch_idx * len(data), 
            num_batch * len(data),
            batch_idx / num_batch * 100,
            [loss_list[i].item() for i in range(len(loss_list))]
            )
        )

    # Print average losses
    time_stamp = datetime.datetime.now()
    print("time_stamp:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    print('====> Epoch: {} Average loss: {}'.format(
        epoch,
        [train_loss_sum[i] / num_batch for i in range(len(loss_list))]
        )
    )


def Generation(model, latent_point, label_point, latent_index):
    gen_latent = Sampling().Inter(latent_point[latent_index], number_points=10000)
    gen_data = model.Decoder(torch.tensor(gen_latent, device=device).float()).detach().cpu().numpy()
    gif_ploter.Plot_Generation(latent_point[0], latent_point[latent_index], latent_point[-1], gen_latent, gen_data, label_point, title = path + '/Generation.png')


def Generalization(Model, path):
    test_data, test_label = dataset.LoadData(
        data_name=param['DATASET'],
        data_num=8000,
        seed=param['SEED']+1,
        noise=param['Noise'],  
        remove='fivecircle',
        test=True
    )   

    InlinePlot(Model, param['BATCHSIZE'], test_data, test_label, path=path, name='Test', indicator=False, mode=param['Mode'])


def InlinePlot(model, batch_size, datas, labels, path, name, indicator=False, mode='ML-AE'):

    """
    For testing models, saving intermediate data, and plotting figs.

    Arguments:
        model {torch model} -- a model need to train
        batch_size {int} -- batch size
        datas {tensor} -- the train data
        labels {tensor} -- the train label, for unsuprised method, it is only used in plot figs

    Keyword Arguments:
        path {str} -- the path to save the fig
        name {str} -- the name of current fig
        indicator {bool} -- a flag to calculate the indicator (default: {True})
        mode {str} -- set the mode for plotting. (default: {'ML-AE'})
    """
    
    model.train()
    train_loss_sum = [0, 0, 0, 0]
    num_train_sample = datas.shape[0]
    num_batch = (num_train_sample - 1) // batch_size + 1

    for batch_idx in torch.arange(num_batch):

        start_number = (batch_idx * batch_size).int()
        end_number = torch.min(
                        torch.tensor(
                            [batch_idx * batch_size + batch_size, num_train_sample]
                        )
                    ).int()

        data = datas[start_number:end_number].float().to(device)
        label = labels[start_number:end_number].to(device)

        train_info = model(data)
        loss_list = loss.CalLosses(train_info)

        # Converting intermediate results as numpy for subsequent metrics evaluation and plotting
        for i, loss_item in enumerate(loss_list):
            train_loss_sum[i] += loss_item.item()

        if batch_idx == 0:
            latent_point = []
            for train_info_item in train_info:
                latent_point.append(train_info_item.detach().cpu().numpy())

            label_point = label.cpu().detach().numpy()
        else:
            for i, train_info_item in enumerate(train_info):
                latent_point_c = train_info_item.detach().cpu().numpy()
                latent_point[i] = np.concatenate((latent_point[i], latent_point_c), axis=0)

            label_point = np.concatenate((label_point, label.cpu().detach().numpy()), axis=0)

    # Plotting a new fig for the current epoch
    if param['DATASET'] != 'MNIST' or param['Visualization'] == True:
        gif_ploter.AddNewFig(
            latent_point, 
            label_point,
            title = path + '/' + name,
            loss = train_loss_sum,
            dataset = param['DATASET']
        )

    # Used for metrics evaluation and executed at the completion of the entire training process.
    if indicator:
        latent_index = 2 * len(param['NetworkStructure']) - 3
        indicator = CompPerformMetrics(
                        datas.reshape(datas.shape[0], -1), 
                        torch.tensor(latent_point[latent_index], device=device),
                        dataset = param['DATASET']
                    )

        # Saving intermediate results
        if os.path.exists(path + '/out/') is False:
            os.makedirs(path + '/out/')
        for i, info in enumerate(latent_point):
            np.savetxt(path + '/out/{}.txt'.format(i), info)
        np.savetxt(path + '/out/label.txt', label_point)

        # Save the metrics to a csv file
        outFile = open(path + '/PerformMetrics.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        names = []
        results = []
        for v, k in indicator.items():
            names.append(v)
            results.append(str(round(k, 6)))
        writer.writerow(names)
        writer.writerow(results)

        print(indicator)

        # Perform sampling in idden layer, generate new manifold, and plot figs
        if mode == 'Generation':
            Generation(model, latent_point, label_point, latent_index)


def SetSeed(seed):

    """
    function used to set a random seed

    Arguments:
        seed {int} -- seed number, will set to torch and numpy
    """
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def SetParam():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--name", default=None, type=str)   # File names where data and figs are stored
    parser.add_argument("-PP", "--ParamPath", default='None', type=str)   # Path for an existing parameter
    parser.add_argument("-M", "--Mode", default='ML-AE', type=str)
    parser.add_argument("-D", "--DATASET", default='SwissRoll', type=str, choices=['SwissRoll', 'SCurve', 'MNIST', 'Spheres10000', 'Spheres5500'])
    parser.add_argument("-LR", "--LEARNINGRATE", default=1e-3, type=float)
    parser.add_argument("-B", "--BATCHSIZE", default=800, type=int)
    parser.add_argument("-RB", "--RegularB", default=3, type=float)   # Boundary parameters for push-away Loss
    parser.add_argument("-ND", "--N_Dataset", default=800, type=int)   # The data number used for training
    parser.add_argument("-GC", "--GradualChanging", default=[500, 1000], type=int, nargs='+')   # Range for the gradual changing of push-away Loss
    parser.add_argument("-R", "--ratio", default=[0.2, 1.0, 0.0, 1.0], type=float, nargs='+')   # The weight ratio for loss_ae/loss_iso/loss_angle/loss_push-away
    parser.add_argument("-EPS", "--Epsilon", default=0.23, type=float)   # The boundary parameters used to determine the neighborhood
    parser.add_argument("-MK", "--MAEK", default=15, type=int)
    parser.add_argument("-E", "--EPOCHS", default=10000, type=int)
    parser.add_argument("-P", "--PlotForloop", default=1000, type=int)   # Save data and plot every 1000 epochs
    parser.add_argument("-SD", "--SEED", default=0, type=int)   # Seeds used to ensure reproducible results
    parser.add_argument("-NS", "--NetworkStructure", default=[3, 100, 100, 100, 3, 2], type=int, nargs='+')
    parser.add_argument("-Noise", "--Noise", default=0.0, type=float)   # Noise added to the generated data
    parser.add_argument("-MultiRun", "--Train_MultiRun", default=False, action='store_true')
    parser.add_argument("-Visualization", "--Visualization", default=False, action='store_true')
    args = parser.parse_args()

    if args.DATASET == 'MNIST' and args.Visualization == False:
        args.ParamPath = './param/mnist_25.json'
    if args.DATASET == 'MNIST' and args.Visualization == True:
        args.ParamPath = './param/mnist_2.json'
    if args.DATASET == 'Spheres5500':
        args.ParamPath = './param/spheres5500.json'
    if args.DATASET == 'Spheres10000':
        args.ParamPath = './param/spheres10000.json'
    if args.ParamPath is not 'None':
        jsontxt = open(args.ParamPath, 'r').read()
        param = json.loads(jsontxt)
    else:
        args = parser.parse_args()
        param = args.__dict__

    if param['name'] == None:
        if param['DATASET'] == 'SwissRoll' or param['DATASET'] == 'SCurve':
            path = "./pic/{}_{}_N{}_SD{}".format('MLDL', param['DATASET'], param['N_Dataset'], param['SEED'])
        else:
            path = "./pic/{}_{}_N{}".format('MLDL', param['DATASET'], param['N_Dataset'])
    else:
        path = "./pic/{}".format(param['name'])

    # Save parameters
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump(param, open(path + '/param.json', 'a'), indent=2)

    return param, path


def SetModel(param):
    Model = MLDL_model(param).to(device)
    loss = MLDL_Loss(args=param, cuda=device)

    return Model, loss


def Train_MultiRun():
    # Combination of multiple parallel training parameters (only SEED is set below, different parameters can be set as needed)
    cmd=[]
    for i in range(10):
        cmd.append('CUDA_VISIBLE_DEVICES={} '+'python main.py -SD {seed}'.format(seed=i))

    signal.signal(signal.SIGTERM, term)
    gpustate=Manager().dict({str(i):True for i in range(1,8)})
    processes=[]
    idx=0

    # Open multiple threads to perform multiple GPU parallel training
    while idx<len(cmd):
        for gpuid in range(1,8):
            if gpustate[str(gpuid)]==True:
                print(idx)
                gpustate[str(gpuid)]=False
                p=Process(target=run,args=(cmd[idx],gpuid,gpustate),name=str(gpuid))
                p.start()

                print(gpustate)
                processes.append(p)
                idx+=1

                break

    for p in processes:
        p.join()


if __name__ == '__main__':

    param, path = SetParam()
    if param['Train_MultiRun']:
        Train_MultiRun()
    else:
        SetSeed(param['SEED'])

        # Load training data
        train_data, train_label = dataset.LoadData(
            data_name=param['DATASET'],
            data_num=param['N_Dataset'],
            seed=param['SEED'],
            noise=param['Noise'],
            test=False   
        )

        if param['DATASET'] == 'Spheres5500':
            test_data = train_data[-5500:]
            test_label = train_label[-5500:]
            train_data = train_data[:-5500]
            train_label = train_label[:-5500]
        if param['DATASET'] == 'Spheres10000':
            test_data = train_data[9000:]
            test_label = train_label[9000:]
            train_data = train_data[:7500]
            train_label = train_label[:7500]

            
        # Init the model
        Model, loss = SetModel(param)
        optimizer = torch.optim.Adam(Model.parameters(), lr=param['LEARNINGRATE'])
        param_enc = [str(i*2) for i in range(len(param['NetworkStructure']) - 1)]
        param_dec = [str(int(param_enc[-1]) + 1 + i*2) for i in range(len(param['NetworkStructure']) - 1)]
        optimizer_enc = torch.optim.Adam([{'params': [param for name, param in Model.named_parameters() if
                                                any([s in name for s in param_enc])]}], lr=param['LEARNINGRATE'])
        optimizer_dec = torch.optim.Adam([{'params': [param for name, param in Model.named_parameters() if
                                            any([s in name for s in param_dec])]}], lr=param['LEARNINGRATE'])

        index_generator = dataset.SampleIndexGenerator(train_data, param['BATCHSIZE'])
        gif_ploter = GIFPloter(param, Model)

        # Start training
        for epoch in range(param['EPOCHS'] + 1):
            Train(Model, loss, epoch, train_data, train_label, index_generator, param['BATCHSIZE'])

            if epoch % param['PlotForloop'] == 0:
                name = 'Epoch_' + str(epoch).zfill(5)
                InlinePlot(Model, param['BATCHSIZE'], train_data, train_label, path, name, indicator=False)
                if param['DATASET'] == 'Spheres5500' or param['DATASET'] == 'Spheres10000':
                    InlinePlot(Model, param['BATCHSIZE'], test_data, test_label, path, 'Test_' + name, indicator=False)

        # Plotting the final results and evaluating the metrics
        # InlinePlot(Model, param['BATCHSIZE'], test_data, test_label, path, name='Test', indicator=True, mode=param['Mode'])
        InlinePlot(Model, param['BATCHSIZE'], train_data, train_label, path, name='Train', indicator=True, mode=param['Mode'])
        if param['DATASET'] != 'MNIST' or param['Visualization'] == True:
            gif_ploter.SaveGIF(path=path)

        # Testing the generalizability of the model to out-of-samples
        if param['Mode'] == 'Test':
            Generalization(Model, path)