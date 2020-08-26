import os
import time
import math
import torch
import signal
import imageio
import subprocess
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process, Manager
from scipy.spatial.distance import pdist, squareform


def run(command, gpuid, gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)] = True


def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()

    except Exception as e:
        print(str(e))

        
class GIFPloter():
    def __init__(self, args, model):

        self.plot_method = 'Li'
        self.plot_every_epoch = args['PlotForloop']

        self.index_list = model.index_list
        self.name_list = model.name_list
        self.num_subfig = len(model.index_list)
        self.current_subfig_index = 2
        self.fig, self.ax = plt.subplots()

        self.his_loss = None

        if self.plot_method == 'Li':
            self.num_fig_every_row = 2
            self.num_row = int(1 + (self.num_subfig - 0.5) // self.num_fig_every_row)
            self.sub_position_list = [i*2 + 1 for i in range(self.num_subfig//2)] + [self.num_subfig] + list(reversed([i*2 + 2 for i in range(self.num_subfig//2)]))
        else:
            self.num_fig_every_row = int(np.sqrt(self.num_subfig)) + 1
            self.num_row = int(1 + (self.num_subfig - 0.5) // self.num_fig_every_row)
            self.sub_position_list = [i + 1 for i in range(self.num_subfig)]


    # Plotting the results of a single layer in the network based on the input data
    def PlotOtherLayer(self, fig, data, label, title='', fig_position0=1, fig_position1=1, fig_position2=1, s=10, dataset='None'):

        color_list = []
        for i in range(label.shape[0]):
            color_list.append(label[i])

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        data_em = data_em - data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2, projection='3d')
            ax.scatter(data_em[:, 0], data_em[:, 1], data_em[:, 2], c=color_list, s=s, cmap='rainbow')
            ax.set_zticks([])

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)
            ax.scatter(data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
            if dataset == 'Spheres5500' or dataset == 'Spheres10000':
                ax.set_xlim([-2,2])
                ax.set_ylim([-2,2])
            else:
                plt.axis('equal')

        if dataset != 'Spheres5500':
            plt.xticks([])
            plt.yticks([])

        plt.title(title, fontsize=20)

        self.current_subfig_index = self.current_subfig_index + 1


    # Plot the results for all layers of certain epoch based on the input data
    def AddNewFig(self, output_info, label, loss=None, title='', dataset='None'):

        if self.his_loss is None and loss is not None:
            self.his_loss = [[] for i in range(len(loss))]
        elif loss is not None:
            for i, loss_item in enumerate(loss):
                self.his_loss[i].append(loss_item)

        self.current_subfig_index = 1
        fig = plt.figure(figsize=(5 * self.num_fig_every_row, 5 * self.num_row))

        for i, index in enumerate(self.index_list):
            self.PlotOtherLayer(
                fig, 
                output_info[index],
                label, 
                title=self.name_list[index],
                fig_position0=self.num_row,
                fig_position1=self.num_fig_every_row,
                fig_position2=int(self.sub_position_list[i]),
                dataset = dataset
            )

        ax = fig.add_subplot(self.num_row, self.num_fig_every_row, int(max(self.sub_position_list)) + 1)
        l1, = ax.plot(
            [i*self.plot_every_epoch for i in range(len(self.his_loss[0]))],
            self.his_loss[0], 'bo-')
        l2, = ax.plot(
            [i*self.plot_every_epoch for i in range(len(self.his_loss[0]))],
            self.his_loss[1], 'ko-')
        l3, = ax.plot(
            [i*self.plot_every_epoch for i in range(len(self.his_loss[0]))],
            self.his_loss[2], 'yo-')
        l4, = ax.plot(
            [i*self.plot_every_epoch for i in range(len(self.his_loss[0]))],
            self.his_loss[3], 'ro-')
        l5, = ax.plot(
            [i*self.plot_every_epoch for i in range(len(self.his_loss[0]))],
            np.sum(self.his_loss, axis=0), 'go-')

        ax.legend((l1, l2, l3, l4, l5), ('ae', 'iso', 'angle', 'push_away', 'sum'))

        plt.title('loss history', fontsize=20)
        plt.tight_layout()
        plt.savefig(title+'.png')
        plt.close()  


    # Converting all figs to a gif
    def SaveGIF(self, path):

        gif_images_path = os.listdir(path + '/')
        gif_images_path.sort()
        gif_images = []

        for i, img_path in enumerate(gif_images_path):
            print(img_path)
            if '.png' in img_path:
                gif_images.append(imageio.imread(path + '/' + img_path))

        imageio.mimsave(path + '/' + "latent.gif", gif_images, fps=10)


    def Srotate_onepoint(self, angle, valuex, valuey, pointx, pointy):
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy

        return sRotatex,sRotatey


    def Srotate(self, angle, data):
        for i in range(data.shape[0]):
            data[i,0], data[i,1] = self.Srotate_onepoint(angle, data[i,0], data[i,1], 0, 0)

        return data


    # Plotting manifold interpolation and generating diagrams
    def Plot_Generation(self, input_data, latent, rec_data, latent_gen, gen_data, labels, title):
        
        latent = self.Srotate(np.pi/4*0.89, latent)
        latent_gen = self.Srotate(np.pi/4*0.89, latent_gen)

        fig = plt.figure(figsize=(15, 10))

        ax = fig.add_subplot(2, 3, 1, projection='3d')
        input_data = input_data - input_data.mean(axis=0)
        ax.scatter(input_data[:, 0], input_data[:, 1], input_data[:, 2], c=labels, s=10, cmap='rainbow')
        ax.set_title("training data")

        ax = fig.add_subplot(2, 3, 2)
        latent = latent - latent.mean(axis=0)
        ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=10, cmap='rainbow')
        ax.set_ylim([-2, 2])
        ax.set_title("embedding")

        ax = fig.add_subplot(2, 3, 3, projection='3d')
        rec_data = rec_data - rec_data.mean(axis=0)
        ax.scatter(rec_data[:, 0], rec_data[:, 1], rec_data[:, 2], c=labels, s=10, cmap='rainbow')
        ax.set_title("reconstruction")

        ax = fig.add_subplot(2, 3, 5)
        latent_gen = latent_gen - latent_gen.mean(axis=0)
        ax.scatter(latent_gen[:, 0], latent_gen[:, 1], c=latent_gen[:, 0], s=10, cmap='rainbow')
        ax.set_ylim([-2, 2])
        ax.set_title("random samples")
        
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        gen_data = gen_data - gen_data.mean(axis=0)
        ax.scatter(gen_data[:, 0], gen_data[:, 1], gen_data[:, 2], c=latent_gen[:, 0], s=10, cmap='rainbow')
        ax.set_title("generation")

        plt.savefig(title)
        plt.close() 


def CompPerformMetrics(data, latent, lat=None, dataset='None'):

    """
    function used to evaluate metrics

    Arguments:
        data {array} -- the data in input layers
        latent {array} -- the latent in input layers

    Outputs:
        indicator {dictionary} -- all metrics
    """

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(latent):
        latent = latent.detach().cpu().numpy()

    calc = MeasureCalculator(data, latent, 201)

    if lat is not None:
        if len(lat) == 1:
            MPE = calc.pro_error_calc(lat[0])
        else:
            MPE = (calc.pro_error_calc(lat[0]) + calc.pro_error_calc(lat[1])) / 2.0

    L_KL = calc.density_local_kl()
    MRE = calc.rmse()

    mrreZX = []
    mrreXZ = []
    Cont = []
    Trust = []

    LGD = []
    for k in range(4, 10, 1):
        LGD.append(calc.local_rmse(k=k))
    
    for k in range(10, 30, 10):
        mrreZX.append(calc.mrre(k)[0])
        mrreXZ.append(calc.mrre(k)[1])
        Cont.append(calc.continuity(k))
        Trust.append(calc.trustworthiness(k))        

    Lipschitz_min, Lipschitz_max = calc.Lipschitz(data, latent, dataset)

    indicator = {}
    indicator['L_KL'] = L_KL
    indicator['RRE'] = (np.mean(mrreZX) + np.mean(mrreXZ)) / 2.0
    indicator['Trust'] = np.mean(Trust)
    indicator['Cont'] = np.mean(Cont)
    indicator['LGD'] = np.mean(LGD)
    indicator['K_min'] = Lipschitz_min
    indicator['K_max'] = Lipschitz_max
        
    if lat is not None:
        indicator['MPE'] = MPE

    indicator['MRE'] = MRE

    return indicator


class MeasureCalculator():
    def __init__(self, Xi, Zi, k_max):
        self.k_max = k_max

        if torch.is_tensor(Xi):
            self.X = Xi.detach().cpu().numpy()
            self.Z = Zi.detach().cpu().numpy()
        else:
            self.X = Xi
            self.Z = Zi

        self.pairwise_X = squareform(pdist(self.X))
        self.pairwise_Z = squareform(pdist(self.Z))

        self.neighbours_X, self.ranks_X = \
            self._neighbours_and_ranks(self.pairwise_X, k_max)
        self.neighbours_Z, self.ranks_Z = \
            self._neighbours_and_ranks(self.pairwise_Z, k_max)


    def _neighbours_and_ranks(self, distances, k):

        """
        Inputs: 
        - distances,        distance matrix [n times n], 
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i) 
        """

        indices = np.argsort(distances, axis=-1, kind='stable')
        # Extract neighbourhoods
        neighbourhood = indices[:, 1:k+1]
        # # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind='stable')

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):

        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):

        return self.neighbours_Z[:, :k], self.ranks_Z


    # Get Metric MRE
    def rmse(self):

        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(self.pairwise_X - self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences / n**2)


    # Get Metric LGD
    def local_rmse(self, k):
        X_neighbors, _ = self.get_X_neighbours_and_ranks(k)
        mses = []
        n = self.pairwise_X.shape[0]

        for i in range(n):
            x = self.X[X_neighbors[i]]
            z = self.Z[X_neighbors[i]]
            d1 = np.sqrt(np.square(x - self.X[i]).sum(axis=1))/np.sqrt(self.X.shape[1])
            d2 = np.sqrt(np.square(z - self.Z[i]).sum(axis=1))/np.sqrt(self.Z.shape[1])
            mse = np.sum(np.square(d1 - d2))
            mses.append(mse)

        return np.sqrt(np.sum(mses)/(k*n))
        

    def _trustworthiness(self, X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):

        '''
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        '''

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood of the latent space but not in the $k$-neighbourhood of the data space
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(Z_neighbourhood[row], X_neighbourhood[row])

            for neighbour in missing_neighbours:
                result += (X_ranks[row, neighbour] - k)

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result


    # Get Metric Trust
    def trustworthiness(self, k):

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]

        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k)


    # Get Metric Cont
    def continuity(self, k):

        '''
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]

        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood, X_ranks, n, k)


    # Get Metric RRE
    def mrre(self, k):

        '''
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                mrre_ZX += abs(rx - rz) / rz

        mrre_XZ = 0.0
        for row in range(n):
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                mrre_XZ += abs(rx - rz) / rx

        C = n * sum([abs(2*j - n - 1) / j for j in range(1, k+1)])

        return mrre_ZX / C, mrre_XZ / C


    # Get Metric L-KL
    def density_local_kl(self, sigma=0.01):

        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return (density_x * (np.log(density_x) - np.log(density_z))).sum()


    def get_w(self, data):
        num = data.shape[0]
        A = np.ones((num, 3))
        b = np.zeros((num, 1))

        A[:, 0:2] = data[:, 0:2]
        b[:, 0:1] = data[:, 2:3]

        A_T = A.T
        A1 = np.dot(A_T, A)

        if np.linalg.matrix_rank(A1) == 3:
            A2 = np.linalg.inv(A1)
            A3 = np.dot(A2, A_T)
            X = np.dot(A3, b)

            w = np.zeros(4)
            w[0] = X[0, 0]
            w[1] = X[1, 0]
            w[2] = -1
            w[3] = X[2, 0]

        else:
            w = None

        return w


    def project(self, data, w): 
        if w is None:
            return 0
        else:
            A = w[0]
            B = w[1]
            C = w[2]
            D = w[3]
            dis = 0

            for i in range(data.shape[0]):
                p = data[i]
                out = np.zeros(3)  

                out[0] = ((B**2 + C**2)*p[0] - A*(B*p[1] + C*p[2] + D))/(A**2 + B**2 + C**2)
                out[1] = ((A**2 + C**2)*p[1] - B*(A*p[0] + C*p[2] + D))/(A**2 + B**2 + C**2)
                out[2] = ((A**2 + B**2)*p[2] - C*(A*p[0] + B*p[1] + D))/(A**2 + B**2 + C**2)

                dis += np.linalg.norm(out - p, ord=2)

            return dis / data.shape[0]


    # Get Metric MPE
    def pro_error_calc(self, lat):
        scale = np.max(pdist(lat))
        lat = lat/scale
        w = self.get_w(lat)
        dis = self.project(lat, w)

        return dis


    def CalPairwiseDis(self, data, neighbors):

        dis_list = []
        for i in range(data.shape[0]):
            for j in range(neighbors.shape[1]):
                m = int(neighbors[i, j])
                dis = np.linalg.norm(data[i] - data[m], ord=2) / (data.shape[1] ** 0.5)
                dis_list.append(dis)

        dis_list = np.array(dis_list)
        return dis_list


    def Neighbor(self, data, k=5):
        num = data.shape[0]
        dists = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                dists[i, j] = np.linalg.norm(data[i] - data[j], ord=2)

        neighbors = np.zeros((num, k))

        for i in range(num):
            count = 0
            index = np.argsort(dists[i, :])
            for j in range(num):
                if count < k:
                    if i != index[j]:
                        neighbors[i, count] = index[j]
                        count += 1
                else:
                    break

        return neighbors


    # Get Metric K-min and K-max
    def Lipschitz(self, x1, x2, dataset='None', K=5, L_type='Input'):
        if L_type == 'Input':
            neighbors = self.Neighbor(x1, k=K)
        if L_type == 'Latent':
            neighbors = self.Neighbor(x2, k=K)
            
        dis_list_old = self.CalPairwiseDis(x1, neighbors)
        dis_list = self.CalPairwiseDis(x2, neighbors)

        if 'Spheres5500' in dataset:
            dis_list_old = (dis_list_old-dis_list_old.min()) / (dis_list_old.max()-dis_list_old.min()) + 0.001
            dis_list = (dis_list-dis_list.min()) / (dis_list.max()-dis_list.min()) + 0.001

        dis = dis_list / dis_list_old

        dis_list = []
        for j in range(len(dis)//K):
            dis_list.append(max(np.max(dis[j*K:j*K+K]), 1.0/np.min(dis[j*K:j*K+K])))

        dis_list = np.array(dis_list)

        return np.min(dis_list), np.max(dis_list)


# Sampling of the hidden layer based on the triangular pasta sheet.
class Sampling():
    def __init__(self):
        pass

    # Return points on left side of UV
    def split(self, u, v, points):
        return [p for p in points if np.cross(p - u, v - u) < 0]

    # Find furthest point W, and split search to WV, UW
    def extend(self, u, v, points):
        if not points:
            return []

        w = min(points, key=lambda p: np.cross(p - u, v - u))
        p1, p2 = self.split(w, v, points), self.split(u, w, points)

        return self.extend(w, v, p1) + [w] + self.extend(u, w, p2)

    def convex_hull(self, points):
        # Find two hull points, U, V, and split to left and right search
        u = min(points, key=lambda p: p[0])
        v = max(points, key=lambda p: p[0])
        left, right = self.split(u, v, points), self.split(v, u, points)

        # Find convex hull on each side
        return [v] + self.extend(u, v, left) + [u] + self.extend(v, u, right) + [v]

    def GenerateSample(self, P, sample_size=500):
        p1, p2, p3 = P
        x1, y1 = p1
        x3, y3 = p2
        x2, y2 = p3

        theta = np.arange(0, 1, 0.001)
        x = theta * x1 + (1 - theta) * x2
        y = theta * y1 + (1 - theta) * y2
        x = theta * x1 + (1 - theta) * x3
        y = theta * y1 + (1 - theta) * y3
        x = theta * x2 + (1 - theta) * x3
        y = theta * y2 + (1 - theta) * y3

        rnd1 = np.random.random(size=sample_size)
        rnd2 = np.random.random(size=sample_size)
        rnd2 = np.sqrt(rnd2)

        x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
        y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

        return x, y

    def calc_area(self, p):
        p1, p2, p3 = p
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

        return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

    def CalDelTraAndAra(self, points):
        tri = Delaunay(points).simplices.copy()
        area = []

        for i in range(len(tri)):
            area.append(self.calc_area(points[tri[i]]))

        return np.array(tri), area

    def Inter(self, points, number_points=100):

        convex_hull_points = np.array(self.convex_hull(points))
        tri_point, area = self.CalDelTraAndAra(convex_hull_points)

        out_x = np.array([])
        out_y = np.array([])

        a_sum = np.sum(area)
        for i in range(len(tri_point)):
            x, y = self.GenerateSample([
                [convex_hull_points[tri_point[i][0], 0], convex_hull_points[tri_point[i][0], 1]],
                [convex_hull_points[tri_point[i][1], 0], convex_hull_points[tri_point[i][1], 1]],
                [convex_hull_points[tri_point[i][2], 0], convex_hull_points[tri_point[i][2], 1]],
            ], int(number_points*area[i]/a_sum))

            out_x = np.concatenate((out_x, x))
            out_y = np.concatenate((out_y, y))

        return np.concatenate((out_x.reshape(-1, 1), out_y.reshape(-1, 1)), axis=1)


class AutoTrainer():
    def __init__(self,
                 changeList,
                 paramName,
                 mainFunc,
                 deviceList=[4, 5, 6, 7],
                 poolNumber=4,
                 name='AutoTrainer'):
        self.paramName = paramName
        self.mainFunc = mainFunc
        self.changeList = changeList
        self.deviceList = deviceList
        self.poolNumber = poolNumber
        self.name = name

        self.loopList = list(product(*tuple(changeList)))
        # print(self.loopList)
        # input()

    def Run(self, ):

        poolLeftNumber = self.poolNumber - 1
        # gpunum = 0
        for i, item in enumerate(self.loopList):

            txtDevice = "CUDA_VISIBLE_DEVICES={} ".format(
                self.deviceList[i % len(self.deviceList)])
            txtmain = 'python -u ' + self.mainFunc
            txtparam = ''
            for j, param in enumerate(self.paramName):
                txtparam += '--{} {} '.format(param, item[j])
            txtname = '--name ' + self.name + txtparam.replace(
                ' ', '_').replace('--', '_')

            txt = ' '.join([txtDevice, txtmain, txtparam, txtname])
            print(txt)
            # input()
            # os.system(txt)
            time.sleep(2)

            if poolLeftNumber == 0:
                print('continue left:', poolLeftNumber)
                poolLeftNumber = self.poolNumber - 1
                child = subprocess.Popen(txt, shell=True)
                child.wait()
                # subprocess.Popen()
            else:
                print('wait left:', poolLeftNumber)
                child = subprocess.Popen(txt, shell=True)
                # child.wait(2)
                poolLeftNumber -= 1
