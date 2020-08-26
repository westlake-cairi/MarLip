import torch
import numpy as np
from sklearn.datasets import make_s_curve
from samples_generator_new import make_swiss_roll

from torchvision import transforms
from torchvision import datasets as torchvisiondatasets

# shuffling the data to get an index of the individual batches
class SampleIndexGenerator():
    def __init__(self, data, batch_size):

        self.num_train_sample = data.shape[0]
        self.batch_size = batch_size
        self.Reset()

    def Reset(self):

        self.unuse_index = torch.randperm(self.num_train_sample).tolist()

    def CalSampleIndex(self, batch_idx):

        use_index = self.unuse_index[:self.batch_size]
        self.unuse_index = self.unuse_index[self.batch_size:]

        return use_index
     
def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Arguments:
        n {int} -- number of data points in shape
        r {float} -- radius of sphere
        ambient {int, default=None} -- Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data

def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)

    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # Additional big surrounding sphere:
    n_samples_big = 10*n_samples  # int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        # plt.savefig('look.png')
        # plt.show()

    # Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    index_seed = np.linspace(0, 10000, num=20, dtype='int16', endpoint=False)
    arr = np.array([], dtype='int16')
    for i in range(500):
        arr = np.concatenate((arr, index_seed+int(i)))
    # arr.astype(int)
    print(arr.shape)

    # rng.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22+0.5, labels


def create_sphere_dataset5500(n_samples=1500, d=100, bigR=25, n_spheres=11, r=5, seed=42):
    np.random.seed(42)

    # it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance = 10/np.sqrt(d)
    shift_matrix = np.random.normal(0, variance, [n_spheres, d+1])

    spheres = []
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i, :])
        n_datapoints += n_samples

    # additional big surrounding sphere
    n_samples_big = 1*n_samples
    big = dsphere(n=n_samples_big, d=d, r=bigR)
    spheres.append(big)
    n_datapoints += n_samples_big

    # create Dataset
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    dataset = dataset[arr]
    labels = labels[arr]

    return dataset/22 + 0.5, labels

def LoadData(data_name='SwissRoll', data_num=1500, seed=0, noise=0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), remove=None, test=False):

    """
    function used to load data

    Arguments:
        data_name {str} -- the dataset to be loaded
        data_num {int} -- the data number to be loaded
        seed {int} -- the seed for data generation
        noise {float} -- the noise for data generation
        device {torch} -- the device to store data
        remove {str} -- Shape of the points removed from the generated manifold
    """

    # Load SwissRoll Dataset
    if data_name == 'SwissRoll':
        if remove is None:
            train_data, train_label = make_swiss_roll(n_samples=data_num, noise=noise, random_state=seed)
        else:
            train_data, train_label = make_swiss_roll(n_samples=data_num, noise=noise, random_state=seed+1, remove=remove, center=[10, 10], r=8)
        train_data = train_data / 20

    # Load SCurve Dataset
    if data_name == 'SCurve':
        train_data, train_label = make_s_curve(n_samples=data_num, noise=noise, random_state=seed)
        train_data = train_data / 2

    # Load Mnist Dataset
    if data_name == 'MNIST':

        train_data = torchvisiondatasets.MNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).data.float().view(-1, 28*28)/255
        train_label = torchvisiondatasets.MNIST(
            '~/data', train=True, download=True,
            transform=transforms.ToTensor()
        ).targets

        if not test:
            train_data = train_data[:data_num]
            train_label = train_label[:data_num]
        else:
            train_data = train_data[data_num:data_num*2]
            train_label = train_label[data_num:data_num*2]

    if data_name == 'Spheres5500':
        train_data, train_label = create_sphere_dataset5500(seed=seed)
    if data_name == 'Spheres10000':
        train_data, train_label = create_sphere_dataset(seed=seed)

    # Put the data to device
    train_data = torch.tensor(train_data).to(device)[:data_num]
    train_label = torch.tensor(train_label).to(device)[:data_num]

    return train_data, train_label