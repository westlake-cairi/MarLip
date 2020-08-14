import torch
import torch.nn as nn

class MLDL_Loss(object):
    def __init__(self, args, k=5, cuda=True):

        self.args = args
        self.NetworkStructure = args['NetworkStructure']
        self.latent_index = 2 * len(args['NetworkStructure']) - 3

        self.device = cuda
        self.epoch = 0


    def SetEpoch(self, epoch):
        self.epoch = epoch


    def Epsilonball(self, data):

        """
        function used to calculate the distance between point pairs and determine the neighborhood

        Arguments:
            data {tensor} -- the train data

        Outputs:
            d {tensor} -- the distance between point pairs
            kNN_mask {tensor} a mask used to determine the neighborhood of every data point
        """

        Epcilon = self.args['Epcilon']

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-8).sqrt()

        kNN_mask = (d < Epcilon).bool()

        return d, kNN_mask


    # Using the reconstruction loss as Loss_ae
    def ReconstructionLoss(self, pred, target):
        criterion = nn.MSELoss().cuda()
        loss = criterion(pred, target)

        return loss


    def DistanceLoss(self, data, latent, dis_data, dis_latent, kNN_data, kNN_latent):

        """
        function used to calculate loss_iso and loss_push-away

        Arguments:
            data {tensor} -- the data for input layer data
            latent {tensor} -- the data for latent layer data
            dis_data {tensor} -- the distance between point pairs for input layer data
            dis_latent {tensor} -- the distance between point pairs for latent layer data
            kNN_data {tensor} -- the mask to determine the neighborhood for input layer data
            kNN_latent {tensor} -- the mask to determine the neighborhood for latent layer data
        """

        norml_data = torch.sqrt(torch.tensor(float(data.shape[1])))
        norml_latent = torch.sqrt(torch.tensor(float(latent.shape[1])))

        # Calculate Loss_iso
        D1_1 = (dis_data/norml_data)[kNN_data]
        D1_2 = (dis_latent/norml_latent)[kNN_data]
        Error1 = (D1_1 - D1_2) / 1
        loss_iso = torch.norm(Error1)/torch.sum(kNN_data)

        # Calculate Loss_push-away
        D2_1 = (dis_latent/norml_latent)[kNN_data == False]
        Error2 = (0 - D2_1) / 1
        loss_push_away = torch.norm(Error2[Error2 > -1 * self.args['RegularB']]) / torch.sum(kNN_data == False)

        # The gradual changing of weight for Loss_push-away 
        if self.epoch > self.args['GradualChanging'][0]:
            self.push_away = max(0.80 - (self.epoch - self.args['GradualChanging'][0]) / (self.args['GradualChanging'][1] - self.args['GradualChanging'][0]) * 0.80, 0)
        else:
            self.push_away = 0.80

        loss_push_away = -1.0 * self.push_away * loss_push_away

        return loss_iso, loss_push_away


    # Calculating the Angle Matrix
    def CossimiSlow(self, data):

        eps = 1e-8
        a_n, b_n = data.norm(dim=1)[:, None], data.norm(dim=1)[:, None]
        a_norm = data / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = data / torch.max(b_n, eps * torch.ones_like(b_n))
        out = torch.mm(a_norm, b_norm.transpose(0, 1))

        return out


    # Calculating the Loss_angle
    def AngleLossSlow(self, data, latent, kNN_data, kNN_latent):

        angle_loss = torch.zeros(data.shape[0], device=self.device)

        for i in range(data.shape[0]):
            center = data[i]
            other = data[kNN_data[i]] - center
            angle1 = self.CossimiSlow(other)

            center = latent[i]
            other = latent[kNN_data[i]] - center
            angle2 = self.CossimiSlow(other)

            angle_loss[i] = torch.norm(angle1 - angle2)/data.shape[0]
            
        return torch.mean(angle_loss)


    def MorphicLossItem(self, data, latent):

        dis_data, kNN_data  = self.Epsilonball(data)
        dis_latent, kNN_latent = self.Epsilonball(latent)
        loss_iso, loss_push_away = self.DistanceLoss(data, latent, dis_data, dis_latent, kNN_data, kNN_latent)

        if self.args['ratio'][2] < 0.01:
            loss_ang = loss_iso / 100000
        else:
            loss_ang = self.AngleLossSlow(data, latent, kNN_data, kNN_latent)

        return loss_iso, loss_ang, loss_push_away


    def CalLosses(self, train_info):
        
        """
        function used to calculate four losses

        Arguments:
            train_info {tensor} -- results for each intermediate layer in the network

        Outputs:
            loss_list {list} -- four losses: loss_ae, loss_iso, loss_angle, loss_push-away
        """

        train_info[0] = train_info[0].view(train_info[0].shape[0], -1)
        loss_ae = self.ReconstructionLoss(train_info[0], train_info[-1])
        loss_ae += self.ReconstructionLoss(train_info[2], train_info[-2])
        loss_ae += self.ReconstructionLoss(train_info[4], train_info[-4])
        loss_ae += self.ReconstructionLoss(train_info[6], train_info[-6])
        loss_ae += self.ReconstructionLoss(train_info[8], train_info[-8])
        loss_distance, loss_ang, loss_mutex = self.MorphicLossItem(train_info[0], train_info[self.latent_index])      

        loss_list = [loss_ae, loss_distance, loss_ang, loss_mutex]

        # Weights for losses
        for i in range(len(loss_list)):
            loss_list[i] *= self.args['ratio'][i]

        return loss_list