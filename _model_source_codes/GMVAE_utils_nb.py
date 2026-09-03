import torch
from torch import optim
from torch.nn import functional as F
import math
import argparse
import os
import sys
from GMVAE_losses_nb import *


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv')!=-1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm')!=-1:
		m.weight.data.normal_(1.0, 0.02)
		# m.bias.data.fill_(0)


# write strings to file
def writeStatusToFile(filepath, status):
	with open(filepath, 'a') as file:
		file.write(status)



def train(epoch, gmvae, train_loader, optimizer):
	num_of_batch = len(train_loader)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	gmvae.train()

	total_loss_epoch = 0
	KLD_gaussian_epoch = 0
	KLD_pi_epoch = 0
	nb_loss_epoch = 0
	acc_count_epoch = 0
	data_count_epoch = 0

	for batch_idx, (data, target) in enumerate(train_loader):

		data = data.to(device)
		target = target.to(device)
		optimizer.zero_grad()
		batchsize = data.size(0)

		pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_mean, x_recons_disper, \
        x_recon_mean, x_recon_disper =  gmvae(data, target)

		total_loss, KLD_gaussian, KLD_pi, nb_loss \
      = gmvae_losses(data, target, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,\
    x_recons_mean, x_recons_disper)

		acc_count = torch.sum(torch.argmax(pi_x, dim=1)==target)

		total_loss_epoch += total_loss/batchsize
		KLD_gaussian_epoch += KLD_gaussian/batchsize
		KLD_pi_epoch += KLD_pi/batchsize
		nb_loss_epoch += nb_loss/batchsize
		acc_count_epoch += acc_count
		data_count_epoch += data.size(0)

		total_loss.backward()
		optimizer.step()

	accuracy = acc_count_epoch/data_count_epoch
	return total_loss_epoch/num_of_batch, KLD_gaussian_epoch/num_of_batch, KLD_pi_epoch/num_of_batch, \
     nb_loss_epoch/num_of_batch, accuracy



def test(epoch, gmvae, test_loader, optimizer):
	num_of_batch = len(test_loader)
	gmvae.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	total_loss_test = 0
	KLD_gaussian_test = 0
	KLD_pi_test = 0
	nb_loss_test = 0
	acc_count_test = 0
	data_count_test = 0

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data = data.to(device)
			target = target.to(device)
			batchsize = data.size(0)

			pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_mean, x_recons_disper, \
            x_recon_mean, x_recon_disper =  gmvae(data, target)

			total_loss, KLD_gaussian, KLD_pi, nb_loss = \
				gmvae_losses(data, target, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, \
        x_recons_mean, x_recons_disper)

			acc_count = torch.sum(torch.argmax(pi_x, dim=1)==target)

			total_loss_test += total_loss/batchsize
			KLD_gaussian_test += KLD_gaussian/batchsize
			KLD_pi_test += KLD_pi/batchsize
			nb_loss_test += nb_loss/batchsize
			acc_count_test += acc_count
			data_count_test += data.size(0)

	accuracy = acc_count_test/data_count_test
	return total_loss_test/num_of_batch, KLD_gaussian_test/num_of_batch, \
     KLD_pi_test/num_of_batch, nb_loss_test/num_of_batch, accuracy
