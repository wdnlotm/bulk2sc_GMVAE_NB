import torch
from torch import nn
from torch.nn import functional as F

def gmvae_losses(Xdata, Xtarget, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,\
    x_recons_mean, x_recons_disper):
	#pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,
	# x_recons_mean, x_recons_disper, x_recon_mean, x_recon_disper
	eps = 1e-10
	batchsize = Xdata.size(0)
	K = pi_x.size(1)
	z_dim = mu_zs.size(1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	Xdata = Xdata.to(device)
	pi_target = torch.nn.functional.one_hot(Xtarget, num_classes=K).to(device)
	
	KLD_pi = torch.sum(pi_x*(torch.log(pi_x + eps)-torch.log(pi_target + eps)))
	
	KLD_gaussian = 0.5 * (((logvar_genz - logvar_zs) + \
		((logvar_zs.exp() + (mu_zs - mu_genz).pow(2))/logvar_genz.exp())) - 1)
	
	KLD_gaussian = torch.sum(KLD_gaussian * pi_x.unsqueeze(1).expand(batchsize,z_dim,K))
	
	pi_x_expanded = pi_x.unsqueeze(1).expand(batchsize, Xdata.shape[-1], K)

	###################
	# nb_loss_i
	# Negative log-likelihood of the negative binomial, parameterized by
	# mean (mu) and dispersion (theta):
	#   -log p(x) = lgamma(x+1) + lgamma(theta) - lgamma(x+theta)
	#               - theta*log(theta) - x*log(mu)
	#               + (theta+x)*log(theta+mu)
	# Unlike ZINB, this is valid for x == 0 as well, so no case split is needed.
	gamma_terms_i = torch.stack([(torch.lgamma(Xdata + 1) + torch.lgamma(x_recons_disper[:,:,ii] + eps) \
        + (-1.0)*torch.lgamma(Xdata + x_recons_disper[:,:,ii] + eps)) for ii in range(K)], \
            dim=x_recons_disper.dim()-1)

	rate_terms_i = torch.stack([((-1.0)*x_recons_disper[:,:,ii]*torch.log(x_recons_disper[:,:,ii] + eps) \
        + (-1.0)*Xdata*torch.log(x_recons_mean[:,:,ii] + eps) \
            + (x_recons_disper[:,:,ii] + Xdata)*torch.log(x_recons_disper[:,:,ii] + x_recons_mean[:,:,ii] + eps)) \
                for ii in range(K)], dim=x_recons_disper.dim()-1)

	nb_losses = gamma_terms_i + rate_terms_i

	#Sum them up
	nb_loss = torch.sum(nb_losses*pi_x_expanded)

	total_loss = KLD_gaussian + KLD_pi + nb_loss

	return total_loss, KLD_gaussian, KLD_pi, nb_loss
