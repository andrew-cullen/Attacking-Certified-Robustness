import time

from statsmodels.stats.proportion import multinomial_proportions_confint as multi_conf
from statsmodels.stats.proportion import proportion_confint as binom_conf

import numpy as np

import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.utils

from scipy.stats import norm as stats_norm

from autoattack import AutoAttack

from logging_manager import LogSet

CUDA_AVAILABLE = torch.cuda.device_count() > 0

    
###############################################################
# Utilities
###############################################################

class WrapperModule(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, model, samples, sigma):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(WrapperModule, self).__init__()
        self.model = model
        self.samples = samples
        self.sigma = sigma

    def forward(self, inpt: torch.tensor):
        if inpt.shape[1] == 1:
            factor = 3
        else:
            factor = 1
        adv_input = inpt.repeat(self.samples, factor, 1, 1)
        adv_input = adv_input + torch.randn_like(adv_input) * self.sigma

        gs = F.gumbel_softmax(10*self.model(adv_input), tau=1, hard=True, dim=1) 

        sf = torch.sum(gs, dim=0).reshape(1, -1)
        return (sf[0,:] / self.samples).reshape(1,-1)  

def basic_predict(model, images, device):
    images = images.detach().clone()

    _, _, pred_baseline = return_wrapper(model, images, device)

    indices, counts = torch.unique(pred_baseline, sorted=True, return_counts=True)
    pred_baseline = indices[torch.argmax(counts)]

    return pred_baseline.detach()


def return_wrapper(model, inpt, device, validate=False):
    samples = inpt.shape[0]

    if inpt.shape[1] == 1:
        inpt = inpt.repeat(1, 3, 1, 1)

    direct_out = model(inpt)
    if validate:
        
        ix = torch.argmax(direct_out, dim=1)
        indices, counts = torch.unique(ix, return_counts=True)
        return indices, counts, ix
    else:
        gs = F.gumbel_softmax(100 * direct_out, tau=1, hard=True, dim=1)
        sf = torch.sum(gs, dim=0).reshape(1, -1)

        return (sf[0, :] / samples), None, (torch.argmax(gs, dim=1))#.to(device)


def norm_distance(a, b, flag):
    if flag:
        return torch.linalg.norm(a[:, 0, :, :] - b[:, 0, :, :])
    else:
        return torch.linalg.norm(a - b)

def return_cohen(yp, samples, model, device, alpha, sigma):
    # CR as per Cohen
    relu = torch.nn.ReLU()
    norm = torch.distributions.Normal(0, 1)

    batched = True

    if batched:
        samples_subset = 100
        batches = int(samples / samples_subset)
        samples = batches * samples_subset

    else:
        batches = 1
        samples_subset = samples

    sm_output = None

    for _ in range(batches):
        adv_input = yp.repeat(samples_subset, 1, 1, 1)
        adv_input += torch.randn_like(adv_input) * sigma

        if adv_input.shape[1] != 3:
            adv_input = adv_input.repeat(1, 3, 1, 1)

        sm_output_temp, _, _ = return_wrapper(model, adv_input, device)  
        if sm_output is None:
            sm_output = sm_output_temp * ( samples_subset / samples)
        else:
            sm_output += sm_output_temp * ( samples_subset / samples)
        

    vals, indices = torch.topk(sm_output, 2, sorted=True)

    max_class = indices[0]

    E0, E1 = vals[0], vals[1]

    E0_t, E0_u_t = binom_conf(
        E0.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
    )
    E1_l_t, E1_t = binom_conf(
        E1.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
    )  # 1 - E_0
    E0_v, E1_v = E0.detach().cpu().numpy(), E1.detach().cpu().numpy()
    E0_t, E0_u_t = E0_t - E0_v, E0_u_t - E0_v
    E1_l_t, E1_t = E1_l_t - E1_v, E1_t - E1_v

    E0, E0_u, E1_l, E1 = (
        E0 + torch.tensor(E0_t).to(device),
        E0 + torch.tensor(E0_u_t).to(device),
        E1 + torch.tensor(E1_l_t).to(device),
        E1 + torch.tensor(E1_t).to(device),
    )

    cohen_val = relu(0.5 * sigma * (norm.icdf(relu(E0)) - norm.icdf(relu(E1))))
    
    return E0, E1, cohen_val, max_class



############################################
# Attacks
############################################

# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# Modified to attack class difference on expectations
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
# Based upon https://github.com/Harry24k/CW-pytorch/blob/master/CW.ipynb
def cw_l2_attack(
    model,
    images,
    labels,
    samples,
    sigma,
    device,
    targeted=False,
    c=1e-4,
    kappa=0,
    max_iter=100,
    learning_rate=0.01,
    printing=False,
    z_val=4.0,
    definitive=False,
):
    # Original version was having memory issues
    inpt_flag = True if images.shape[1] == 1 else False

    start_time = time.time()

    if definitive is not False:
        alpha = 0.5 * (1 - stats_norm.cdf(z_val))

    images = images.to(device)
    base_images = images.detach().clone()
    labels = labels.to(device)

    flag = False
    min_val, min_recorded = 1e6, None    

    # Define f-function
    def f(x, samples, sigma):

        adv_input = x.repeat(samples, 1, 1, 1)
        adv_input += torch.randn_like(adv_input) * sigma

        if adv_input.shape[1] != 3:
            adv_input = adv_input.repeat(1, 3, 1, 1)

        outputs, _, pred_class = return_wrapper(model, adv_input, device)
        pred_class = pred_class.detach().clone().cpu()

        one_hot_labels = torch.eye(outputs.shape[0]).to(device)
        one_hot_labels = one_hot_labels[labels]

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa), outputs.detach().clone(), pred_class
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa), outputs.detach().clone(), pred_class
        
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    
    for _ in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)        

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        f_loss, gs_outputs, class_pred = f(a, samples, sigma)
        class_pred = torch.mode(class_pred, 0)[0].detach().cpu()
        loss2 = torch.sum(c*f_loss)

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if class_pred != labels[0]:
            vals, _ = torch.topk(gs_outputs, 2, sorted=True)

            E0, E1 = vals[0], vals[1]  

            E_0 = binom_conf(
                E0.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
            )[0]
            E_1 = binom_conf(
                E1.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
            )[
                1
            ]  # 1 - E_0
            E_0, E_1 = torch.tensor(E_0), torch.tensor(E_1)     

            if E_0 > E_1:
                a_comparison = a.detach().clone()
                if inpt_flag:
                    a_comparison = a_comparison[:,0,:,:].reshape(-1)
                else:
                    a_comparison = a_comparison.reshape(-1)
                
                min_temp = torch.linalg.norm((a_comparison - base_images.reshape(-1)).reshape(-1), 2).item()
                if min_temp < min_val:
                    min_val = min_temp
                    final_cw_time = time.time() - start_time
                    min_recorded = a.detach().clone()
                    if flag is False:
                        first_cw = min_temp
                        first_time = time.time() - start_time
                        flag = True

    if flag is False:
        final_cw_time = time.time() - start_time
        first_cw = min_val
        first_time = final_cw_time                           

    return (
        min_recorded,
        flag,
        min_val,
        min_recorded,
        first_cw,
        first_time,
        final_cw_time,
    )


def direct_attack(
    model,
    yp,
    images,
    samples,
    class_set,
    z_val,
    min_recorded,
    sigma,
    device,
    inpt_flag,
    yp_best=None,
    lambda_val=0.5,
    delta=None,
    cutoff_step=0.5,
):
    stepsize = 0.05
    printing = False

    device = yp.device

    relu = torch.nn.ReLU()
    norm = torch.distributions.Normal(0, 1)

    alpha = 0.5 * (1 - stats_norm.cdf(z_val))

    yp.detach_()
    yp.requires_grad = True

    adv_input = yp.repeat(samples, 1, 1, 1)
    adv_input += torch.randn_like(adv_input) * sigma

    if adv_input.shape[1] != 3:
        adv_input = adv_input.repeat(1, 3, 1, 1)

    sm_output, _, _ = return_wrapper(model, adv_input, device)

    vals, indices = torch.topk(sm_output, 2, sorted=True)

    max_class, second_class = indices[0], indices[1]

    E0, E1 = vals[0], vals[1]

    E0_t, E0_u_t = binom_conf(
        E0.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
    )
    E1_l_t, E1_t = binom_conf(
        E1.detach().cpu().numpy() * samples, samples, alpha=alpha, method="beta"
    )  # 1 - E_0
    E0_v, E1_v = E0.detach().cpu().numpy(), E1.detach().cpu().numpy()
    E0_t, E0_u_t = E0_t - E0_v, E0_u_t - E0_v
    E1_l_t, E1_t = E1_l_t - E1_v, E1_t - E1_v

    E0, E0_u, E1_l, E1 = (
        E0 + torch.tensor(E0_t).to(device),
        E0 + torch.tensor(E0_u_t).to(device),
        E1 + torch.tensor(E1_l_t).to(device),
        E1 + torch.tensor(E1_t).to(device),
    )

    cohen_val = relu(0.5 * sigma * (norm.icdf(relu(E0)) - norm.icdf(relu(E1))))

    # What are we looking for:
    # We want
    # - Predicted class to not be equal to class_set[0]
    # - If the above is true, then we also want E0 > E1 - ie that the lower bound of the predicted class is strictly above the upper bound of the secondary class
    # So to do this we want to minimize torch.abs(E0 - E1 - delta) when the classes don't match. When the classes do match, we want to minimize torch.abs(E0 - E1 + delta), where delta > 0
    # Does this work though? Because E0 == E1 is just saying that
    if delta is None:
        delta = torch.max(torch.abs(E0 - E0_u), torch.abs(E1_l - E1))
    delta = 0  # Just trying this out
    if max_class == class_set[0]:
        # Predction class matches. So we want to decrease the delta between the classes, and we want to penalise moving away from the origin.
        # We're minimizing. So we minimize the delta while moving further away from the origin?
        #objective_recorded = torch.abs(E0 - E1 - 0.01)  # + delta)
        objective_recorded = E0 - E1
        cutoff_step = 0.75

        stepsize = torch.min(
            torch.max(1.05 * cohen_val, torch.tensor(0.15)), torch.tensor(cutoff_step)
        )
        if printing:
            print(E0 - E1, "#" * 5)
    else:
        # Now the prediction classes don't match. So now we want to bring it towards the point where the non-class prediction is only just higher than the non-class. So we want to minimize torch.abs(E0_u
        # We want non-class lower bound to be striclty above label-class upper bound. So it should still be (E0 - E1), but now we penalise moving away

        # If in this bracket then E0 is not the max_class. So we want E0 > E1 while minimising this difference. So E0 - E1 > 0, so minimise

        # objective_recorded = torch.abs(E0 - E1 - delta) + lambda_val * torch.linalg.norm(yp-images) #+ torch.linalg.norm(yp - images) # Now we want to bring them closer together, while penalising increasing the delta

        if printing:
            print(E0 - E1, torch.linalg.norm(yp - images), lambda_val, delta, "!" * 5)
        if (E0 < E1):  # Adversarial example but not confident, then still prioritise the adversarial example
            #print('Branch B-1', flush=True)
            objective_recorded = -(E0 - E1) # We want to maximise this now, because we still haven't reached a true adversarial attack
            stepsize = torch.tensor(0.1) #torch.tensor(2.*cutoff_step) #torch.tensor(cutoff_step / 2)
        else:
            #print('Branch B-2', flush=True)        
            # Adversarial example found where E0 > E1, so in other words it's a clear, distinct adversarial example

            objective_recorded = 1*norm_distance(
                yp, images, inpt_flag
            )
            stepsize = torch.min(
                torch.max(cohen_val, torch.tensor(0.05)),
                torch.tensor(cutoff_step),
            )
            stepsize = torch.max(cohen_val, torch.tensor(0.05))
            lambda_val *= 1.15

            distance = norm_distance(yp, images, inpt_flag)
            if distance < min_recorded:
                min_recorded = distance
                yp_best = yp.detach().clone()
            

    yp_grad = torch.autograd.grad(objective_recorded, yp)[0].detach()
    yp_grad = yp_grad / (1e-5 + torch.linalg.norm(yp_grad))

    if (torch.isnan(torch.sum(yp_grad)) == 0) and ((E0 - E1) < (1 - 1e-5)):
        yp_new = yp - stepsize * yp_grad
    else:
        yp_new = yp + 0.05 * torch.randn_like(yp)

    yp_new = torch.clip(yp_new, 0, 1).detach()
    yp_new.requires_grad = True

    return yp_new, yp_grad, min_recorded, yp_best, lambda_val, delta

def deepfool_attack(
    model,
    images,
    label,
    device,
    classes,
    samples,
    sigma,
    overshoot=0.02,
    nb_candidate=10,
    max_iter=100,
):
    # DeepFool modified for attacks against CR
    start_time = time.time()
    ori_images = images.detach()
    nb_candidate = np.min((nb_candidate, classes))

    images.requires_grad_()

    if ori_images.shape[1] != 3:
        images = images.repeat(1, 3, 1, 1)

    adv_input = images.repeat(samples, 1, 1, 1)
    adv_input = adv_input + torch.randn_like(adv_input) * sigma

    logits, _, _ = return_wrapper(model, adv_input, device)

    pred = torch.argmax(logits)

    w = torch.squeeze(torch.zeros(adv_input.size()[1:])).to(device)
    r_tot = torch.zeros(images.size()).to(device)

    iteration = 0


    while (pred == label) and iteration < max_iter:
        predictions_val = torch.topk(logits, nb_candidate)[0]
        gradients = torch.stack(jacobian(predictions_val, images, nb_candidate), dim=1)
        with torch.no_grad():
            pert = 1e10
            if pred != label:
                continue
            for k in range(1, nb_candidate):
                w_k = gradients[0, k, ...] - gradients[0, 0, ...]
                f_k = predictions_val[k] - predictions_val[0]
                pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert * w / w.view(-1).norm()
            r_tot += r_i

        if torch.sum(torch.isnan(r_tot)) > 0:
            return False, None, 0.0, time.time() - start_time
        images = torch.clamp(r_tot + images, 0, 1).detach().requires_grad_()

        if ori_images.shape[1] != 3:
            images = images.mean(dim=1).repeat(1, 3, 1, 1)

        adv_input = images.repeat(samples, 1, 1, 1)
        adv_input = adv_input + torch.randn_like(adv_input) * sigma

        logits, _, _ = return_wrapper(model, adv_input, device)
        pred = torch.argmax(logits)

        iteration = iteration + 1

    adv_x = torch.clamp((1 + overshoot) * r_tot + images, 0.0, 1.0)
    distance = torch.linalg.norm(adv_x - ori_images).detach().cpu().numpy()
    flag = False
    if pred != label:
        flag = True

    return flag, adv_x, distance, time.time() - start_time


def jacobian(predictions, x, classes):
    list_derivatives = []

    for class_ind in range(classes):
        outputs = predictions[class_ind]  # [:, class_ind]
        (derivatives,) = torch.autograd.grad(
            outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True
        )
        list_derivatives.append(derivatives)

    return list_derivatives


def pgd_attack(
    model,
    images,
    labels,
    samples,
    sigma,
    device,
    epsilon=20 / 255,
    iters=40,
    z_val=4,
    probabilistic=False,
):
    # This is now the Iterative Fast Gradient Method for L2 Norms, modified for attacks against CR
    inpt_flag = True if images.shape[1] == 1 else False
    start_time = time.time()
    first_flag = False

    device = images.device
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    alpha_probability = 0.5 * (1 - stats_norm.cdf(z_val))

    ori_images = images.data.detach()
    flag = False
    adv_input = images.repeat(samples, 1, 1, 1)
    adv_input += torch.randn_like(adv_input) * sigma

    if adv_input.shape[1] != 3:
        adv_input = adv_input.repeat(1, 3, 1, 1)

    sm, _, _ = return_wrapper(model, adv_input, device)

    min_val = torch.tensor(1e6)
    min_recorded = None
    for _ in range(iters):
        images.requires_grad = True

        adv_input = images.repeat(samples, 1, 1, 1)
        adv_input += torch.randn_like(adv_input) * sigma

        if adv_input.shape[1] != 3:
            adv_input = adv_input.repeat(1, 3, 1, 1)

        sm, _, _ = return_wrapper(model, adv_input, device)
        pred = torch.argmax(sm)
        if pred != labels[0]:
            second_class = sm.detach().clone()
            second_class[pred] = 0
            second_class = torch.argmax(second_class)
            pred, second_class = (
                pred.detach().cpu().numpy(),
                second_class.detach().cpu().numpy(),
            )
            m_c = multi_conf(
                samples * sm.detach().cpu().numpy(), alpha=alpha_probability
            )
            lab = labels[0].detach().cpu().numpy()

            if m_c[pred, 0] > m_c[lab, 1]:
                flag = True
                delta = norm_distance(images, ori_images, inpt_flag)
                if delta < min_val:
                    final_pgd_time = time.time() - start_time
                    min_val = delta
                    min_recorded = images.detach()
                    if first_flag is False:
                        first_pgd = min_val.detach().clone().cpu().numpy()
                        first_time = time.time() - start_time
                        first_flag = True

        model.zero_grad()
        cost = loss(sm.reshape(1, -1), labels).to(device)
        cost.backward()

        grad = images.grad
        grad_norm = torch.linalg.norm(grad)
        images = torch.clamp(
            images.detach() + epsilon * (grad / (grad_norm + 1e-12)), min=0, max=1
        ).detach_()

    if first_flag is False:
        final_pgd_time = time.time() - start_time
        first_time = final_pgd_time
        first_pgd = min_val.detach().cpu().numpy()

    return (
        images,
        flag,
        pred,
        min_val.detach().cpu().numpy(),
        min_recorded,
        first_pgd,
        first_time,
        final_pgd_time,
    )
    
def pgd_attack_base(
    model,
    images,
    labels,
    samples,
    sigma,
    device,
    epsilon=20 / 255,
    iters=40,
    z_val=4,
    probabilistic=False,
):
    # This is now the Iterative Fast Gradient Method for L2 Norms, modified for attacks against CR
    inpt_flag = True if images.shape[1] == 1 else False
    start_time = time.time()
    first_flag = False

    device = images.device
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    samples = 1
    sigma = 0

    ori_images = images.data.detach()
    flag = False
    adv_input = images.repeat(samples, 1, 1, 1)
    adv_input += torch.randn_like(adv_input) * sigma

    if adv_input.shape[1] != 3:
        adv_input = adv_input.repeat(1, 3, 1, 1)

    sm = model(adv_input)

    min_val = torch.tensor(1e6)
    min_recorded = None
    for _ in range(iters):
        images.requires_grad = True

        adv_input = images.repeat(samples, 1, 1, 1)
        adv_input += torch.randn_like(adv_input) * sigma

        if adv_input.shape[1] != 3:
            adv_input = adv_input.repeat(1, 3, 1, 1)

        sm = model(adv_input.to(device))
        pred = torch.argmax(sm)
        if pred != labels[0]:
            flag = True
            delta = norm_distance(images, ori_images, inpt_flag)
            if delta < min_val:
                final_pgd_time = time.time() - start_time
                min_val = delta
                min_recorded = images.detach()
                if first_flag is False:
                    first_pgd = min_val.detach().clone().cpu().numpy()
                    first_time = time.time() - start_time
                    first_flag = True

        model.zero_grad()
        cost = loss(sm.reshape(1, -1), labels).to(device)
        cost.backward()

        grad = images.grad
        grad_norm = torch.linalg.norm(grad)
        images = torch.clamp(
            images.detach() + epsilon * (grad / (grad_norm + 1e-12)), min=0, max=1
        ).detach_()

    if first_flag is False:
        final_pgd_time = time.time() - start_time
        first_time = final_pgd_time
        first_pgd = min_val.detach().cpu().numpy()

    return (
        images,
        flag,
        pred,
        min_val.detach().cpu().numpy(),
        min_recorded,
        first_pgd,
        first_time,
        final_pgd_time,
    )    
 

def find_step(v0, s, d, x_set, r_set):
    with torch.no_grad():
        v = v0 - s*d
        
        for (centre, r) in zip(*[x_set, r_set]):
            delta_v = v - v0
            delta_c = centre - v0

            a, b, c = (delta_v**2).sum(), -2 * (delta_v * delta_c).sum(), (delta_c**2).sum() - r**2 
            delta = b**2 - 4*a*c            
            if delta > 0:
                t = torch.tensor([(-b + delta.sqrt()) / 2*a, (-b - delta.sqrt()) / 2*a ])
                t_min, t_max = torch.min(t), torch.max(t)
                if (t_max > 1.01) and (t_min < 0.99):
                    s = -1.*t_max*delta_v / (d) 
                    return find_step(v0, s, d, x_set, r_set)
        return s
        
def find_smallest(images, x, r, min_step, max_step):
    # To minimise L2 distance between clean sample and adversarial example while remaining within the adversarial example CR
    with torch.no_grad():
        vector = (images - x)
        vector /= (vector**2).sum().sqrt()
        if (0.95*r < min_step) and (r > min_step):
            r = min_step
        else:
            r = 0.95 * r
        yp = x + torch.min(r, max_step) * vector 
        
        return yp

def direct_new(
    model, labels, samples, images, sigma, device, alpha, iters=100, stationary_count=10, min_step=0.075, max_step=1000, delta=0.05
):
    min_recorded, best_time, first_val, first_time = np.nan, np.nan, np.nan, np.nan
    min_step, max_step, delta = torch.tensor(min_step).to(device), torch.tensor(max_step).to(device), torch.tensor(delta).to(device)
    start_time = time.time()
    success = False
    min_recorded = 1000
    stationary_counter = 0

    #yp_best, lambda_val, delta_val = None, 0.5, None
    inpt_flag = True if images.shape[1] == 1 else False    

    x_set, r_set = [], []
    yp = images.detach().clone()
    

    for iter_count in range(iters):       
        model.zero_grad()

        yp.detach_()
        
        yp.requires_grad = True

        E0, E1, cohen_val, max_class = return_cohen(yp, samples, model, device, alpha, sigma)

        if max_class == labels:
            # Code has not yet identified an adversarial example 
            x_set.append(yp.detach().clone())
            r_set.append(cohen_val)
            yp_grad = torch.autograd.grad(E0 - E1, yp)[0].detach()
            with torch.no_grad():
                yp_grad = yp_grad / (1e-5 + torch.linalg.norm(yp_grad))
                
                
                s = find_step(yp, cohen_val, yp_grad, x_set, r_set)
                s = torch.clip(s*(1+delta), min_step, max_step)
                yp = (yp - s*yp_grad).clip(0., 1.)
                                   
        elif E0 < E1:
            # Code has not yet identified an adversarial example, but is no longer able to certify
            yp_grad = torch.autograd.grad(E0 - E1, yp)[0].detach()
            with torch.no_grad():
                yp_grad = yp_grad / (1e-5 + torch.linalg.norm(yp_grad))

                stepsize = 4 * min_step 
                if stepsize > max_step:
                    stepsize = max_step                
                yp = (yp + stepsize*yp_grad).clip(0.,1.) 
        else: 
            # Adversarial example has been identified
            with torch.no_grad():               
                yp = find_smallest(images, yp, cohen_val, min_step, max_step).clip(0., 1.)

                distance = norm_distance(yp, images, inpt_flag)
                if distance < min_recorded:
                    if min_recorded > 999:
                        success = True
                        first_time = time.time() - start_time
                        first_val = distance.detach().clone()
                    min_recorded = distance.detach().clone().item()
                    yp_best = yp.detach().clone()           
                    best_time = time.time() - start_time     
                elif min_recorded < 1000:
                    stationary_counter += 1
                    if stationary_counter == stationary_count:
                        break
                if cohen_val < 1e-2:
                    break

        yp.requires_grad = True            
    if success:        
        return min_recorded, best_time, first_val, first_time, success, yp_best
    else:
        return min_recorded, best_time, first_val, first_time, success, 0
    


# Outer process for attacks using our new process
def direct_loop(
    model,
    classes,
    labels,
    samples,
    yp,
    images,
    z_val,
    sigma,
    device,
    iters=100,
    new=True,
    cutoff_step=0.5,
):
    start_time = time.time()
    first_flag = False

    class_set = torch.zeros(classes).to(device)

    class_set[0] = labels[0]
    class_dummy = torch.arange(classes).to(device)

    class_set[1:] = class_dummy[class_dummy != labels[0]]#[0]]
    min_recorded = 1000

    stationary_counter = 0
    sample_size = samples
    yp_best, lambda_val, delta_val = None, 0.5, None
    inpt_flag = True if images.shape[1] == 1 else False
    for _ in range(iters):
        min_recorded_old = min_recorded
        if new:
            yp, _, min_recorded, yp_best, lambda_val, delta_val = direct_attack(
                model,
                yp,
                images,
                sample_size,
                class_set,
                z_val,
                min_recorded,
                sigma,
                device,
                inpt_flag,
                yp_best=yp_best,
                lambda_val=lambda_val,
                delta=delta_val,
                cutoff_step=cutoff_step,
            )
        else:
            (
                yp,
                _,
                min_recorded,
                yp_best,
                lambda_val,
                delta_val,
            ) = direct_attack_old(
                model,
                yp,
                images,
                sample_size,
                class_set,
                z_val,
                min_recorded,
                sigma,
                device,
                inpt_flag,
                yp_best=yp_best,
                lambda_val=lambda_val,
                delta=delta_val,
                cutoff_step=cutoff_step,
            )

        if (min_recorded_old - 1e-5) > min_recorded:
            new_d_time = time.time() - start_time

        if min_recorded < 1000:
            if first_flag is False:
                first_time = time.time() - start_time
                first_radii = min_recorded.detach().cpu().numpy()
                first_flag = True
            if stationary_counter == 0:
                sample_size = samples
            if torch.abs(min_recorded - min_recorded_old) < 1e-5:
                stationary_counter += 1
            else:
                stationary_counter = 0

    if min_recorded >= 1000:
        new_d_time = time.time() - start_time
        first_time = new_d_time
        first_radii = min_recorded

    return new_d_time, min_recorded, first_time, first_radii


############################################
# Core Management
############################################


class Evaluation(object):
    def __init__(self, dataset_name, classes, device, model, test_loader, sigma, samples, total_cutoff=250, autoattack_radii=-1, pgd_radii=20/255, cw=True, new=True, fool=True, pgd=True, ablation=False, autoattack=True, new_min_step=0.01, new_max_step=0.125, delta_val=0.025, filename=None, start_point=0):
        self.classes, self.device, self.model, self.test_loader, self.sigma, self.samples, self.total_cutoff, self.autoattack_radii, self.pgd_radii, self.delta_val = classes, device, model, test_loader, sigma, samples, total_cutoff, autoattack_radii, pgd_radii, delta_val
        self.start_point = start_point
        self.dataset = dataset_name
        
        self.save_images = False

        if dataset_name == 'imagenet':
            autoattack = False

        self.z_val = 2.58
        self.alpha = 0.5 * (1 - stats_norm.cdf(self.z_val))
        self.norm = torch.distributions.Normal(0, 1)

        self.cw, self.new, self.fool, self.pgd, self.autoattack, self.ablation, self.min_step, self.max_step = cw, new, fool, pgd, autoattack, ablation, new_min_step, new_max_step
        self.lagrange = True
        
        self.image_count = 0

        if filename is None:
            filename = dataset_name

        self.log = LogSet(
            filename + "-" + str(sigma) + "-samples-" + str(samples) + "-" + str(autoattack_radii) + "-" + str(pgd_radii),
            decimals=4,
            means=False,
            console=False,
        )
    def evaluate(self):
        total, overall_count = 0, 0

        for surrogate_count, (images, labels) in enumerate(self.test_loader):
            self.image_count += 1

            if surrogate_count >= self.start_point:
                overall_count += 1
                if total > self.total_cutoff:
                    break
                
                images, labels = images.to(self.device), torch.tensor(labels).to(self.device).reshape(-1)

                if len(images.shape) == 3:
                    images = images.reshape(1, images.shape[0], images.shape[1], images.shape[2])

                inference_time = time.time()
                adv_input = images.repeat(self.samples, 1, 1, 1)

                adv_input += torch.randn_like(adv_input) * self.sigma
                pred_baseline = basic_predict(self.model, adv_input, self.device)

                inference_time = time.time() - inference_time
                print(f'{pred_baseline.device} and {labels.device} and {self.device}', flush=True)
                if pred_baseline == labels:
                    total += 1 

                    self.log.append({"ix": total, "rej": overall_count - total, "lab": labels, "inf_t": inference_time}, None)

                    self.certify_cohen(adv_input)
                    if self.cw:
                        self.evaluate_cw(images, labels)
                        if self.ablation:
                            if self.dataset != 'imagenet':
                                self.evaluate_cw_ablation(images, labels)
                    if self.new:
                        #self.evaluate_new(images, labels)
                        self.evaluate_new_v2(images, labels)
                        if self.ablation:
                            self.evaluate_new_ablation(images, labels)
                            #self.evaluate_new_delta_ablation(images, labels)                        
                    if self.fool:
                        self.evaluate_deepfool(images, labels)
                    if self.pgd:
                        self.evaluate_pgd(images, labels)
                        if self.ablation:
                            #if self.dataset != 'imagenet':
                            self.evaluate_pgd_ablation(images, labels)
                            self.evaluate_pgd_ablation_base(images, labels)
                    if self.autoattack:
                        self.evaluate_autoattack(adv_input, images, labels)
                    if self.lagrange:
                        self.evaluate_lagrange(images, labels)
                    self.log.print()

    def certify_cohen(self, adv_input):
        cohen_time = time.time()

        alpha = 0.5 * (1 - stats_norm.cdf(self.z_val))
        sm_output, _, _ = return_wrapper(self.model, adv_input, self.device)
        vals, _ = torch.topk(sm_output, 2, sorted=True)

        max_count, second_count = (
            vals[0].detach().cpu().numpy() * self.samples,
            vals[1].detach().cpu().numpy() * self.samples,
        )

        E_0c = binom_conf(max_count, self.samples, alpha=alpha, method="beta")[0]
        E_1c = binom_conf(second_count, self.samples, alpha=alpha, method="beta")[1]

        baseline_cohen = np.max(
            [
                (0.5 * self.sigma * (self.norm.icdf(torch.tensor(E_0c)) - self.norm.icdf(torch.tensor(E_1c)))).detach().cpu().numpy(),
                0,
            ]
        )
        cohen_time = time.time() - cohen_time

        self.log.append({"co_d": baseline_cohen, "co_t": cohen_time, "E_0": E_0c, "E_1": E_1c}, None)
                

    def evaluate_cw(self, images, labels):
        (
            min_recorded_thing,
            cw_flag,
            min_val,
            cw,
            first_cw,
            first_cw_time,
            final_cw_time,
        ) = cw_l2_attack(
            self.model,
            images,
            labels,
            self.samples,
            self.sigma,
            self.device,
            z_val=self.z_val,
            definitive=True,
        )
        if cw_flag is True:
            cw_distance = min_val
            cw_success = 1
        else:
            cw_distance, final_cw_time, first_cw, first_cw_time = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            cw_success = 0
            
        if cw_success and self.save_images:
            torchvision.utils.save_image(images, str(self.image_count) + '-cw_reference.png')        
            torchvision.utils.save_image(min_recorded_thing, str(self.image_count) + '-cw.png')        
            torchvision.utils.save_image(min_recorded_thing - images, str(self.image_count) + '-cw_diff.png')             

        self.log.append({"cw_d": cw_distance, "cw_t": final_cw_time, "cw_f_d": first_cw, "cw_f_t": first_cw_time, "cw_s": cw_success}, None)
        
    def evaluate_cw_ablation(self, images, labels):
        for cw_num in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3]: #[1/255, 10/255, 20/255, 30/255, 40/255, 50/255]:
            suffix = "-" + str(cw_num)
            cw_time = time.time()
            
            (
                _,
                cw_flag,
                min_val,
                cw,
                first_cw,
                first_cw_time,
                final_cw_time,
            ) = cw_l2_attack(
                self.model,
                images,
                labels,
                self.samples,
                self.sigma,
                self.device,
                z_val=self.z_val,
                definitive=True,
                c = cw_num
            )

            if cw_flag is True:
                cw_distance = min_val
                cw_success = 1
            else:
                cw_distance, final_cw_time, first_cw, first_cw_time = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                cw_success = 0  
            cw_time = time.time() - cw_time                      
            
            self.log.append({"cw_d" + suffix: cw_distance, "cw_t" + suffix: cw_time, "cw_f_d" + suffix: first_cw, "cw_f_t" + suffix: first_cw_time, "cw_s" + suffix: cw_success}, None)                    

    def evaluate_new(self, images, labels):
        yp = images.clone().detach()

        new_d_time, new_attack, first_time, first_radii = direct_loop(
            self.model, self.classes, labels, self.samples, yp, images, self.z_val, self.sigma, self.device
        )

        if new_attack >= 1000:
            new_attack, first_radii, first_time, new_d_time = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            direct_success = 0
        else:
            new_attack = new_attack.detach().cpu().numpy()
            direct_success = 1

        self.log.append({"n_d": new_attack, "n_t": new_d_time, "n_f_d": first_radii, "n_f_t": first_time, "n_s": direct_success}, None)

    def evaluate_new_ablation(self, images, labels):
        for min_step in [1./255., 5./255., 10./255.]:
            for max_step in [20./255., 40./255., 100./255., 1.]:
                for delta_val in [0.01, 0.025, 0.05, 0.075, 0.1]:
                    self.model.zero_grad()
                    suffix = "-" + str(min_step) + "-" + str(max_step) + '-' + str(delta_val)
                    min_recorded, best_time, first_val, first_time, direct_success, _ = direct_new(self.model, labels.detach().clone(), self.samples, images, self.sigma, self.device, self.alpha, min_step=min_step, max_step=max_step, delta=delta_val)
                    self.log.append({"v2_d" + suffix: min_recorded, "v2_t" + suffix: best_time, "v2_f" + suffix: first_val, "v2_f_t" + suffix: first_time, "v2_s" + suffix: direct_success}, None)
                
    def evaluate_new_delta_ablation(self, images, labels):
        for delta in [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25]:
            suffix = "#" + str(delta)
            min_recorded, best_time, first_val, first_time, direct_success = direct_new(self.model, labels, self.samples, images, self.sigma, self.device, self.alpha, min_step=self.min_step, max_step=self.max_step, delta=delta)
            self.log.append({"v2_d" + suffix: min_recorded, "v2_t" + suffix: best_time, "v2_f" + suffix: first_val, "v2_f_t" + suffix: first_time, "v2_s" + suffix: direct_success}, None)            
            
    def evaluate_new_v2(self, images, labels):
        min_recorded, best_time, first_val, first_time, direct_success, yp_best = direct_new(self.model, labels, self.samples, images, self.sigma, self.device, self.alpha, min_step=self.min_step, max_step=self.max_step, delta=self.delta_val)
        
        if direct_success and self.save_images:
            torchvision.utils.save_image(images, str(self.image_count) + '-ours_reference.png')                
            torchvision.utils.save_image(yp_best, str(self.image_count) + '-ours.png')        
            torchvision.utils.save_image(yp_best - images, str(self.image_count) + '-ours_diff.png')                
        #raise ValueError('Bloody reviewers suck')        
        
        self.log.append({"v2_d": min_recorded, "v2_t": best_time, "v2_f": first_val, "v2_f_t": first_time, "v2_s": direct_success}, None)

    def evaluate_deepfool(self, images, labels):
        deep_flag, _, deep_dist, deep_time = deepfool_attack(
            self.model,
            images,
            labels,
            self.device,
            self.classes,
            self.samples,
            self.sigma,
            overshoot=0.02,
            nb_candidate=10,
            max_iter=100,
        )
        if deep_flag:
            deep_success = 1
        else:
            deep_success = 0
            deep_time, deep_dist = np.nan, np.nan

        self.log.append({"d_d": deep_dist, "d_t": deep_time, "d_s": deep_success}, None)        
    
    def evaluate_pgd(self, images, labels):
        if self.pgd_radii >= 1:
            pgd_radii = self.pgd_radii / 255
        else:
            pgd_radii = self.pgd_radii
        pgd_time = time.time()
        (
            _,
            flag,
            _,
            min_val,
            min_image,
            first_pgd,
            first_pgd_time,
            _,
        ) = pgd_attack(
            self.model,
            images,
            labels,
            self.samples,
            self.sigma,
            self.device,
            epsilon=pgd_radii,
            iters=100,
            probabilistic=True,
            z_val=self.z_val,
        )
        pgd_time = time.time() - pgd_time
        if (min_val is not None) and (min_val > 1e-5):
            pgd_success = 1
            pgd_radius = min_val
        else:
            pgd_success = 0
            pgd_radius, pgd_time, first_pgd_time, first_pgd = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        if flag and self.save_images:
            print(type(min_image), 'image type', flush=True)
            torchvision.utils.save_image(images, str(self.image_count) + '-pgd_reference.png')                
            torchvision.utils.save_image(min_image, str(self.image_count) +'-pgd.png')        
            torchvision.utils.save_image(min_image - images, str(self.image_count) + '-pgd_diff.png')            
            
        self.log.append({"pgd_d": pgd_radius, "pgd_t": pgd_time, "pgd_f_d": first_pgd, "pgd_f_t": first_pgd_time, "pgd_s": pgd_success}, None)    
        
    def evaluate_pgd_ablation(self, images, labels):
        for pgd_num in [1, 4, 8, 10, 20, 30, 40, 50, 100, 200]: #[1/255, 10/255, 20/255, 30/255, 40/255, 50/255]:
            self.model.zero_grad()
            suffix = "-" + str(pgd_num)
            pgd_radii = pgd_num / 255
            pgd_time = time.time()
            (
                _,
                _,
                _,
                min_val,
                _,
                first_pgd,
                first_pgd_time,
                _,
            ) = pgd_attack(
                self.model,
                images,
                labels,
                self.samples,
                self.sigma,
                self.device,
                epsilon=pgd_radii,
                iters=100,
                probabilistic=True,
                z_val=self.z_val,
            )
            pgd_time = time.time() - pgd_time
            if (min_val is not None) and (min_val > 1e-5):
                pgd_success = 1
                pgd_radius = min_val
            else:
                pgd_success = 0
                pgd_radius, pgd_time, first_pgd_time, first_pgd = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            self.log.append({"pgd_d" + suffix: pgd_radius, "pgd_t" + suffix: pgd_time, "pgd_f_d" + suffix: first_pgd, "pgd_f_t" + suffix: first_pgd_time, "pgd_s" + suffix: pgd_success}, None)            
    def evaluate_pgd_ablation_base(self, images, labels):
        for pgd_num in [1, 4, 8, 10, 20, 30, 40, 50, 100, 200]: #[1/255, 10/255, 20/255, 30/255, 40/255, 50/255]:
            self.model.zero_grad()
            suffix = "-" + str(pgd_num)
            pgd_radii = pgd_num / 255
            pgd_time = time.time()
            (
                _,
                _,
                _,
                min_val,
                _,
                first_pgd,
                first_pgd_time,
                _,
            ) = pgd_attack_base(
                self.model,
                images,
                labels,
                self.samples,
                self.sigma,
                self.device,
                epsilon=pgd_radii,
                iters=100,
                probabilistic=True,
                z_val=self.z_val,
            )
            pgd_time = time.time() - pgd_time
            if (min_val is not None) and (min_val > 1e-5):
                pgd_success = 1
                pgd_radius = min_val
            else:
                pgd_success = 0
                pgd_radius, pgd_time, first_pgd_time, first_pgd = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            self.log.append({"base_d" + suffix: pgd_radius, "base_t" + suffix: pgd_time, "base_f_d" + suffix: first_pgd, "base_f_t" + suffix: first_pgd_time, "base_s" + suffix: pgd_success}, None)                        
    def evaluate_autoattack(self, adv_input, images, labels, cohen_scaling=2.):       
        autoattack_time = time.time() 
        
        alpha = 0.5 * (1 - stats_norm.cdf(self.z_val))
        sm_output, _, _ = return_wrapper(self.model, adv_input, self.device)
        vals, _ = torch.topk(sm_output, 2, sorted=True)

        max_count, second_count = (
            vals[0].detach().cpu().numpy() * self.samples,
            vals[1].detach().cpu().numpy() * self.samples,
        )

        E_0c = binom_conf(max_count, self.samples, alpha=alpha, method="beta")[0]
        E_1c = binom_conf(second_count, self.samples, alpha=alpha, method="beta")[1]

        baseline_cohen = np.max(
            [
                (0.5 * self.sigma * (self.norm.icdf(torch.tensor(E_0c)) - self.norm.icdf(torch.tensor(E_1c)))).detach().cpu().numpy(),
                0,
            ]
        )
        
          
        wrapper_model = WrapperModule(self.model, self.samples, self.sigma)
        n = 6 #cohen_scaling
        upper_bound = n*baseline_cohen 
        ub_adversary = AutoAttack(wrapper_model, norm='L2', eps=upper_bound, version='rand')
        ub_adv_complete  = ub_adversary.run_standard_evaluation(images, labels, bs=1)

        autoattack_distance = torch.linalg.norm(ub_adv_complete - images)
        output = wrapper_model(ub_adv_complete).reshape(-1)
        
        autoattack_time = time.time() - autoattack_time
        
        if torch.argmax(output) != labels:
            autoattack_success = 1
        else:
            autoattack_success = 0
            autoattack_distance, autoattack_time = 1000, 1000
        
        self.log.append({"au_d": autoattack_distance, "au_t": autoattack_time, "au_s" : autoattack_success}, None)
