import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

from collections import OrderedDict

from datasets import DATASETS
from models import model_settings
from train import train

#from evaluation_another_copy import evaluation_loop, cutoff_test, cr_loop, scaling_analysis
from evaluation import Evaluation, cutoff_test, cr_loop, scaling_analysis


def rename_state_dict(base_dict, new_prefix):
    new_dict = OrderedDict()
    for key, value in base_dict.items():
        new_key = new_prefix + key.partition('.')[2].partition('.')[2] # Corrects key to match loading in here
        new_dict[new_key] = value
        
    return new_dict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean Value Expected')

PARALLEL_CHOICES = ['never', 'always', 'eval']
SAVED_LOC = "../../trained_models/main/"

torch.manual_seed(0)
import random
random.seed(0)

parser = argparse.ArgumentParser(description='Certifying examples as per Smoothing')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--filename', type=str, default='None')
parser.add_argument('--parallel', type=str, choices=PARALLEL_CHOICES, help='Never if parallel will never be used, always = Training & Eval, eval = only at evaluation')
parser.add_argument('--batch_size', type=int, default=0, help='Batch Size (0 == Model default)')
parser.add_argument('--certification_iters', type=int, default=100, help='Batch Size (0 == Model default)')
parser.add_argument('--lr', type=float, default=0, help='Learning Rate (0 == Model default)')

parser.add_argument('--resume', type=int, default=0, help='Epoch to start training from (0 == Train from scratch)')

parser.add_argument('--sigma', type=float, default=0.0, help='Noise level')
parser.add_argument('--gamma_start', type=float, default=0.0005, help='Step size starting point')
parser.add_argument('--samples', type=int, default=1500, help='Number of samples')
parser.add_argument('--epochs', type=int, default=80, help='Training Epochs')
parser.add_argument('--total_cutoff', type=int, default=250, help='Number of samples tested over')

parser.add_argument('--train', action='store_true', help='If training is required')
parser.add_argument('--eval', action='store_true', help='If evaluation is required')
parser.add_argument('--cutoff_experiment', action='store_true', help='If cutoff experiment is performed')
parser.add_argument('--new_cr', action='store_true', help='If improved cr experiment is performed')
parser.add_argument('--plotting', type=str2bool, nargs='?', const=True, default=True, help='If cutoff experiment is performed')

parser.add_argument('--ablation', action='store_true', help='If ablation study is required')
   
parser.add_argument('--autoattack_radii', type=float, default=-1, help='Noise level')
parser.add_argument('--pgd_radii', type=float, default=20, help='Noise level')
parser.add_argument('--new_min_step', type=float, default=0.01, help='Noise level')
parser.add_argument('--new_max_step', type=float, default=0.125, help='Noise level')

parser.add_argument('--start_point', type=int, default=0, help='Because of a fixed seed, this offsets the start point in the evaluation search')

args = parser.parse_args()

if args.filename == 'None':
    args.filename = None


cudnn.benchmark = True

print(torch.__version__, ' Version ', flush=True)

def to_dataparallel(model):
    cuda_device_count = torch.cuda.device_count()        
    print('Cuda device count: ', cuda_device_count)
    model = model.to("cpu")
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)])
    if cuda_device_count == 1:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        device = None #torch.device("cuda:0") # None #f'cuda:{model.device_ids[0]}'
        model.cuda()
    return model, device



def to_dataparallel_eval(model):
    cuda_device_count = torch.cuda.device_count()        
    print('Cuda device count in dataparallel evel: ', cuda_device_count)
    model = model.to("cpu")
    #model = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)])
    if cuda_device_count == 1:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        #model_out = lambda x: model(x).to(torch.device("cpu"))
        model_out = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)]) #model
        model_out.cuda()
        device = torch.device("cuda:0") #None #torch.device("cuda:0") # None #f'cuda:{model.device_ids[0]}'
        return model_out, device
        #model.cuda()
    return model, device


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Preload model settings
    model, loss, optimizer, lr_scheduler, train_loader, val_loader, test_loader, device, classes = model_settings(args.dataset, args)
    
    print(args.resume, args.train, flush=True)
    if (args.resume > 0) or (args.train == False):
        print('Loading Model', flush=True)
        pth = SAVED_LOC + args.dataset + '-' + str(args.sigma) + '-weight.pth'               
        loc = 'cuda'#:{}'.format(args.gpu_num)
    
        checkpoint = torch.load(pth)   
        try:               
            model.load_state_dict(checkpoint)
        except:
            print('Modifying dict')
            new_model_state = {}
            for key in checkpoint.keys():
                new_model_state['model.' + key.partition('1.')[2]] = checkpoint[key]
            #checkpoint = {'model.' + k.partition('1.')[2]: v for k,v in checkpoint}
            model.load_state_dict(new_model_state)
        model.eval()     
        
    # Train
    if args.train:
        print('Training', flush=True)
        #if args.parallel == 'always': # 18 September
        #    model, device = to_dataparallel(model)                       
        model, device = to_dataparallel(model)                       
        model, cutoff = train(device, model, optimizer, lr_scheduler, args.epochs, train_loader, val_loader, args, args.dataset, val_cutoff=1e6, resume_epoch=args.resume)
  
                     
    del train_loader, val_loader
   
    if args.eval or args.new_cr:      
        print('Evaluating attacks')

        if args.parallel == 'eval' or ((args.parallel == 'always') and (args.train is False)):
            model, device = to_dataparallel_eval(model)        
        else:
            device = torch.device("cuda:0")
            model.to(device)
        print(f'Device is {device}', flush=True)

        if args.dataset == 'imagenet':
            autoattack = False
        else:
            autoattack = True
        autoattack = False
        print('Autoattack and deepfool disabled just for speed purposes')
            
        cert = Evaluation(args.dataset, classes, device, model, test_loader, args.sigma, args.samples, ablation=args.ablation, pgd_radii=args.pgd_radii, new_min_step=args.new_min_step, new_max_step=args.new_max_step, filename=args.filename, total_cutoff=args.total_cutoff, fool=False, cw=False, autoattack=autoattack, start_point=args.start_point)
        cert.evaluate()
            
