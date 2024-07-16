import argparse
import os

import torch

import numpy as np


class Logger(object):
    '''
    Tracks the reported values of an individual statistic
    '''

    def __init__(self, name=''):
        self.name = name
        self.log = []
        self.count = 0
        self.mean = None
        self.median = None
        self.min = None
        self.max = None
    def append(self, numbers):
        if isinstance(numbers, int) or isinstance(numbers, float):
            numbers = [numbers]
        if isinstance(numbers, torch.Tensor):
            numbers = numbers.detach().cpu().numpy()
        if isinstance(numbers, np.ndarray):
            numbers = numbers.reshape(-1)
        for number in numbers:
            self.log.append(number)
            self.count += 1
    def calculate_statistics(self):
        self.mean = np.mean(self.log)
        self.median = np.median(self.log)
        self.min = np.min(self.log)
        self.max = np.max(self.log)
    def return_statistics(self):
        self.calculate_statistics()
        return self.mean, self.median, self.min, self.max
        
class LogSet(object):
    '''
    Tracks the reported values of a set of statistic, including statistics and printing to file
    '''
    def __init__(self, filename, decimals=4, normalise=False, means=False, console=False):
        self.header_flag = False
        self.header_flag_normalised = False
        self.header_flag_means = False
        self.logs = {}
        self.logset = []
        self.filename = filename

        self.normalise = normalise
        self.filename_normalised = filename + '-normalise'
        self.means = means
        self.filename_means = filename + '-means'
        self.filename_medians = filename + '-medians'
        
        self.decimals = decimals
        self.console = console
    
    def append(self, value, log):
        if type(value) is dict:
            for key in value:
                self.append(value[key], key)
        else:    
            if value == False or value is None:
                value = 0           
        
            if log not in self.logset:
                if (self.header_flag is False) or (self.header_flag_normalised is False):
                    # Rejects the addition of new logged values if a the headers have already been established by printing
                    self.logset.append(log)
                    self.logs[log] = Logger(log)
            if torch.is_tensor(value):
                value = value.item()
            self.logs[log].append(value)
    
    def return_statistics(self, log):
        return self.logs[log].return_statistics()
        
    def print(self):   
        # Need to make sure I'm clearing the file at the start of the codebase    
        print(' \t '.join(self.logset), flush=True)
        if self.header_flag is False:
            with open(self.filename, 'w') as f:
                print(' \t '.join(self.logset), file=f, flush=True)
            if (self.console is True):
                print(' \t '.join(self.logset), flush=True)                                   
            self.header_flag = True
        if self.header_flag_normalised is False:
            with open(self.filename_normalised, 'w') as f:
                print(' \t '.join(self.logset), file=f, flush=True)
            self.header_flag_normalised = True
        if self.header_flag_means is False:
            with open(self.filename_means, 'w') as f:
                print(' \t '.join(self.logset), file=f, flush=True)
            with open(self.filename_medians, 'w') as f:
                print(' \t '.join(self.logset), file=f, flush=True)                
            self.header_flag_means = True            

        rounding_format = "{:." + str(self.decimals) + "f}"
        with open(self.filename, 'a') as f:         
            print_vals = []
            normalised_vals = []
            mean_vals = []
            median_vals = []
            for log in self.logset:
                self.logs[log].calculate_statistics()
                if log is not 'iter':
                    mean_vals.append(self.logs[log].mean)
                    median_vals.append(self.logs[log].median)                    
                else: 
                    mean_vals.append(self.logs[log].log[-1])
                    median_vals.append(self.logs[log].log[-1])                    
                if (self.normalise is not False) and (log not in ['iter', 'label', 'E0', 'rej']):
                    normalised_vals.append(self.logs[log].log[-1] / self.logs[self.normalise].log[-1])                    
                else:
                    normalised_vals.append(self.logs[log].log[-1])
                print_vals.append(self.logs[log].log[-1])
            print_string = ' \t '.join([rounding_format.format(val) for val in print_vals])            
            print(print_string, file=f, flush=True)
            mean_string = ' \t '.join([rounding_format.format(val) for val in mean_vals])
            median_string = ' \t '.join([rounding_format.format(val) for val in median_vals])            
            if self.console is True:
                print(print_string, flush=True)
                print(mean_string, flush=True)
        if self.normalise is not False:
            with open(self.filename_normalised, 'a') as f:
                normalised_string = ' \t '.join([rounding_format.format(val) for val in normalised_vals])            
                print(normalised_string, file=f, flush=True) 
        if self.means is True:
            with open(self.filename_means, 'a') as f:
                print(mean_string, file=f, flush=True)
            with open(self.filename_medians, 'a') as f:
                print(median_string, file=f, flush=True)
                    
