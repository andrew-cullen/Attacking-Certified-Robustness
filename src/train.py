import time

import torch
from torch.autograd import Variable


SAVE_LOC = "../../trained_models/main/"

import time
from torch.cuda.amp import GradScaler

def train(device, model, optimizer, lr_scheduler, num_epochs, train_loader, val_loader, args, name, val_cutoff=1e6, resume_epoch=0):
    loss_fn = torch.nn.CrossEntropyLoss()

    amp = False
    scaler = GradScaler()
    print_flag_temp = True

    for epoch in range(num_epochs):
        if epoch >= resume_epoch:
            if device is not None:
                epoch_numerator, epoch_denominator = torch.tensor(0.).to(device), torch.tensor(0.).to(device)            
            else:
                epoch_numerator, epoch_denominator = torch.tensor(0.), torch.tensor(0.)           
            model.train()
            total_batch = len(train_loader)
            start_time = time.time()
            
            i = 0
            for data, target in train_loader:
                i += 1                                        
                optimizer.zero_grad() 
           
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)            
                
                if device is not None:
                    data, target = data.to(device), target.to(device)
                else:
                    data, target = data.cuda(), target.cuda()
                    if print_flag_temp:
                        print('Device is ', device, data.device, target.device, isinstance(model, torch.nn.DataParallel), flush=True)
                        print_flag_temp = False
                
                data += Variable(data.data.new(data.size()).normal_(0,args.sigma))


                if amp:
                    with torch.cuda.amp.autocast():#torch.autocast("cuda", dtype=torch.float16): #torch.cuda.amp.autocast(dtype=torch.float16): #torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):  
                        pred = model(data)        
                        loss = loss_fn(pred, target)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    pred = model(data)        
                    loss = loss_fn(pred, target)

                    loss.backward()
                    optimizer.step()
                if device is None: # This needs to be corrected, the step is wasteful
                    pred = pred.to('cpu')
                    target = target.to('cpu')
                                        
                epoch_numerator += 100. * torch.sum(torch.argmax(pred, dim=1) == target)
                epoch_denominator += pred.shape[0]
                
                secondary_accuracy = 100. * torch.sum(torch.argmax(pred, dim=1) == target) / pred.shape[0]
                batch_accuracy = secondary_accuracy / 100
                                
                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f, Acc: %.4f, Time: %.4f'
                         %(epoch+1, num_epochs, i+1, total_batch, loss.item(), 100*batch_accuracy, time.time() - start_time))                   

            epoch_accuracy = epoch_numerator / epoch_denominator
            epoch_numerator, epoch_denominator = torch.tensor(0.), torch.tensor(0.)
            print('Epoch Accuracy: {}'.format(100*epoch_accuracy))
            
            model.eval()
            total_batch = len(val_loader)
            val_loss = 0
            secondary_val_accuracy = 0.
            total_shape = 0.
            for i, (data, target) in enumerate(val_loader):        
                data = data + torch.randn_like(data) * args.sigma
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                if device is not None:
                    data, target = data.to(device), target.to(device)
                else:
                    data, target = data.cuda(), target.cuda()

                pred = model(data)        
                loss = loss_fn(pred, target)
                val_loss += loss.item() * pred.shape[0]
                                
                if device is None: # This needs to be corrected, the step is wasteful
                    pred = pred.to('cpu')             
                    target = target.to('cpu')
                
                secondary_val_accuracy += torch.sum(torch.argmax(pred, dim=1) == target) 
                total_shape += pred.shape[0]
                
            val_accuracy = secondary_val_accuracy / total_shape
            val_loss = val_loss / total_shape
            print('VAL Epoch [%d/%d], Loss: %.4f, Acc: %.4f'            
                     %(epoch+1, num_epochs, val_loss , 100*val_accuracy))#.compute(), secondary_val_accuracy))            
                     
            if val_loss < val_cutoff:
                print('Saving at epoch: {}'.format(epoch+1))
                torch.save(model.module.state_dict(), SAVE_LOC + name + '-' + str(args.sigma) + '-weight.pth')
                val_cutoff = val_loss

                 
            if lr_scheduler is not None:
                lr_scheduler.step()

    return model, val_cutoff

