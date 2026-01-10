import time
import utils
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance

def l1_penalty(net):

    l1_loss = 0.0
    for param in net.parameters():
        l1_loss += torch.sum(torch.abs(param))
    
    return(l1_loss)


def train_epoch(train_loader, net, optimizer, device):
    epoch_loss = 0
    
    for i, (x_t, a, delta_t, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        x_t = x_t.to(device) # [B, N, 3]
        x_t_perm = x_t.permute(0, 2, 1) # [B, 3, N]
        a = a.to(device) # [B, 1, A]
        delta_t = delta_t.to(device) # [B, 1, N, 3]
        delta_pred = net(x_t_perm, a[:, 0, :]) # [B, N, 3]
        delta_gt = delta_t[:, 0, :, :] # [B, N, 3]
        magnitude_gt = torch.norm(delta_gt, dim=-1, keepdim=True)
        # mse_loss = torch.mean(magnitude_gt * (delta_pred - delta_gt) ** 2)/ (torch.mean(magnitude_gt * delta_gt ** 2) + 1e-8)
        # mse_loss = 0
        # direction_loss = (1 - F.cosine_similarity(delta_pred, delta_gt, dim=-1)).mean()
        # loss = mse_loss +  2 * direction_loss
        # loss = weighted_loss(delta_pred, delta_gt, x_t, a)
        # loss, _ = chamfer_distance(delta_gt, delta_pred)
        loss = torch.mean((delta_pred - delta_gt) ** 2)
        # loss += 1e-6*l1_penalty(net)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        # torch.nn.util.clip_grad_value_(net.parameters(), clip_value=0.1)
        optimizer.step()
        epoch_loss += loss.item()

    # print(magnitude_gt.shape, mse_loss, direction_loss)


    return epoch_loss/(i+1)


def test_batch(x_t, a, delta_t, net, device):
    with torch.no_grad():
        x_t = x_t.to(device)
        a = a.to(device)
        delta_t = delta_t.to(device)

        x_t_perm = x_t.permute(0, 2, 1)
        delta_pred = net(x_t_perm, a[:, 0, :])
        delta_gt = delta_t[:, 0, :, :]
        
        # magnitude_gt = torch.norm(delta_gt, dim=-1, keepdim=True)
        # weight = 1.0 + magnitude_gt
        # mse_loss = torch.mean(weight * (delta_pred - delta_gt) ** 2)/ (torch.mean(weight * delta_gt ** 2) + 1e-8)
        # direction_loss = (1 - F.cosine_similarity(delta_pred, delta_gt, dim=-1)).mean()
        # loss = mse_loss + 2 * direction_loss
        # loss, _ = chamfer_distance(delta_gt, delta_pred)
        loss = torch.mean((delta_pred - delta_gt) ** 2)
        # loss += 1e-6*l1_penalty(net)
        x_tp1_pred = x_t + delta_pred

    return loss.item(), x_tp1_pred.cpu()


def test_epoch(test_loader, net, device):
    with torch.no_grad():
        epoch_loss = 0
        for i, (x_t, a, delta_t, _) in enumerate(test_loader):
            loss, _ = test_batch(x_t, a, delta_t, net, device)
            epoch_loss += loss
    return epoch_loss/(i+1)


def train_model(train_loader, test_loader, net, epochs, optimizer, device, save_results, output_folder):
    train_loss_list = []  
    test_loss_list = []
    
    for i in range(epochs):
        startTime = time.time()
        
        train_loss = train_epoch(train_loader, net, optimizer, device)
        train_loss_list.append(train_loss)
        
        test_loss = test_epoch(test_loader, net, device)
        test_loss_list.append(test_loss)
        
        epoch_time = time.time() - startTime

        writeString = (
            f"epoch {i} train loss: {train_loss} test loss: {test_loss} epoch time: {epoch_time}\n"
        )
        
        plt.plot(train_loss_list, label="Train")
        plt.plot(test_loss_list, label="Test")
        plt.legend()
    
        if save_results:
            with open(output_folder + "prints.txt","a") as file: 
                file.write(writeString)

            plt.savefig(output_folder + "loss.png")
            plt.close()
    
            if i % 50 == 0:
                test_samples, test_actions, test_deltas, test_samples_next = next(iter(test_loader))

                loss, test_output = test_batch(test_samples, test_actions, test_deltas, net, device)

                utils.plotPCbatch(
                    test_samples,
                    test_samples_next[:, -1, :, :],
                    test_output / 100,
                    show=False,
                    save=True,
                    name=(output_folder + f"epoch_{i}")
                )
