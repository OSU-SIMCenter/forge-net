import time
import utils
import matplotlib.pyplot as plt
import torch
from pytorch3d.loss import chamfer_distance


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
        loss, _ = chamfer_distance(delta_gt, delta_pred)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/(i+1)


def test_batch(x_t, a, delta_t, net, device):
    with torch.no_grad():
        x_t = x_t.to(device)
        a = a.to(device)
        delta_t = delta_t.to(device)

        x_t_perm = x_t.permute(0, 2, 1)
        delta_pred = net(x_t_perm, a[:, 0, :])
        delta_gt = delta_t[:, 0, :, :]
        loss, _ = chamfer_distance(delta_gt, delta_pred)
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
                    test_output,
                    show=False,
                    save=True,
                    name=(output_folder + f"epoch_{i}")
                )
