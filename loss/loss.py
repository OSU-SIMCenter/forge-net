
def l1_penalty(net):

    l1_loss = 0.0
    for param in net.parameters():
        l1_loss += torch.sum(torch.abs(param))
    
    return(l1_loss)
