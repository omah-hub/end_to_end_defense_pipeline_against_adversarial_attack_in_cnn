import torch
import torch.nn.functional as F

def cw_attack(model, images, labels, c=1, iters=10, lr=0.01):
    adv_images = images.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([adv_images], lr=lr)

    for _ in range(iters):
        outputs = model(adv_images)

        one_hot = torch.eye(outputs.size(1), device=images.device)[labels]
        real = torch.sum(one_hot * outputs, dim=1)
        other = torch.max((1 - one_hot) * outputs - one_hot * 1e4, dim=1)[0]

        loss = torch.sum(torch.clamp(real - other, min=0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adv_images.data = torch.clamp(adv_images.data, 0, 1)

    return adv_images.detach()