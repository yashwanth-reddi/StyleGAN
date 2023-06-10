
import torch

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    
    Batch_size = real.shape[0]
    Channels = real.shape[1]
    Height = real.shape[2]
    Width = real.shape[3]
    
    beta = torch.rand((Batch_size, 1, 1, 1)).repeat(1, Channels, Height, Width).to(device)
    
    detached = fake.detach()
    
    interpolated_images = real * beta +  detached - beta*detached
    
    interpolated_images.requires_grad_(True)

    mixed_image_scores = critic(interpolated_images, alpha, train_step)

    
    gradient = torch.autograd.grad(
        inputs  = interpolated_images,
        outputs = mixed_image_scores,
        grad_outputs=torch.ones_like(mixed_image_scores),
        create_graph=True,
        retain_graph=True,)[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    
    gradient_norm = gradient.norm(2, dim=1)
    
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty

