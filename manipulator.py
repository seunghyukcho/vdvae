import cv2
import torch
import numpy as np
from vae import VAE
from data import set_up_data
from train_helpers import set_up_hyperparams, restore_params

H, _ = set_up_hyperparams()
H, _, data, preprocess_fn = set_up_data(H)

model = VAE(H).cuda()
restore_params(model, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
model.requires_grad_(False)

cutoff = 33
inp1, inp2 = list(data[1]), list(data[0])
cv2.imwrite(f'examples/mixed2/{cutoff}/original1.jpg', cv2.cvtColor(inp1[0].numpy(), cv2.COLOR_BGR2RGB))
cv2.imwrite(f'examples/mixed2/{cutoff}/original2.jpg', cv2.cvtColor(inp2[0].numpy(), cv2.COLOR_BGR2RGB))
inp1[0] = inp1[0].unsqueeze(dim=0)
inp1, _ = preprocess_fn(inp1)
inp2[0] = inp2[0].unsqueeze(dim=0)
inp2, _ = preprocess_fn(inp2)

with torch.no_grad():
    outputs1 = model.forward_get_latents(inp1)
    outputs2 = model.forward_get_latents(inp2)
    latents1, latents2 = [], []

    for output1, output2 in zip(outputs1, outputs2):
        latents1.append(output1['z'])
        latents2.append(output2['z'])

    latents = latents1[:cutoff]
    latents += latents2[cutoff:]

    samples = model.forward_samples_set_latents(16, latents)
    samples = np.array(samples)
    
    for idx, sample in enumerate(samples):
        cv2.imwrite(f'examples/mixed2/{cutoff}/sample_{idx}.jpg', cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

# cutoff = 64
# inp = list(data[0])
# cv2.imwrite(f'examples/{cutoff}/original.jpg', cv2.cvtColor(inp[0].numpy(), cv2.COLOR_BGR2RGB))
# inp[0] = inp[0].unsqueeze(dim=0)
# inp, _ = preprocess_fn(inp)

# with torch.no_grad():
#     outputs = model.forward_get_latents(inp)
#     latents = []
    
#     for output in outputs:
#         latents.append(output['z'])
#     latents = latents[:cutoff]

#     samples = model.forward_samples_set_latents(16, latents)
#     samples = np.array(samples)
#     print(samples.shape)
#     for idx, sample in enumerate(samples):
#         cv2.imwrite(f'examples/{cutoff}/sample_{idx}.jpg', cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

