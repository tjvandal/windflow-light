import torch
import datetime as dt
import matplotlib.pyplot as plt

from windflow import inference_flows
from windflow.datasets import goesr

# load model runner
checkpoint_file = 'model_weights/windflow.raft.pth.tar'
inference = inference_flows.FlowRunner('RAFT', 
                                     overlap=128, 
                                     tile_size=512,
                                     device=torch.device('cpu'),
                                     batch_size=1)
inference.load_checkpoint(checkpoint_file)


# load data
file1 = 'data/OR_ABI-L1b-RadC-M6C10_G16_s20222751101170_e20222751103554_c20222751103590.nc'
file2 = 'data/OR_ABI-L1b-RadC-M6C10_G16_s20222751106170_e20222751108554_c20222751108596.nc'

g16_1 = goesr.L1bBand(file1).open_dataset()#.isel(x=list(range(400,1400)), y=list(range(400,1400)))
g16_2 = goesr.L1bBand(file2).open_dataset()#.isel(x=list(range(400,1400)), y=list(range(400,1400)))

# Perform inference
_, flows = inference.forward(g16_1['Rad'].values, g16_2['Rad'].values)

# Plot results
fig, axs = plt.subplots(1,2,figsize=(10,4))
axs = axs.flatten()
speed = (flows[0]**2 + flows[1]**2)**0.5
axs[0].imshow(g16_1['Rad'], vmin=200)
axs[0].set_title("Input frame 1")
axs[1].imshow(speed)
axs[1].set_title("Flow intensity")
plt.tight_layout()
plt.savefig("example-flows.png", dpi=200)