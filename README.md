# PyTorch implementation of soft-argmax 1D/2D/3D

This function assumes an input tensor in shape (batch_size, channel, height, width, depth) and returns 3D coordinates in shape (batch_size, channel, 3).

For example, if your network output is (batch_size, 16, 64, 64, 64)  voxels, then the output is 3D coordinates in shape (batch_size, 16, 3). This case is usually seen when you has 16 3D heat-voxels and try to find the locations of maximum. 

To apply to 2D cases, just set depth to 1 and grab the first two coordinates. For example, your network output is 16 (200, 200) heatmaps and you are trying to find 2D locations of maximum on each map. You can sent (batch_size, 16, 200, 200, 1) to this function, and the output would be (batch_size, 3) and you take the first 2 of the 3. This idea can be applied to 1D cases by setting both width and depth to 1. 

Of course you can set channel to 1 then you'll get one coordinates for each instance in the batch. 

Be careful that, you don't want use coordinates.floor() or coordinates.round() to get integer coordinates. Because these operations are not differentiable. 

Implementation For 1D soft-argmax

```
import torch.nn as nn
def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, L)
    Return: 1D coordinates in shape (batch_size, channel)
    """
    assert voxels.dim()==3
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 10000.0 
    N,C,L = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=L).unsqueeze(0)
    indices_kernel = indices_kernel.view((L))
    conv = soft_max*indices_kernel
    indices = conv.sum(-1)
    x = indices%L
    coords = torch.stack([x],dim=2).squeeze(-1)
    return coords

```
