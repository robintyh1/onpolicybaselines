import numpy as np

def action_mask(action, numdims, K):
	# generate action masks
	# input: action of shape [num-action, num-of-action-dim]
	# output: mask of shape [num-action, num-of-action-dim, bins]
	totalshape = [action.shape[0], numdims, K]
	mask = np.zeros(totalshape)
	mask[:,:,:] = np.arange(K)
	index = mask <= action[:,:,np.newaxis]
	newmask = np.zeros_like(mask)
	newmask[index] = 1.0
	return newmask

def construct_mask(bins):
    a = np.zeros([bins,bins])
    for i in range(bins):
        for j in range(bins):
            if i+j <= bins-1:
                a[i,j] = 1.0
    return a

