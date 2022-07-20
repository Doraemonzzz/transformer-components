import torch

def orthogonal_random_matrix_block(d):
    w = torch.rand(d, d)
    q, r = torch.linalg.qr(w)
    
    # columns of q are orthogonal, return the transpose of them
    return q.t()

def orthogonal_random_matrix(row, col):
    """
        Implement orthogonal random features.
    """
    block_num = row // col
    remaining = row % col
    block_list = []
    
    for _ in range(block_num):
        q = orthogonal_random_matrix_block(col)
        block_list.append(q)
    
    if remaining > 0:
        q = orthogonal_random_matrix_block(col)
        block_list.append(q[:remaining])
        
    # row, col
    w = torch.cat(block_list)
    
    # normalize
    scale = torch.randn((row, col)).norm(dim=-1).reshape(-1, 1)
    w = w / scale
    
    return w

# row = 33
# col = 32
# w = orthogonal_random_matrix(row, col)
# print(w.shape)