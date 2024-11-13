import torch

# parameters
batch_size = 32  # num_samples
seq_len = 800    # seq_len
matrix_dim = 9  # matrix_dim

sequence_tensor = torch.stack([
    torch.full((matrix_dim, matrix_dim), fill_value=i+1) 
    for i in range(seq_len)
])

out = sequence_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
out = out.permute(0, 2, 3, 1)
print(out.shape)