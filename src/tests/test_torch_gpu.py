import torch

print(f'torch.cuda.is_available = {torch.cuda.is_available()}')
print(f'torch.cuda.current_device = {torch.cuda.current_device()}')
print(f'device_name = {torch.cuda.get_device_name(torch.cuda.current_device())}')