import torch

# Check if GPU is available
print(torch.cuda.is_available())  # noqa

# Get the number of GPUs
print(torch.cuda.device_count())  # noqa

# Get the name of the current GPU
print(torch.cuda.get_device_name(0))  # noqa
