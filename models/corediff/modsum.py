from corediff_wrapper import Network

import torch
from torchinfo import summary

model = Network(in_channels=1, out_channels=1, context=True)
model.to('cuda')  # Move the model to GPU if available

# Define example inputs to the model that match the expected shapes and data types
example_input1 = torch.randn(1, 300).to('cuda')  # Adjust shape as necessary
example_input2 = torch.randint(0, 10, (1, 300)).to(torch.long).to('cuda')  # Adjust shape and range as necessary

# Use torchinfo's summary with example inputs
summary(model, input_data=[example_input1, example_input2])
