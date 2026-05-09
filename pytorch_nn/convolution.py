import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, num_classes, activation, padding=0):
        super().__init__()
        
        self.height, self.width = input_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        out_h = (self.height + 2 * padding - kernel_size) // stride + 1
        out_w = (self.width + 2 * padding - kernel_size) // stride + 1
        
        self.flat = nn.Flatten()
        self.activation = activation    
        self.linear_1 = nn.Linear(out_channels * out_h * out_w, 100)
        self.linear_2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.flat(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x
        

    