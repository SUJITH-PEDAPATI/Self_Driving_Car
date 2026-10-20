import torch
import torch.nn as nn
"""
    SteeringCNN Architecture (NVIDIA's Architecture for End-to-End Self-Driving Cars):
    1. Input: RGB road image (3, 66, 200).
    2. Convolutional Layers:
        - 5 layers extract road features such as lane markings and curves.
        - ReLU activation follows each convolution.
        - Strided convolutions reduce image dimensions.
    3. Flatten:
        - Converts extracted feature maps into 1,152 features.
    4. Fully Connected Layers:
        - 1,152 -> 100 -> 50 -> 10 neurons.
        - Learn the relationship between road features and steering.
    5. Output Layer:
        - Produces one continuous steering-angle prediction.
        - No activation, allowing positive and negative values.
"""
class SteeringCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            ## FIrst Layer of Convolution
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=5,
                stride = 2
            ),
            nn.ReLU(),

            ## Second Layer of Convolution
            nn.Conv2d(
                in_channels=24,
                out_channels=36,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),

            ## Third Layer of Convolution
            nn.Conv2d(
                in_channels=36,
                out_channels=48,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),

            ## Fourth Layer of Convolution
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),

            ## Fifth Layer of Convolution
            nn.Conv2d(
                in_channels = 64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )

        self.fully_connected_layers = nn.Sequential(
            ## Flatten
            nn.Flatten(),
            ## First Layer of Fully Connected Layers
            nn.Linear(
                in_features= 64*1*18, ## 1152 features
                out_features= 100,
            ),
            nn.ReLU(),

            ### Second Layer of Fully Connected Layers
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.ReLU(),

            ## third Layer of Fully Connected Layers
            nn.Linear(
                in_features=50,
                out_features=10
            ),
            nn.ReLU(),

            ## Fourth Layer of Fully Connected layers
            nn.Linear(
                in_features=10,
                out_features=1
            )
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fully_connected_layers(x)
        return x

    
if __name__ == "__main__":
    model = SteeringCNN()

    ## TEsting the Architecture
    sample_images = torch.randn(32,3,66,200)
    predictions = model(sample_images)
    print("Input shape:", sample_images.shape)
    print("Prediction shape:", predictions.shape)
    print("First 5 predictions:")
    print(predictions[:5])

    assert predictions.shape == (32, 1)

    print("Model test passed!")

