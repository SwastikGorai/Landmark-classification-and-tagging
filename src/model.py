import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # vgg 16
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) #  224x224x3 -> 224x224x64
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1) # 224x224x64 -> 224x224x64
        self.pool = nn.MaxPool2d(2, 2) # 224x224x64 -> 112x112x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 112x112x64 -> 112x112x128
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1) # 112x112x128 -> 112x112x128
        self.flatten = nn.Flatten() # 112x112x128 -> 128*112*112
        self.fc1 = nn.Linear(112 * 112 * 128, 4096) # 112x112x128 -> 4096
        self.fc2 = nn.Linear(4096, 4096) # 4096 -> 4096
        self.fc3 = nn.Linear(4096, num_classes) # 4096 -> num_classes
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = self.flatten(x)  #flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1) #aactivation func (this or log_softmax)
        
        return x
        


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
