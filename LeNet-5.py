''' Implementation of LeNet-5 CNN architecture utilizing MNIST dataset '''

import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import logging

# Local imports provided by sebastianraschka course
from helper_evaluation import set_all_seeds, set_deterministic, compute_confusion_matrix
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_dataset import get_dataloaders_mnist

# Logging init
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger()


# Set settings for reproducibility 
RANDOM_SEED = 123
BATCH_SIZE = 256
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_all_seeds(RANDOM_SEED)
log.info(f"Device setting: {DEVICE}")

# MNIST dataset set-up (Not super important, but useful to know down the road)
resize_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5,), (0.5,)) 
    ]
)

train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    batch_size=BATCH_SIZE, 
    validation_fraction=0.1, 
    train_transforms=resize_transform, 
    test_transforms=resize_transform)

# Check the loaded dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape, ' shows 256 images w/ 1 channel at 32x32')
    print('Image label dimensions:', labels.shape, ' shows 256 labels')
    print('Class labels of 10 examples:', labels[:10])
    break

# Model creation
class LeNet5(torch.nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super().__init__()
        self.grayscale=grayscale
        self.num_classes = num_classes

        # Setting number of color channels
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3
        
        # Set feature detector using nn.sequential (stride of kernel default=1)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels=6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.Conv2d(6,16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # Set dense layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 16*5*5, out_features = 120),
            torch.nn.Tanh(), 
            torch.nn.Linear(in_features = 120, out_features = 84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features = 84, out_features = num_classes)
        )

    # Define the forward process of the model
    def forward(self, x):
        x = self.features(x)  # Runs feature detector over x
        x = torch.flatten(x,1)  # Flattens the output from convolutional/pooling layers
        logits = self.classifier(x)  # Runs flattened vector through dense layers
        return logits

if __name__ == "__main__": 
    model = LeNet5(grayscale=True, num_classes=10)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1) # Init backprop

    # Set learning rate decay during plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        factor = 0.1,
        mode = 'max', 
        verbose = True
    )
    # Implement model training.
    ###### REVIEW THE CODE IN TRAIN_MODEL HELPER SCRIPT ########
    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model = model, 
        num_epochs = NUM_EPOCHS, 
        train_loader = train_loader, 
        valid_loader = valid_loader, 
        test_loader=test_loader, 
        optimizer = optimizer, 
        device = DEVICE, 
        logging_interval = 100
    )

    plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=NUM_EPOCHS,
                   iter_per_epoch=len(train_loader),
                   results_dir=None,
                   averaging_iterations=100)
    plt.show()

    plot_accuracy(train_acc_list=train_acc_list,
                valid_acc_list=valid_acc_list,
                results_dir=None)
    plt.ylim([80, 100])
    plt.show()

    model.cpu()
    show_examples(model=model, data_loader=test_loader)

    class_dict = {0: '0',
              1: '1',
              2: '2',
              3: '3',
              4: '4',
              5: '5',
              6: '6',
              7: '7',
              8: '8',
              9: '9'}

    mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
    plot_confusion_matrix(mat, class_names=class_dict.values())
    plt.show()