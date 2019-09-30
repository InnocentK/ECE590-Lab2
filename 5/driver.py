from lenet5cifar10 import *

def main():
    DECAY = 1.00
    EPOCHS = 30
    MOMENTUM = 0.85
    mytransform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE])
    run(mytransform, 100)#, 0.92, MOMENTUM, 30)
main()