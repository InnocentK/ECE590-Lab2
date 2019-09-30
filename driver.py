from lenet5cifar10 import *

def main():
    DECAY = 1.00
    EPOCHS = 30
    MOMENTUM = 0.85
    mytransform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE])
    #run(mytransform, 60, 0.92, MOMENTUM, 100)
    run(mytransform, 60, 0.93, MOMENTUM, 100)
    run(mytransform, 61, 0.91, MOMENTUM, 100)
    run(mytransform, 62, 0.89, MOMENTUM, 100)

    #mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE])#,
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
    #]
    #for i,transform in enumerate(mytransforms):
     #   run(transform, i + 2, 0.01)
main()