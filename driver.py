from lenet5cifar10 import *

def main():
    DECAY = 1.00
    EPOCHS = 30
    MOMENTUM = 0.85
    mytransform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE])
    run(mytransform, 50)
    run(mytransform, 51, MOMENTUM, 60)
    run(mytransform, 52, 0.80, 70)
    run(mytransform, 53, 0.75, 80)

    #mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE])#,
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
    #]
    #for i,transform in enumerate(mytransforms):
     #   run(transform, i + 2, 0.01)
main()