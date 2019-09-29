from lenet5cifar10 import *

def main():
    MOMENTUM = 0.9
    REG = 1e-4
    mytransform = transforms.Compose([transforms.ToTensor(), NORMALIZE])
    run(mytransform, 20)
    run(mytransform, 21, MOMENTUM, 1e-3)
    run(mytransform, 22, MOMENTUM, 1e-5)
    run(mytransform, 23, 0.85)
    run(mytransform, 24, 0.85, 1e-3)
    run(mytransform, 25, 0.85, 1e-5)
    run(mytransform, 26, 0.95)
    run(mytransform, 27, 0.95, 1e-3)
    run(mytransform, 28, 0.95, 1e-5)

    #mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE])#,
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
    #]
    #for i,transform in enumerate(mytransforms):
     #   run(transform, i + 2, 0.01)
main()