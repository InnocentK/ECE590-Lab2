from lenet5cifar10 import *

def main():
    DECAY = 1.00
    DECAY_EPOCHS = 2
    mytransform = transforms.Compose([transforms.ToTensor(), NORMALIZE])
    run(mytransform, 300)
    #run(mytransform, 31, 0.5)
    #run(mytransform, 32, 0.25)
    #run(mytransform, 33, 0.1)
    #run(mytransform, 34, 0.05)
    #run(mytransform, 35, 0.02)
    #run(mytransform, 36, 0.01)
    run(mytransform, 37, 1.10)
    run(mytransform, 38, 1.25)
    run(mytransform, 39, 1.50)
    run(mytransform, 40, 2.00)

    #mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE])#,
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
    #]
    #for i,transform in enumerate(mytransforms):
     #   run(transform, i + 2, 0.01)
main()