from lenet5cifar10 import *

def main():
    mytransform = transforms.Compose([transforms.ToTensor(), NORMALIZE])
    run(mytransform, 10)
    run(mytransform, 11, 0.02)
    run(mytransform, 12, 0.04)
    #mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE])#,
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
    #]
    #for i,transform in enumerate(mytransforms):
     #   run(transform, i + 2, 0.01)
main()