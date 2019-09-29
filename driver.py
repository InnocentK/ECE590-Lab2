from lenet5cifar10 import *

def main():
    mytransforms = [
        #transforms.Compose([transforms.ToTensor(), NORMALIZE]),
        #transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
        #transforms.Compose([transforms.RandomCrop(3,2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.CenterCrop(4),transforms.Pad(4),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(3,2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.CenterCrop(4),transforms.Pad(4),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomCrop(3,2),transforms.CenterCrop(4),transforms.Pad(4),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(3,2),transforms.CenterCrop(4),transforms.Pad(4),transforms.ToTensor(), NORMALIZE])
    ]
    for i,transform in enumerate(mytransforms):
        run(transform, i)
main()