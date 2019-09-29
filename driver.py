from lenet5cifar10 import *

def main():
    mytransforms = [
        transforms.Compose([transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomCrop(2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.Pad(2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.Pad(2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomCrop(2),transforms.Pad(2),transforms.ToTensor(), NORMALIZE]),
        transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(2),transforms.Pad(2),transforms.ToTensor(), NORMALIZE])
    ]
    for i,transform in enumerate(mytransforms):
        run(transform, i)
main()