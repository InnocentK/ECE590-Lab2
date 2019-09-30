from mycnn import *

def main():
    #DECAY = 1.00
    #EPOCHS = 30
    #MOMENTUM = 0.85
    #mytransform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), NORMALIZE])
    #run(102)
    #run(103, epochs=65)
    #run(103, epochs=100, loadTest=True)
    run(epochs=0, loadTest=True)
main()