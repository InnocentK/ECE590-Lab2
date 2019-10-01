from cnntester import *

def main():

    out_file = open(OUTROOT + "/finalResults.csv", "w")
    out_file.write( "trial,reg,decay,momentum,epochs,lr,acc\n")
    out_file.close()

    #INITIAL_LR = 0.1
    lr1 = 0.1
    lr2 = 0.01
    
    DECAY = 0.92
    dmax = 1.00
    dmin = 0.85

    MOMENTUM = 0.85
    m_min = 0.75
    m_max = 1.00

    REG = 1e-5
    reg_max = 5e-4
    reg_min = 1e-5
    
    ep1 = 30
    ep2 = 60
    ep3 = 90

    reg = REG
    decay = DECAY
    momentum = MOMENTUM

    for i in range(25):
        lr = lr1

        if  i % 3 == 0:
            reg *= 5
        if  i % 6 == 0:
            decay -= 0.02
        if  i % 5 == 0:
           momentum += 0.02

        if i % 2 == 0:
            lr = lr2
        
        run(i, reg, decay, momentum, ep1, lr, i, False)

        if i % 4 == 0:
            run(i, reg, decay, momentum, ep2, lr, i, True)
        else:
            run(i, reg, decay, momentum, ep2, lr, i, False)
            run(i, reg, decay, momentum, ep3, lr, i, True)

main()