import Behavior

print('Pa_B = ', .95, ', Pa_nB = ', .05)

def PB(K):
    Pa = [1] * K + [0] * (5-K)
    Pa_B = [.95] * 5
    Pa_nB = [.05] * 5
    PB = .005
    #print('Pa = ', Pa, ', Pa_B = ', Pa_B, ', Pa_nB = ', Pa_nB, ', PB = ', PB)
    result = Behavior.calcProb(Pa, Pa_B, Pa_nB, PB)
    return result
def PB2(K):
    Pa1 = [1] * K + [0] * (5-K)
    Pa2 = [0] * K + [1] * (5-K)
    acct = [(1, Pa1), (2, Pa2)]
    Pa_B1 = [.95] * 5
    Pa_B2 = [.95] * 4 + [0]
    Pa_B = [Pa_B1, Pa_B2]
    PPa = [.1] * 5
    PB = [.005, .005]
    #print('Pa = ', Pa, ', Pa_B = ', Pa_B, ', Pa_nB = ', Pa_nB, ', PB = ', PB)
    result = Behavior.calcAllProbs(acct, Pa_B, PB, PPa)
    return result


for k in range(6):
    pb = PB(k)
    print('K = ', k, 'P(B) = ', pb)

    pb = PB2(k)
    print('K(all) = ', k, 'P(B) = ', pb)
