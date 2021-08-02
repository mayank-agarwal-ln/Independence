import numpy as np

# Calculate P(B | Pa[1], ... Pa[K])
# Given:
# Pa[K] -- A vector of K Boolean values, one for each Pattern
# P(Pa | B) -- A vector of K values
# P(Pa | ~B) -- A vector of K values
# P(B) -- A scalar

def calcProb(Pa, Pa_B, Pa_nB, PB):
    K = len(Pa)
    # Calculate Likelihood = Product(k)(Pa[k] * Pa_B[k] + (1-Pa[k]) * (1-Ba_B[k]))
    aL = np.zeros((K,))
    for k in range(K):
        l = Pa[k] * Pa_B[k] + (1 - Pa[k]) * (1 - Pa_B[k])
        aL[k] = l
    L = np.prod(aL)
    # Now calculate W =  P(Pa) = Product(k)(Pa[k] * )
    aPa_B = np.zeros((K,))
    aPa_nB = np.zeros((K,))
    for k in range(K):
        #print('Pa = ', Pa, ', Pa_B = ', Pa_B)
        Pa_B1 = (Pa[k] * Pa_B[k] + (1 - Pa[k]) * (1 - Pa_B[k]))
        Pa_nB1 = (Pa[k] * Pa_nB[k] + (1 - Pa[k]) * (1 - Pa_nB[k]))
        aPa_B[k] = Pa_B1
        aPa_nB[k] = Pa_nB1
    W = np.prod(aPa_B) * PB + np.prod(aPa_nB) * (1 - PB)
    #print('L = ', L, ', PB = ', PB, ', W = ', W)
    result = L * PB / W
    return result

# Calculate, for each account, the probability of each mutually exclusive behavior,
# given the truth of each pattern for the account.
# N is the number of accounts.  J is the number of behaviors.  K is the number of patterns.
# Parameters:
#  APa is the list of the truth state for each pattern.  [(accountId0 [Pa0, Pa1, ... , PaK])] for each on N accounts
#  Pa_B is [[P(Pa0 | B0), P(Pa1 | B0), ... , P(PaK | B0)], [P(Pa0 | B1), ...], [... P(PaK | BJ]]
#  PB is the probability of each behavior -- [P(B0), P(B1), ... , P(BJ)]
#  PPa is the probability of each pattern -- [P(Pa0), P(Pa1), ... , P(PaK)]
#
#  Returns [(accountId, [P(B0), P(B1), ... , P(BJ)])] for each of N accounts
def calcAllProbs(APa, Pa_B, PB, PPa):
    results = []
    for acct in APa:
        id = acct[0]
        Pa = acct[1]
        J = len(PB)
        K = len(PPa)
        PB_PaList = []
        for j in range(J):
            # For each behavior
            PBj = PB[j] # Probability of Behavior j
            PPa_BjAccum = [] # Accumulator of Probability of Bj given Pa[0-(K-1)]
            PPaAccum = [] # Accumulator for Probabiity of Pattern given Pa[0-(K-1)]
            for k in range(K):
                PPak = PPa[k] # Probability of Pattern k
                PPak_Bj0 = Pa_B[j][k] # Probability of Pattern k given Behavior j
                # Adjust the  Probability of Pak given Bj when not used by Bj
                if PPak_Bj0 == 0:
                    # Pattern not used for this behavior.  Use Probability of Pak
                    continue
                # Now calculate the probability of this pattern being in this truth state given Bj
                PPak_Bj = Pa[k] * PPak_Bj0 + ((1-Pa[k]) * (1 - PPak_Bj0))
                PPa_BjAccum.append(PPak_Bj)
                # Calculate Probability of the pattern being in this truth state as:
                # P(Pa | Bj) * P(Bj) + P(Pa | ~Bj) * (1 - P(Bj)) =
                #   P(Pa | Bj) * P(B) + (P(Pa) - P(Pa | Bj) * P(Bj)) / (1 - P(B))
                # This formulation keeps 0 <= P <= 1 in the presence
                # of imprecise estimates.
                PPak_Pa = Pa[k] * PPak + (1-Pa[k]) * (1- PPak)
                PPaAccum.append(PPak_Pa)
            # Now we can calculate the probability of Bj given Pa using
            # Bayes rule.  P(Bj | Pa) = P(Bj) * P(Pa | Bj) / P(Pa) =
            #   P(Bj) * P(Pa | Bj) * P(Pa | Bj) / (P(Pa | Bj) * P(Bj) + (P(Pa) - P(Pa | Bj) * P(Bj)) / (1 - P(Bj)) 
            PPa_Bj = np.prod(PPa_BjAccum)
            PPa_Pa0 = np.prod(PPaAccum)
            # Note: We constrain P(Pa) - P(Pa | Bj) * P(Bj) to  non-negative
            # in order to prevent imperfect estimates from generating a prob > 1.
            PPa_Pa = PPa_Bj * PBj + max([0, (PPa_Pa0 - PPa_Bj * PBj)]) / (1 - PBj)
            #print(PBj, PPa_Bj, PPa_Pa, PPa_Pa0)
            PBj_Pa = PBj * PPa_Bj / PPa_Pa
            PB_PaList.append(PBj_Pa)
        results.append((id, PB_PaList))
    return results

                



