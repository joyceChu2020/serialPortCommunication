import numpy as np
import math
########### we have the following functions: ###################
#################  parityOnRaw###############
################## splitRaw #################
################## fiveBytesToFloat #############




def parityOnRaw(raw, bytestoread): #bitxor in matlab has result in unit8.
	idx = 1
	parity = 0
	while idx <= bytestoread-1:
		parity = np.uint8(parity)^np.uint8(raw[0][idx])
		idx+=1
		if (idx+1)%53 == 0:
			if type(parity) is type(np.uint8(raw[0][idx])):
				if parity != np.uint8(raw[0][idx]):

					print('PARITY CHECK ERROR. parity and raw values are:', parity, raw[0][idx])
					
			idx += 2
			parity = 0

						



def splitRaw(raw, activeDets, recsPerSrcPos):
	nr = 52
	R = np.zeros((20,recsPerSrcPos*nr))
	i = 1
	det = 0
	sc = np.zeros((1,20)) #sampe count
	
	while activeDets[0][det] == 0:#advance to first active detector
		det += 1

		
	while (i+nr) < raw.shape[1] and ((int(sc[0][det]+1)*nr) < len(R[det])):
		if int(raw[0][i-1]) != det:
			print('Detector Sequence Error')
		R[det][int(sc[0][det]*nr) : ((int(sc[0][det]+1)*nr))] = raw[0][i:(i+nr)]
		sc[0][det] = sc[0][det]+1
		i=i+nr+1
		det = (det+1)%20

		while activeDets[0][det] == 0:
			det = (det+1)%20
	return R 





def fiveBytesToFloat(iN): # called in function: decodeFirst/SecondFreq
	if iN.shape == (5,):
		out = iN[4]*(256**4) + iN[3]*(256**3) + iN[2]*(256**2) + iN[1]*256 + iN[0]
		if out > ((2**35)-1):
			out = out - 2**36
	else:
		print('input not in required format')
	return out


############################################################################3
def decodeFirstFreq(iN, kN): # called by rawToSig func
	# if iN.shape != (1,20):
	if iN.shape != (20,):
		print('input not in required format and the size of the input is', iN.shape)

	stage1 = fiveBytesToFloat(iN[0:5])
	stage2 = fiveBytesToFloat(iN[5:10])
	stage3 = fiveBytesToFloat(iN[10:15])
	stage4 = fiveBytesToFloat(iN[15:20])
	stage0 = -1 * stage4

	out = stage0 + stage1 * np.exp(-complex(0,1)*2*math.pi*kN) - stage2 * np.exp(complex(0,1)*2*math.pi*2*kN) - stage3 * np.exp(complex(0,1)*2*math.pi*kN)
	return out



def decodeSecondFreq(iN, kN): #called by rawToSig function.
	if iN.shape != (30,):
		print('input not in required format')
	stage1 = fiveBytesToFloat(iN[0:5])
	stage2 = fiveBytesToFloat(iN[5:10])
	stage3 = fiveBytesToFloat(iN[10:15])
	stage4 = fiveBytesToFloat(iN[15:20])
	stage5 = fiveBytesToFloat(iN[20:25])
	stage6 = fiveBytesToFloat(iN[25:30])

	stage0 = -1 * stage6;

	out = stage0 + stage1 * np.exp(-complex(0,1)*2*math.pi*kN) + stage2 * np.exp(-complex(0,1)*2*math.pi*2*kN) - stage3 * np.exp(complex(0,1)*2*math.pi*3*kN) - stage4 * np.exp(complex(0,1)*2*math.pi*2*kN) - stage5 * np.exp(complex(0,1)*2*math.pi*kN)
	return complex(out)

####################################################################################3
def rawToSig(A):
	N = 4e6 #sample length of DFT block
	k1N = 3/8 #DFT bin of first modulation frequency
	k2N = 5/12 #DFT bin of second modulation frequency
	db1k = 1999998  # samples between even and odd blocks
	pkLen = 52
	nBytes = A.shape[0]
	nOvfl = 0
	nStageOvfl = 0
	cntr = A[0]%64

	R1=[]
	R2=[]

	for idx in range(0, nBytes, pkLen):
		if A[idx]%64 == cntr:
			cntr = (cntr+1)%64
		else:
			print('prob occurs at idx=', idx,'Index error. Expected= '+str(cntr)+'Read =  '+str(A[idx]%64))
		if A[idx]%128 >= 64:
			nOvfl += 1
		if A[idx]%2 == 0:
			R1.append(decodeFirstFreq(A[(idx+1):(idx+21)], k1N))
			R2.append(decodeSecondFreq(A[(idx+21):(idx+51)], k2N))
		else:
			R1.append(decodeFirstFreq(A[(idx+1):(idx+21)], k1N)*np.exp(complex(0,1)*2*math.pi*k1N*db1k))
			R2.append(decodeSecondFreq(A[(idx+21):(idx+51)], k2N)*np.exp(complex(0,1)*2*math.pi*k2N*db1k))
				
	if (nOvfl>0) or (nStageOvfl>0):
		print(str(nOvfl)+' blocks contain ADC overflows')
		print(str(nStageOvfl)+' stages are within factor 2 of limit')

	return R1, R2


