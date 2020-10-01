import serial
import numpy as np 
import math
import struct

import time

# from rawToSig import * 
from testrawToSig import *






def collectData(ser, repetitions, sourceSeq, activeDets, recsPerSrcPos):
#%activeDets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	nDets = np.sum(activeDets)
	print('Number of active detectors: ',nDets)
	sourceSeqLen = len(sourceSeq)
	A = np.zeros((repetitions, sourceSeqLen, 20, 2, recsPerSrcPos))
#   A(repetition, source, detector, wavelength, sample)

#   Flush Buffer
	ba = ser.inWaiting()
	if ba > 0:
		ser.read(ba)
#Send parameters to detector box
	ser.write('P'.encode())

	bytevalue=struct.pack('>B', math.floor(recsPerSrcPos/256))
	ser.write(bytevalue)
	
	bytevalue=struct.pack('>B', recsPerSrcPos%256)
	ser.write(bytevalue)

	bytevalue=struct.pack('>B', sourceSeqLen%256)
	ser.write(bytevalue)

	bytevalue=struct.pack('>B', math.floor(repetitions/256))
	ser.write(bytevalue)

	bytevalue=struct.pack('>B', repetitions%256)
	ser.write(bytevalue)


	for d in range(1, 21):
		bytevalue=struct.pack('>B', int(activeDets[0][d-1]))
		#print(d,bytevalue)
		ser.write(bytevalue)

	time.sleep(1e-3)





#Send source sequence to source box

	ser.write('S'.encode()) #cmd to det box to relay data

	bytevalue=struct.pack('>B', sourceSeqLen+2) #how many bytes to tx 
	ser.write(bytevalue)

	ser.write('S'.encode()) #cmd to src box to store sourceSeq

	bytevalue=struct.pack('>B', sourceSeqLen)
	ser.write(bytevalue)

	for idx in range(1, sourceSeqLen+1):
		bytevalue=struct.pack('>B', sourceSeq[idx-1])
		ser.write(bytevalue)

	time.sleep(1e-3)

#Send cmd to turn lasers on
	ser.write('S'.encode())

	bytevalue=struct.pack('>B', 3)
	ser.write(bytevalue)

	ser.write('L'.encode())

	bytevalue=struct.pack('>B', 1) #690nm
	ser.write(bytevalue)

	bytevalue=struct.pack('>B', 1) #830nm
	ser.write(bytevalue)

	time.sleep(1e-3)

#Start data aquisition streaming
	ser.write('r'.encode())




	bamax = 0

	for ii in range(repetitions):
		while ser.inWaiting()<2:
			pass

		rep = ser.read(2)   # unpack requires a buffer of 4 bytes. orig:2
		
		rep = rep[0]*256+rep[1]

		if rep != ii:
			print("Repetition Sequence Error.")

		for jj in range(sourceSeqLen):
			print('Rep:', str(ii), 'Src:', str(jj))
			
			while ser.inWaiting() == 0:
				pass

			seq = ser.read(1)[0]
			
			# seq = int.from_bytes(seq, byteorder='little', signed=False)
			
			if seq != jj:
				print('seq type', type(seq), 'jj type', type(jj))##############
				print('Source Sequence Error.')



			bytestoread = nDets * recsPerSrcPos *53  
			
			br = 0
			

			try:
			    del raw
			except NameError:
				print('We currently do not yet have a variable "raw" defined.We are setting it up now.')
				raw=None




			while br<bytestoread:
###############################################
				#print('br=', br, 'bytestoread=', bytestoread)
				###################################
				ba = ser.inWaiting()

			################################################333
				#print('ba=', ba)
				#######################
				bamax = max(ba, bamax)
				
				if ba>0:
					rdnow = min(ba, bytestoread-br)
					# try:
					# 	raw
					# except NameError:
					# 	raw = None

					if raw is not None:
						#anoRaw=np.zeros((1,br+rdnow))
						temp=ser.read(rdnow)
						for i in range(0, rdnow):
							raw[0][br+i]=temp[i]
							#for i in range(0, rdnow):; raw[0][br+i] = ser.read(rdnow)[i] every time it products a different value to to .read command
						#print(raw.shape,anoRaw.shape)
						#raw=np.concatenate((raw, anoRaw),axis=1)
						# print('concatenation is done for raw.') ####	

						


					else:
						raw=np.zeros((1,int(bytestoread)))
						temp=ser.read(rdnow)
						# for i in range(0, rdnow):
						# 	raw[0][br+i] = ser.read(rdnow)[i]
						for i in range(0, rdnow):
							raw[0][br+i]=temp[i]
						 
					br = br + rdnow
					# print(br,raw.shape[1])####################
				
			print('concatenation is done for raw.') ####	
				
			# np.savetxt('rawFromPy.txt', raw)#######################################
				

			parityOnRaw(raw, bytestoread) # from rawToSig
			rawsep = splitRaw(raw, activeDets, recsPerSrcPos) #from rawToSig.py

			for kk in range(0,20):
				if activeDets[0][kk] == 1:
					R1, R2 = rawToSig(rawsep[kk][:])
					A=A.astype(dtype=np.complex128)
					for s in range(len(R1)):
						A[ii][jj][kk][0][s] = R1[s]
						A[ii][jj][kk][1][s] = R2[s]
	print('Max Serial Buffer Use: '+str(bamax))
	return A

			# idx = 2
			# parity = 0
			# while idx <= bytestoread:
			# 	print('parity check',raw[0][idx-1])##############
			# 	print('type parity', type(parity), 'type raw', type(int(raw[0][idx-1])))
			# 	parity = parity ^ int(raw[0][idx-1])
			# 	idx+=1
			# 	if idx%53 == 0:
			# 		if parity != raw[0][idx-1]:
			# 			print('Parity check error')
			# 		idx = idx+2
			# 		parity = 0


################################### MOVED TO FILE: rawToSig.py #######################################
			# def parityOnRaw(raw, bytestoread): #bitxor in matlab has result in unit8.
			# 	idx = 1
			# 	parity = 0
			# 	while idx <= bytestoread-1:
			# 		parity = np.uint8(parity)^np.uint8(raw[0][idx])
			# 		# print('parity value', parity)
			# 		idx+=1
			# 		if (idx+1)%53 == 0:
			# 			if type(parity) is type(np.uint8(raw[0][idx])):
			# 				# print('no type error,parity and raw values are:', parity, raw[0][idx]) ###########################
			# 				if parity != np.uint8(raw[0][idx]):

			# 					print('PARITY CHECK ERROR. parity and raw values are:', parity, raw[0][idx])
			# 				# if parity == np.uint8(raw[0][idx]):
			# 				# 	print('No parity check error.')

			# 			# else:
			# 			# 	print('The type of parity is different than the type of raw element')

			# 			idx += 2
			# 			parity = 0
###################################################################################################################
						
			# parityOnRaw(raw, bytestoread) # from rawToSig





############################ MOVED TO FILE:rawToSig.py####################
			# def splitRaw(raw, activeDets, recsPerSrcPos):
			# 	nr = 52
			# 	R = np.zeros((20,recsPerSrcPos*nr))
			# 	i = 1
			# 	det = 0
			# 	sc = np.zeros((1,20)) #sampe count
			# 	# print('activeDet is', activeDets)###############
			# 	while activeDets[0][det] == 0:#advance to first active detector
			# 		det += 1
			# 	# print('det after while loop is: ',det)
			# 	# print('type of det:',type(det))

			# 	# print('raw is', raw)#################3
			# 	# print('raw shape', raw.shape)###############
			# 	while (i+nr) < raw.shape[1] and (int(sc[0][det]+1)*nr) < len(R[det]):
			# 		if int(raw[0][i-1]) != det:
			# 			print('Detector Sequence Error')

			# 		# print(int(sc[0][det]*nr))
			# 		# print(((int(sc[0][det]+1)*nr)))

			# 		# print(len(R[det]))
						
			# 		# print(R[det][int(sc[0][det]*nr) : ((int(sc[0][det]+1)*nr))])##########3
			# 		# print(raw[0][i:(i+nr)])#####################

			# 		R[det][int(sc[0][det]*nr) : ((int(sc[0][det]+1)*nr))] = raw[0][i:(i+nr)]
			# 		sc[0][det] = sc[0][det]+1
			# 		i=i+nr+1
			# 		det = (det+1)%20

			# 		while activeDets[0][det] == 0:
			# 			det = (det+1)%20
			# 	return R 
###########################################################################
			# rawsep = splitRaw(raw, activeDets, recsPerSrcPos) #from rawToSig.py

			# print('This works', rawsep)




########################### MOVED TO FILE: rawToSig.py ##############################
			# def fiveBytesToFloat(iN):
			# 	# print('iN shape is', iN.shape)#####################
			# 	if iN.shape == (5,):
			# 		out = iN[4]*(256**4) + iN[3]*(256**3) + iN[2]*(256**2) + iN[1]*256 + iN[0]
			# 		if out > ((2**35)-1):
			# 			out = out - 2**36
			# 	else:
			# 		print('input not in required format')
			# 	return out
###################################################################################
			# test=fiveBytesToFloat(np.array([[1,2,2,5,6]]))
			# print('output for fiveBytesToFloatFunc', test)
			# iN=np.array([[1,2,3,4,5,6,7,8,9]])
			# stage1=fiveBytesToFloat(iN[0][0:5])
			# print('test stage1', stage1)############################33




############################ MOVED TO FILE rawToSig.py ########################
			# def decodeSecondFreq(iN, kN):
			# 	# print("iN shape", iN.shape)#####################
			# 	# if iN.shape != (1,30):
			# 	if iN.shape != (30,):
			# 		print('input not in required format')

			# 	# print(iN[0][0:5])###################################3
			# 	# print(iN[0][0:5][3])#####################################33
			# 	# stage1 = fiveBytesToFloat(iN[0][0:5])
			# 	stage1 = fiveBytesToFloat(iN[0:5])

			# 	# print('stage1', stage1)#######################################
			# 	# stage2 = fiveBytesToFloat(iN[0][5:10])
			# 	stage2 = fiveBytesToFloat(iN[5:10])
			# 	# stage3 = fiveBytesToFloat(iN[0][10:15])
			# 	stage3 = fiveBytesToFloat(iN[10:15])
			# 	# stage4 = fiveBytesToFloat(iN[0][15:20])
			# 	stage4 = fiveBytesToFloat(iN[15:20])

			# 	# print('stage4', stage4)###############3
			# 	# stage5 = fiveBytesToFloat(iN[0][20:25])
			# 	# stage6 = fiveBytesToFloat(iN[0][25:30])
			# 	stage5 = fiveBytesToFloat(iN[20:25])
			# 	stage6 = fiveBytesToFloat(iN[25:30])

			# 	stage0 = -1 * stage6;
			# 	# print('stage0 is not supporse to be None', stage0)#####################

			# 	out = stage0 + stage1 * np.exp(-complex(0,1)*2*math.pi*kN) + stage2 * np.exp(-complex(0,1)*2*math.pi*2*kN) - stage3 * np.exp(complex(0,1)*2*math.pi*3*kN) - stage4 * np.exp(complex(0,1)*2*math.pi*2*kN) - stage5 * np.exp(complex(0,1)*2*math.pi*kN)
			# 	return complex(out)

###############################################################################################
			# test=decodeSecondFreq(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]), 10)
			# print('test', test)


			


########################## MOVED TO FILE: rawToSig.py #############################
			# def decodeFirstFreq(iN, kN):
			# 	# if iN.shape != (1,20):
			# 	if iN.shape != (20,):
			# 		print('input not in required format and the size of the input is', iN.shape)

			# 	stage1 = fiveBytesToFloat(iN[0:5])
			# 	stage2 = fiveBytesToFloat(iN[5:10])
			# 	stage3 = fiveBytesToFloat(iN[10:15])
			# 	stage4 = fiveBytesToFloat(iN[15:20])
			# 	# stage1 = fiveBytesToFloat(iN[0][0:5])
			# 	# stage2 = fiveBytesToFloat(iN[0][5:10])
			# 	# stage3 = fiveBytesToFloat(iN[0][10:15])
			# 	# stage4 = fiveBytesToFloat(iN[0][15:20])
			# 	stage0 = -1 * stage4

			# 	out = stage0 + stage1 * np.exp(-complex(0,1)*2*math.pi*kN) - stage2 * np.exp(complex(0,1)*2*math.pi*2*kN) - stage3 * np.exp(complex(0,1)*2*math.pi*kN)
			# 	return out

#################################################################################



######################### MOVED TO FILE: rawToSig.py #########################################
			# def rawToSig(A):
			# 	N = 4e6 #sample length of DFT block
			# 	k1N = 3/8 #DFT bin of first modulation frequency
			# 	k2N = 5/12 #DFT bin of second modulation frequency
			# 	db1k = 1999998  # samples between even and odd blocks

			# 	pkLen = 52
			# 	print('Shape of A:',A.shape)
			# 	print('Type of A:',type(A))
			# 	nBytes = A.shape[0]###########################################################33
			# 	print('nBytes is', nBytes)#####################

			# 	nOvfl = 0
			# 	nStageOvfl = 0


			# 	cntr = A[0]%64 #################

			# 	R1=[]
			# 	R2=[]

			# 	for idx in range(1, nBytes+1, pkLen):
			# 		if A[idx-1]%64 == cntr:
			# 			cntr = (cntr+1)%64
			# 		else:
			# 			print('Index error. Expected= '+str(cntr)+'Read =  '+str(A[idx-1]%64))
			# 		if A[idx-1]%128 >= 64:
			# 			nOvfl += 1
			# 		if A[idx-1]%2 == 0:

			# 			# print('decodeFirstFreq Func', A[idx:(idx+20)], k1N)##########################3
			# 			# print('decodeFirstFreq result', decodeFirstFreq(A[idx:(idx+20)], k1N))################
			# 			R1.append(decodeFirstFreq(A[idx:(idx+20)], k1N))
			# 			R2.append(decodeSecondFreq(A[(idx+20):(idx+50)], k2N))
			# 		else:
			# 			# va=np.exp(complex(0,1)*2*math.pi*k1N*db1k)#############33
			# 			# print('value', va)##########################3
			# 			# val=decodeFirstFreq(A[idx:(idx+20)], k1N)###########33
			# 			# print('value from func', val)################33
			# 			# print('multi for unsup operand', va*val)######################

			# 			R1.append(decodeFirstFreq(A[idx:(idx+20)], k1N)*np.exp(complex(0,1)*2*math.pi*k1N*db1k))
			# 			R2.append(decodeSecondFreq(A[(idx+20):(idx+50)], k2N)*np.exp(complex(0,1)*2*math.pi*k2N*db1k))
							
			# 	if (nOvfl>0) or (nStageOvfl>0):
			# 		print(str(nOvfl)+' blocks contain ADC overflows')
			# 		print(str(nStageOvfl)+' stages are within factor 2 of limit')

			# 	return R1, R2

###################################################################################################





############################# MOVED ABOVE ##################################
	# 		for kk in range(0,20):
	# 			if activeDets[0][kk] == 1:


	# 				# print(A.shape)
	# 				# print(type(A))#############################
	# 				# print(A.shape[1])

	# 				R1, R2 = rawToSig(rawsep[kk][:])

	# 				# print('R1 is', R1)
	# 				# print('R2 is', R2)###########333
	# 				# print('R1 type', type(R1[0]))##############
	# 				A=A.astype(dtype=np.complex128)
	# 				# print('change type of A')##########3
	# 				# print('A type', type(A), 'and A element type is', type(A[0][0][0][1][0]))###########
	# 				for s in range(len(R1)):
	# 					A[ii][jj][kk][0][s] = R1[s]
	# 					A[ii][jj][kk][1][s] = R2[s]

	# 				# A[ii][jj][kk][0][:] = R1
	# 				# A[ii][jj][kk][1][:] = R2
	# print('Max Serial Buffer Use: '+str(bamax))
	# # print('Shape of A: ',A.shape)#############################
	# return A

  

					# def rawToSig(M):
					# 	N = 4e6
					# 	k1N = 3/8
					# 	k2N = 5/12
					# 	db1k = 1999998

					# 	pkLen = 52

					# 	nBytes = M.shape[1]

					# 	nOvfl = 0
					# 	nStageOvfl = 0


					# 	cntr = M[0][0][0][0][0]%64

					# 	R1=[]
					# 	R2=[]

						# def fiveBytesToFloat(iN):
						# 	if iN.shape == (1,5):
						# 		out = iN[4][0][0][0][0]*(256**4) + iN[3][0][0][0][0]*(256**3) + iN[2][0][0][0][0]*(256**2) + iN[1][0][0][0][0]*256 + iN[0][0][0][0][0]
						# 		if out > ((2**35)-1):
						# 			out = out - 2**36
						# 	else:
						# 		print('input not in required format')
						# 	return out


						# def decodeFirstFreq(iN, kN):
						# 	if iN.shape != (1,20):
						# 		print('input not in required format')
						# 	stage1 = fiveBytesToFloat(iN[0][0:4])
						# 	stage2 = fiveBytesToFloat(iN[0][5:9])
						# 	stage3 = fiveBytesToFloat(iN[0][10:14])
						# 	stage4 = fiveBytesToFloat(iN[0][15:19])

						# 	stage0 = - stage4

						# 	out = stage0 + stage1 * math.exp(-complex(0,1)*2*math.pi*kN) - stage2 * math.exp(complex(0,1)*2*math.pi*2*kN) - stage3 * math.exp(complex(0,1)*2*math.pi*kN)
						# 	return out

						# def decodeSecondFreq(iN, kN):
						# 	if iN.shape != (1,30):
						# 		print('input not in required format')

						# 	stage1 = fiveBytesToFloat(iN[0][0:4])
						# 	stage2 = fiveBytesToFloat(iN[0][5:9])
						# 	stage3 = fiveBytesToFloat(iN[0][10:14])
						# 	stage4 = fiveBytesToFloat(iN[0][15:19])
						# 	stage5 = fiveBytesToFloat(iN[0][20:24])
						# 	stage6 = fiveBytesToFloat(iN[0][25:29])

						# 	stage0 = - stage6;

						# 	out = stage0 + stage1 * math.exp(-complex(0,1)*2*math.pi*kN) + stage2 * math.exp(-complex(0,1)*2*math.pi*2*kN) - stage3 * math.exp(complex(0,1)*2*math.pi*3*kN) - stage4 * math.exp(complex(0,1)*2*math.pi*2*kN) - stage5 * math.exp(complex(0,1)*2*math.pi*kN)
						# 	return out


						# for idx in range(1, nBytes+1, pkLen):
						# 	if A[idx-1][0][0][0][0]%64 == cntr:
						# 		cntr = (cntr+1)%64
						# 	else:
						# 		print('Index error. Expected= '+str(cntr)+'Read =  '+str(A[idx-1][0][0][0][0]%64))
						# 	if A[idx-1][0][0][0][0]%128 >= 64:
						# 		nOvfl += 1
						# 	if A[idx-1][0][0][0][0]%2 == 0:
						# 		R1.append(decodeFirstFreq(A[idx:(idx+19), 0, 0, 0, 0], k1N))
						# 		R2.append(decodeSecondFreq(A[(idx+20):(idx+49), 0, 0, 0, 0], k2N))
						# 	else:
						# 		R1.append(decodeFirstFreq(A[idx:(idx+19), 0, 0, 0, 0], k1N))*math.exp(complex(0,1)*2*math.pi*k1N*db1k)
						# 		R2.append(decodeSecondFreq(A[(idx+20):(idx+49), 0, 0, 0, 0], k2N))*math.exp(complex(0,1)*2*math.pi*k2N*db1k)
							
						# if (nOvfl>0) or (nStageOvfl>0):
						# 	print(str(nOvfl)+'blocks contain ADC overflows')
						# 	print(str(nStageOvfl)+'stages are within factor 2 of limit')

						# return R1, R2

					
	# 				R1, R2 = rawToSig(rawsep[kk][:])
	# 				A[ii][jj][kk][0][:] = R1
	# 				A[ii][jj][kk][1][:] = R2
	# print('Max Serial Buffer User: '+str(bamax))
	# return A

 #  
