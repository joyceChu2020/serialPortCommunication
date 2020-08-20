import sys #to clear up all variables
sys.modules[__name__].__dict__.clear()

import serial
import matplotlib.pyplot as plt #for close all figures
import numpy as np
from readRawdata import *
from scipy import signal
import array
import collectData 


plt.close('all')


acport = 'COM3'
baudRate=500000
inputBufferSize=100000
ser=serial.Serial(acport, baudRate)

# def accboard():
# 	if (ser.isOpen() == False):
# 		ser.open()
# 	print("Serial port is open now")
if (ser.isOpen() == False):
	ser.open()
print("Serial port is open now")


def readDetTemperatures(ser):
	if ser.inWaiting()>0: #fread in matlab is a function to read binary data
		ser.read(ser.inWaiting())

	ser.write('t'.encode())
	raw = list(ser.read(60))
	#print(raw)
	#print()



	temperatures = np.zeros((1,20))
	for ii in range(20):
		#print(raw[0])

		if raw[ii*3] != ii: #weird strings
			print("Temperature index error")

		if raw[ii*3+1]<32:
			temperatures[0][ii] = (raw[ii*3+1]*256+raw[ii*3+2])/32
		else:
			temperatures[0][ii] = (raw[ii*3+1]*256+raw[ii*3+2]-16384)/32
	return temperatures

arr=readDetTemperatures(ser)
nonzero_index=np.nonzero(arr)
nonzero_elements=arr[np.nonzero(arr)]

print('Temperatures:{}deg C'.format(arr[nonzero_index]))#, type(arr))
#print()
#print(nonzero_index, type(nonzero_index))

print()
print('Cards present:{}'.format(nonzero_index[1]))




#turn lasers on/off skipped
#for test_card in [11, 15, 16]:
for test_card in nonzero_index[1]:

	if test_card<0 or test_card>19:
		print("card number out of range")
	print("Reading data from card:"+str(test_card))

	valuestoread=16380
	ba=ser.inWaiting()
	if ba>0:
		ser.read(ba)

	ser.write('u'.encode(encoding='ascii'))
	bytevalue=struct.pack('>B', test_card)
	ser.write(bytevalue)

	# while ser.in_waiting<valuestoread*2:
	# 	print("Value is less that 16380")

	raw=ser.read(valuestoread*2)
	
	rawdata=np.zeros((1, valuestoread))


	for ii in range(valuestoread+1):

		rawdata[0][ii-1]=raw[ii*2-2]*256+raw[ii*2-1]

		if rawdata[0][ii-1]>(2**15-1):
			rawdata[0][ii-1]=rawdata[0][ii-1]-2**16

	print('rawdata is completed')		




	#a=readRawdata(ser, test_card-1)
	x,y=signal.welch(rawdata, 180e6)


# 	plt.figure()
# 	plt.plot(x,y.transpose())
# 	plt.title(test_card)

# plt.show()








actdet=np.zeros((1,20))
actdet[0][10]=1;

# C=collectData.collectData(ser,1,[0],actdet,90)
C=collectData.collectData(ser,1,[1],actdet,90)

# plt.figure()
# plt.plot(np.abs(A[0][0][10][0][:]))

plt.show()
plt.figure()
plt.plot(abs(np.squeeze(C[0][0][15][0])))
plt.show()

