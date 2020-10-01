import struct

def readRawdata(ser, card):
	
	if card < 0 or card>19:
		print("card number out of range")

	print('Reading data from card: '+str(card))

	valuestoread=16380

	ba = ser.in_waiting

	if ba>0:
		ser.read(ba)

	ser.write('u'.encode(encoding='ascii'))

	bytevalue=struct.pack('>B', card)

	ser.write(bytevalue)

	while ser.in_waiting < valuestoread*2:
		pass

	raw = ser.read(valuestoread*2)
	rawdata = np.zeros((1,valuestoread))

	for ii in range(valuestoread+1):
		rawdata[ii-1]=raw[ii*2-2]*256 + raw[ii*2-1]
		if rawdata[ii-1] > (2**15-1):
			rawdata[ii-1] = rawdata[ii-1]-2**16

	return rawdata
