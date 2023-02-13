# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 01:31:49 2019

@author: diffabs
"""

# -*- coding: utf-8 -*-
"""
@author: MOCUTA
"""




from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import numpy
import tables, h5py, os
from scipy.optimize import curve_fit  # fitting
def integration_radiale_du_pauvre(first_img, last_img, step):
    fileNameRoot = "scan_0"
    # C:\Users\diffabs\Documents\Renault\2011\DataNxs
    pathRoot = "C:/Users/Phymat/Documents/renault/soleil2207/"
    inputDataFolder = pathRoot + "DataNxs/"
    pathData = inputDataFolder
    GLOBAL_xpad_dataset = "28"
    plt.figure()
    ctr = 0
    nscans = len(np.arange(first_img, last_img+1, step))
    for scanNo in np.arange(first_img, last_img+1, step):
        fileName = fileNameRoot + str(scanNo) + "_0001.nxs"
        file1 = tables.open_file(pathData+fileName)
        fileNameRoot1 = file1.root._v_groups.keys()[0]
        command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
        xpadImage = eval(command+".scan_data.data_" +
                         GLOBAL_xpad_dataset+".read()")
        # deltaArray = eval(command+".scan_data.data_01.read()")
        # deltaArray =eval(command+".scan_data.actuator_1_1.read()")
        file1.close()
        xmean = np.zeros((240, 560))
        for iii in np.arange(xpadImage.shape[0]):
            xmean += xpadImage[iii]
        xmean /= xpadImage.shape[0]
        plt.plot(xmean.sum(axis=0)-500*ctr, label=str(scanNo),
                 color=plt.cm.jet(ctr*1./nscans))
        ctr += 1


def flatscan(first_scan, last_scan, path, file_extension, dataset, plot_flag, save_flag, destination_save):
	final_sum = numpy.zeros((240, 560), dtype=numpy.int32)
	try:
		# filename = destination_save + "flatscan_slow_xpad_sum.raw"
		# filename = destination_save + "flatscan_sum112-116.raw"
		# filename = destination_save + "flatscan_sum.raw"#meilleur en 2109
		filename = destination_save + "BONflatscan_sum.raw"  # moins bon en 2109
		final_sum = numpy.reshape(numpy.fromfile(
		    filename, dtype=numpy.int32), (240, 560))
		print('flat trouve')
		# with h5py.File(filename, 'r') as f:
		#	final_sum += f['sum_xpad'][0]
	except(IOError):
		final_sum = numpy.zeros((240, 560), dtype=numpy.int32)
		for i in range(last_scan-first_scan + 1):
			filename = "scan_000" + str(first_scan + i) + file_extension
			filename = os.path.join(path, filename)
			print("Searching images in " + filename + " file.")
			flat_ouv = h5py.File(filename, 'r')
			group_flat = list(flat_ouv.keys())[0]
			xpad_images = flat_ouv[group_flat]['scan_data/data_31'][()]
			# file1 = tables.open_file(filename)
			# fileNameRoot1 = file1.root._v_groups.keys()[0]
			# command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
			# attention à modifier si le fichier est trop GROS
			# xpad_images = eval(command+".scan_data.data_"+dataset+".read()")#xpad_images = eval(command+".scan_data.data_"+dataset+".read(0,10)")
			for data in xpad_images:
				final_sum += data
		if save_flag:
			print("Writing flatscan_sum.raw in : " + destination_save)
			final_sum.astype('int32').tofile(os.path.join(
			    destination_save, "flatscan_sum_new.raw"))
	if plot_flag:
		plt.figure()
		plt.imshow(final_sum)
		plt.title("Flatscan")
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()
	return final_sum


def visu_one_rawImage(scanNo, imgIndex, fileNameRoot, inputDataFolder, logScale, mini, maxi):
	# visualise the raw XPAD data (image) of the n-th point in the scan (n = imgIndex)
	# NB: 1st image has index 0 !!!
	# use negative values for mini and maxi to keep autoscale
	pathData = inputDataFolder
	fileName = fileNameRoot + str(scanNo) + "_0001.nxs"
	GLOBAL_xpad_dataset = "31"

	# will properly close it once the indented code is executed
	with h5py.File(pathData+fileName, 'r') as f:
	        # group1 = f.get(f.keys()[0])
	        # deltaArray = numpy.array(group1['DIFFABS/D13-1-CX1__EX__DIF.1-DELTA__#1/raw_value'])
	        # gam = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value/'])
	        # gam = f[f'scan_{self.ScanNb:0>4d}/DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value/'][()][0]
            group1 = list(f.keys())[0]
            gam = f[group1]['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value/'][()]
            xpadImage = f[f'scan_{scanNo:0>4d}/scan_data/data_{GLOBAL_xpad_dataset}'][()]
            deltaArray = f[f'scan_{scanNo:0>4d}/scan_data/data_08'][()]

	# reading images in the .nxs file
	# file1 = tables.open_file(pathData+fileName)
	# fileNameRoot1 = file1.root._v_groups.keys()[0]
	# command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
	# xpadImage = eval(command+".scan_data.data_"+GLOBAL_xpad_dataset+".read()")
	# deltaArray = eval(command+".scan_data.data_14.read()")#06

	#deltaArray =eval(command+".scan_data.actuator_1_1.read()")#
	print('deltaArray=%s' % (deltaArray))
	# file1.close()

	print("looking at img no. %d (file=%s)   delta=%f" %
	      (imgIndex, fileName, deltaArray[imgIndex]))
	thisImg = 0.0+xpadImage[imgIndex]

	for imgIndex in range(0, np.shape(deltaArray)[0], 1):
		if logScale:
			thisImg = 0.0+xpadImage[imgIndex]
			thisImg = numpy.log10(thisImg)
			thisImg[numpy.isinf(thisImg)] = 0
			print("using LOGscale")
		if mini < 0 and maxi < 0:
			mini = numpy.min(thisImg); maxi = numpy.max(thisImg);
		print("using min/max (%f %f)" % (mini, maxi))
		fig = plt.figure()
		plt.imshow(thisImg, vmin=mini, vmax=maxi, cmap="jet")
		plt.colorbar()
		plt.xlabel("x-coord (pixels)"); plt.ylabel("y-coord (pixels)");
		plt.title("scanNo=%d pointIndex = %d\nTemp=%.1f" %
		          (scanNo, imgIndex, deltaArray[imgIndex]))
		fig.savefig("FIT_%d_%d.png" % (scanNo, imgIndex))
		plt.show()
		# plt.close()
	'''
	for imgIndex in range (0,np.shape(deltaArray)[0],1):
		if logScale:
			thisImg = 0.0+xpadImage[imgIndex]
			thisImg = numpy.log10(thisImg)
			thisImg[numpy.isinf(thisImg)] = 0
			print("using LOGscale")
		if mini <0 and maxi <0:
			mini = numpy.min(thisImg); maxi = numpy.max(thisImg);
		print("using min/max (%f %f)"%(mini, maxi))
		fig=plt.figure()
		plt.plot(np.sum(xpadImage[imgIndex], axis=0),'o')
		plt.xlabel("x-coord (pixels)"); plt.ylabel("y-coord (pixels)");
		plt.title("scanNo=%d pointIndex = %d\nTemp=%.1f" %
		          (scanNo, imgIndex, deltaArray[imgIndex]))
		fig.savefig("fig_%d_%d.png"%(scanNo, imgIndex))
		plt.show()
		# plt.close()
  	'''


def extract_and_correct_1scan(scanNo, flatImg, fileNameRoot, inputDataFolder, outputPath):
	GLOBAL_xpad_dataset = "31"
	GLOBAL_flat_dataset = "31"
	print("scanNo = %s" % (scanNo))

	# flat field correction
	# make sure it is properly resetted to 1, in case it was not modified in the header
	factorIdoublePixel = 1.0
	'''
	fileNameFlatRoot = "scan_0" #root name of the files containing the flat data
	flat_scanStart = 15464; flat_scanEnd = 15464;
	pathFlat = pathRoot + "flat/"
	# print "   ... looking at flatFiled image"
	flatImg = numpy.zeros((240, 560))
	for indexFlat in range (flat_scanStart, flat_scanEnd + 1):
		fileNameFlat = fileNameFlatRoot+str(indexFlat)+"_0001.nxs"
		file1Flat = tables.open_file(pathFlat+fileNameFlat)
		print("   ... looking at file flat = %s" %(fileNameFlat))
		fileNameRoot2Flat = file1Flat.root._v_groups.keys()[0]
		commandFlat = "file1Flat.root.__getattr__(\""+str(fileNameRoot2Flat)+"\")"
		imagesFlat = eval(commandFlat+".scan_data.data_"+ \
		                  GLOBAL_flat_dataset+".read()") # xpad images
		for imgIndex in range (0, imagesFlat.shape[0]):
			flatImg += imagesFlat[imgIndex]
		file1Flat.close()
	'''

	flatImg = 1.0*flatImg / flatImg.mean()
	flatImg_inv = 1.0/flatImg
	flatImg_inv[numpy.isnan(flatImg_inv)] = -10000000
	flatImg_inv[numpy.isinf(flatImg_inv)] = -10000000
	# plt.figure();plt.plot(np.sum(flatImg, axis=0));plt.show()
	# print "   ... DONE looking at flatFiled image"

	# geometry informations*********************************
	calib = 91.8;  # pixels in 1 deg.cabib du 210923
	# position of direct beam on xpad at (delltaOffset, gamOffset). Use the 'corrected' positions (add 3 pixels whenever cross 80*i in X and 120 in Y)
	XcenDetector = 274.0+3*3; YcenDetector = 118.0 
	# positions in diffracto angles for which the above values XcenDetector, YcenDetectors are reported
	deltaOffset = 13.50+2.87; gamOffset = 0.0;
	numberOfModules = 2; numberOfChips = 7;  # detector dimension, XPAD S-140
	# chip dimension, in pixels (X = horiz, Y = vertical)
	chip_sizeX = 80; chip_sizeY = 120;
	# adding 3 more lines, corresponding to the double pixels on the last and 1st line of the modules
	lines_to_remove_array = [0, -3];
	# ******************************************************

	deg2rad = numpy.pi/180; inv_deg2rad = 1/deg2rad;
	# distance xpad to sample, in pixel units
	distance = calib/numpy.tan(1.0*deg2rad);
	print("sample-detector distance = %f pixels	= %f mm" %
	      (distance, distance*0.13))

	# calculate the total number of lines to remove from the image
	# initialize to 0 for calculating the sum. For xpad 3.2 these lines (negative value) will be added
	lines_to_remove = 0;
	for i in range(0, numberOfModules):
		lines_to_remove += lines_to_remove_array[i]
	# size of the resulting (corrected) image
	image_corr1_sizeY = numberOfModules * chip_sizeY - lines_to_remove;
	image_corr1_sizeX = (numberOfChips-1)*3+numberOfChips * \
	                     chip_sizeX;  # considers the 2.5x pixels

	# ---------- double pix corr ---------
	# =====================================
	newX_array = numpy.zeros(
	    image_corr1_sizeX); newX_Ifactor_array = numpy.zeros(image_corr1_sizeX)
	for x in range(0, 79):  # this is the 1st chip (index chip = 0)
		newX_array[x] = x;
		newX_Ifactor_array[x] = 1  # no change in intensity

	newX_array[79] = 79; newX_Ifactor_array[79] = 1/factorIdoublePixel;
	newX_array[80] = 79; newX_Ifactor_array[80] = 1/factorIdoublePixel;
	newX_array[81] = 79; newX_Ifactor_array[81] = -1

	for indexChip in range(1, 6):
		temp_index0 = indexChip * 83
		for x in range(1, 79):  # this are the regular size (130 um) pixels
			temp_index = temp_index0 + x;
			newX_array[temp_index] = x + 80*indexChip;
			newX_Ifactor_array[temp_index] = 1;  # no change in intensity
		newX_array[temp_index0] = 80*indexChip; newX_Ifactor_array[temp_index0] = 1 / \
		    factorIdoublePixel;  # 1st double column
		newX_array[temp_index0-1] = 80 * \
		    indexChip; newX_Ifactor_array[temp_index0-1] = 1/factorIdoublePixel;
		newX_array[temp_index0+79] = 80*indexChip+79; newX_Ifactor_array[temp_index0 +
		    79] = 1/factorIdoublePixel;  # last double column
		newX_array[temp_index0+80] = 80*indexChip + \
		    79; newX_Ifactor_array[temp_index0+80] = 1/factorIdoublePixel;
		newX_array[temp_index0+81] = 80*indexChip + \
		    79; newX_Ifactor_array[temp_index0+81] = -1;

	for x in range(6*80+1, 560):  # this is the last chip (index chip = 6)
		temp_index = 18 + x;
		newX_array[temp_index] = x;
		newX_Ifactor_array[temp_index] = 1;  # no change in intensity

	newX_array[497] = 480; newX_Ifactor_array[497] = 1/factorIdoublePixel;
	newX_array[498] = 480; newX_Ifactor_array[498] = 1/factorIdoublePixel;

	newY_array = numpy.zeros(image_corr1_sizeY);  # correspondance oldY - newY
	# will keep trace of module index
	newY_array_moduleID = numpy.zeros(image_corr1_sizeY);

	newYindex = 0;
	for moduleIndex in range(0, numberOfModules):
		for chipY in range(0, chip_sizeY):
			y = chipY + chip_sizeY*moduleIndex;
			newYindex = y - lines_to_remove_array[moduleIndex]*moduleIndex;
			newY_array[newYindex] = y;
			newY_array_moduleID[newYindex] = moduleIndex;

	print("   ... done double pixel spreading")

	# END---------- double pix corr ---------
	# ==========================================

	# create the folder to save data if it does not exist
	try:
		os.stat(outputPath)
	except:
		os.mkdir(outputPath)
	fileSavePath = outputPath + "scan_%d/" % (scanNo)
	try:
		os.stat(fileSavePath)
	except:
		os.mkdir(fileSavePath)

	pathData = inputDataFolder
	fileName = fileNameRoot + str(scanNo) + "_0001.nxs"

	# read here the scan informations (delta, gamma)
	# will properly close it once the indented code is executed
	with h5py.File(pathData+fileName, 'r') as f:
	        # group1 = f.get(f.keys()[0])
	        # deltaArray = numpy.array(group1['DIFFABS/D13-1-CX1__EX__DIF.1-DELTA__#1/raw_value'])
	        # gam = numpy.array(group1['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value/'])

            group1 = list(f.keys())[0]
            gam = f[group1]['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value'][()]
            # xpadImage = f[f'scan_{scanNo:0>4d}/scan_data/data_{GLOBAL_xpad_dataset}'][()]
            # deltaArray = f[f'scan_{scanNo:0>4d}/scan_data/data_08'][()]

            xpadImage = f[group1]['scan_data/data_31'][()]
            # le tableau de valeur delta
            deltaArray = f[group1]['scan_data/data_11'][()]
            # une seule valeur
            delta = f[group1]['DIFFABS/d13-1-cx1__ex__dif.1-delta/raw_value'][()]

	'''
	# reading images in the .nxs file
	file1 = tables.open_file(pathData+fileName)
	fileNameRoot1 = file1.root._v_groups.keys()[0]
	command = "file1.root.__getattr__(\""+str(fileNameRoot1)+"\")"
	# attention à modifier si le fichier est trop GROS
	# xpad_images = eval(command+".scan_data.data_"+dataset+".read(0,1000)")#xpad_images = eval(command+".scan_data.data_"+dataset+".read()")
	xpadImage = eval(command+".scan_data.data_"+GLOBAL_xpad_dataset+".read()")


	# deltaArray = eval(command+".scan_data.trajectory_1_1.read()")
	# deltaArray = list(essai[u'scan_0096/DIFFABS/d13-1-cx1__ex__dif.1-delta/raw_value'])
	deltaArray = eval(command+".scan_data.data_14.read()")#06
	# deltaArray = eval(command+".scan_data.trajectory_1_1.read()")#c est bien mais c'est moins precis que de lire la vraie valeur
	print(deltaArray)
	print('deltaArray shape=%s'%(deltaArray.shape[0]))
	print(deltaArray.shape)

	file1.close()
	'''
	print(deltaArray.shape[0])
	print('tuc')
	for pointIndex in range(deltaArray.shape[0]):
		print
        # extracting the XY coordinates for the rest of the scan transformation
        # ========psiAve = 1, deltaPsi = 1=============================================
		print("... looking at point %d out of %d: delta=%.3f gam=%.3f" %
		      (pointIndex, deltaArray.shape[0], deltaArray[pointIndex], gam))
		delta = deltaArray[pointIndex]

		diffracto_delta_rad = (delta+deltaOffset)*deg2rad;
		sindelta = numpy.sin(diffracto_delta_rad); cosdelta = numpy.cos(
		    diffracto_delta_rad);
		diffracto_gam_rad = (gam+gamOffset)*deg2rad;
		singamma = numpy.sin(diffracto_gam_rad); cosgamma = numpy.cos(
		    diffracto_gam_rad);

		# the array thisCorrectedImage contains the corrected image (double pixels corrections)
		twoThArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
		psiArray = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))

		x_line = numpy.linspace(0, image_corr1_sizeX-1, image_corr1_sizeX)
		x_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
		for a in range(image_corr1_sizeY):
			x_matrix[:, a] = x_line[:]

		y_line = numpy.linspace(0, image_corr1_sizeY-1, image_corr1_sizeY)
		y_matrix = numpy.zeros((image_corr1_sizeX,  image_corr1_sizeY))
		for a in range(image_corr1_sizeX):
			y_matrix[a, :] = y_line[:]

		corrX = distance;  # for xpad3.2 like
		corrZ = YcenDetector-y_matrix;  # for xpad3.2 like
		corrY = XcenDetector-x_matrix;  # sign is reversed
		tempX = corrX; tempY = corrZ*(-1.0); tempZ = corrY;

		x1 = tempX*cosdelta - tempZ*sindelta;
		y1 = tempY;
		z1 = tempX*sindelta + tempZ*cosdelta;
		# apply Rz(-gamma); due to geo consideration on the image, the gamma rotation should be negative for gam>0
		# apply the same considerations as for the delta, and keep gam values positive
		corrX = x1*cosgamma + y1*singamma;
		corrY = -x1*singamma + y1*cosgamma;
		corrZ = z1;
		# calculate the square values and normalization
		corrX2 = corrX*corrX; corrY2 = corrY*corrY; corrZ2 = corrZ*corrZ;
		norm = numpy.sqrt(corrX2 + corrY2 + corrZ2);
		# calculate the corresponding angles
		# delta = angle between vector(corrX, corrY, corrZ) and the vector(1,0,0)
		thisdelta = numpy.arccos(corrX/norm)*inv_deg2rad;
		# psi = angle between vector(0, corrY, corrZ) and the vector(0,1,0)

		sign = numpy.sign(corrZ);
		cos_psi_rad = corrY/numpy.sqrt(corrY2+corrZ2);
		psi = numpy.arccos(cos_psi_rad)*inv_deg2rad*sign;

		psi[psi < 0] += 360
		psi -= 90;
		psiArray = psi.T
		twoThArray = thisdelta.T
		# end geometry

		# dealing now with the intensitiespointIndex
		thisSize = image_corr1_sizeX*image_corr1_sizeY
		thisImage = xpadImage[pointIndex]
		thisImage = flatImg_inv * thisImage
		thisImage = ndimage.median_filter(thisImage, 3)
		# plt.figure();plt.imshow(thisImage, label='ligne360');plt.title("flatscan_sum.raw");plt.legend();plt.colorbar();plt.show()
		"""
		# masking the bad pixels
		thisImage[110:122, 373:385] =  -1000000000
		thisImage[112:118, 541:547] =  -1000000000
		"""

		thisCorrectedImage = numpy.zeros((image_corr1_sizeY, image_corr1_sizeX))
		Ifactor = newX_Ifactor_array  # x
		newY_array = newY_array.astype('int')
		newX_array = newX_array.astype('int')

		for x in range(0, image_corr1_sizeX):
			thisCorrectedImage[:, x] = thisImage[newY_array[:], newX_array[x]]
			if Ifactor[x] < 0:
				# print "%s %s" %(x, Ifactor[x])
				thisCorrectedImage[:, x] = (thisImage[newY_array[:], newX_array[x]-1] +
				                            thisImage[newY_array[:], newX_array[x]+1])/2.0/factorIdoublePixel
		thisCorrectedImage[numpy.isnan(thisCorrectedImage)] = -100000

		# correct the double lines (last and 1st line of the modules, at their junction)
		lineIndex1 = chip_sizeY-1;  # last line of module1 = 119, is the 1st line to correct
		# 1st line of module2 (after adding the 3 empty lines), becomes the 5th line tocorrect
		lineIndex5 = lineIndex1 + 3 + 1;
		lineIndex2 = lineIndex1+1; lineIndex3 = lineIndex1+2; lineIndex4 = lineIndex1+3;
		# thisSize = image_corr1_sizeX*image_corr1_sizeY #out of the loop
		# IntensityArray = numpy.zeros(thisSize)

		i1 = thisCorrectedImage[lineIndex1,
		    :]; i5 = thisCorrectedImage[lineIndex5, :];
		i1new = i1/factorIdoublePixel; i5new = i5 / \
		    factorIdoublePixel; i3 = (i1new+i5new)/2.0;
		thisCorrectedImage[lineIndex1,
		    :] = i1new; thisCorrectedImage[lineIndex2, :] = i1new;
		thisCorrectedImage[lineIndex3, :] = i3;
		thisCorrectedImage[lineIndex5,
		    :] = i5new; thisCorrectedImage[lineIndex4, :] = i5new

		IntensityArray = thisCorrectedImage.T.reshape(
		    image_corr1_sizeX*image_corr1_sizeY)
		# this is the corrected intensity of each pixel, on the image having the new size

		# saving the Intensity file (in correspondence with XY file)
		xyzLine = ""
		for x in range(0, image_corr1_sizeX):
			for y in range(0, image_corr1_sizeY):
				xyzLine += ""+str(twoThArray[y, x])+" "+str(psiArray[y, x]
				                  )+" "+str(thisCorrectedImage[y, x])+"\n"
		# saving the file
		print("   ... writing intensity file")
		XYZlogFileName = "raw_%d.txt" % (pointIndex)
		with open(fileSavePath+XYZlogFileName, "a") as saveFile:
			saveFile.write(xyzLine)

	print("FINISHED %s\n" % (scanNo))


def visu_one_corrImage(scanNo, imgIndex, compressionFactor, fileNameRoot, inputDataFolder, logScale, mini, maxi):
	# visualise the 2th psi  corrected XPAD data (image) of the n-th point in the scan (n = imgIndex)
	# NB: 1st image has index 0 !!!
	# use negative values for mini and maxi to keep autoscale
	# make sure you have the proper inputDataFolder and that you already run before the geometry correction (previous step)
	pathData = inputDataFolder + "scan_%d/" % (scanNo)
	fileName = fileNameRoot + str(imgIndex) + ".txt"

	data = numpy.genfromtxt(pathData+fileName, skip_header=0)

	print("looking at file=%s" % (fileName))

	if logScale:
		data[:, 2] = numpy.log10(data[:, 2])
		data[numpy.isinf(data)] = 0

	if mini < 0:
		mini = numpy.min(data[:, 2]);
	if maxi < 0:
		maxi = numpy.max(data[:, 2]);
	x = data[::compressionFactor, 0]; y = data[::compressionFactor,
	    1]; z = data[::compressionFactor, 2];
	plt.figure()
	plt.tripcolor(x, y, z, vmin=mini, vmax=maxi, shading='flat')
	plt.colorbar()
	plt.grid()
	plt.xlabel("twoTh (deg)"); plt.ylabel("psi (pdeg)");
	plt.title("scanNo=%d pointIndex = %d" % (scanNo, imgIndex))


def extract_intensity_vs_2th(scanNo, index1, index2, psiAve, deltaPsi, fileNameRoot, inputDataFolder, outputPath):
	calib = 91.8;  # pixels in 1 deg.
	print("scanNo = %s" % (scanNo))

	psi1 = psiAve-deltaPsi; psi2 = psiAve+deltaPsi;
	try:
		for imgIndex in range(index1, index2+1):
			# reading data XY
			dataXY = numpy.genfromtxt(inputDataFolder+"scan_"+str(scanNo) +
    		                          "/"+fileNameRoot+str(imgIndex)+".txt", skip_header=0)
			twoThArray = dataXY[:, 0]; psiArray = dataXY[:, 1];
      
			maskPsi = numpy.ones(psiArray.shape)
			maskPsi[psiArray < psi1] = 0.0; maskPsi[psiArray > psi2] = 0.0;
      
			print("   ... looking at imgIndex = %d out of max %d " % (imgIndex, index2))
			print("   ... XPAD image angular range:");
			miniTwoTh = round(twoThArray.min())-1; maxiTwoTh = round(twoThArray.max())+1;
			stepTwoTh = 1.0/calib*3/3
			nbOfBins = int((0.0+maxiTwoTh-miniTwoTh)/stepTwoTh)+1;
      		# generate the tables for radial integration, this is delta
			TwoThResult = numpy.zeros(nbOfBins+1);
			for ii in range(0, nbOfBins):
				TwoTh_temp1 = miniTwoTh + ii*stepTwoTh; TwoTh_temp2 = TwoTh_temp1 + stepTwoTh;
				TwoThResult[ii] = 0.5*(TwoTh_temp1+TwoTh_temp2);
			thisBinArray = numpy.floor((twoThArray*maskPsi - miniTwoTh)/stepTwoTh);
			thisBinArray = thisBinArray.astype('int')
			print("          (2th_mini=%.2f   2th_maxi=%.2f)" %
      		      (twoThArray.min(), twoThArray.max()))
			print("          (psi_mini=%.2f   psi_maxi=%.2f)" %
      		      (psiArray.min(), psiArray.max()))
      
      		# this will be the summed intensity
			intensityResult = numpy.zeros(nbOfBins+1);
			IntensityArray = dataXY[:, 2]*maskPsi
      
			indexes_ = numpy.nonzero(IntensityArray > 0)[0]
			my_bin = thisBinArray[indexes_]
      
			my_intensity = IntensityArray[indexes_]
			aggregated = numpy.zeros(nbOfBins+1)
			for i in range(my_bin.max()+1):
				selected_intensities = my_intensity[my_bin == i]
				intensityResult[i] = selected_intensities.mean()
      
			intensityResult[numpy.isnan(intensityResult)] = -1
      		# END calculating binned data
      
      		# writting the 2th result / regrouped file
			resultLine = "#TwoTh Intensity\n"
			for aa in range(len(intensityResult)):
				if (intensityResult[aa] > 0):
					resultLine += "%s %s\n" % (TwoThResult[aa], intensityResult[aa])
      		# saving the file
			XYZlogFileName = "I_vs_2th_%d.txt" % (imgIndex)
			with open(outputPath+"scan_"+str(scanNo)+"/"+XYZlogFileName, "a") as saveFile:
				saveFile.write(resultLine)
      		# print('D:\\choquet\\exploited\\scan_'+str(scanNo)+'\\I_vs_2th_0.txt')
      		# fichier=np.loadtxt('D:\\choquet\\exploited\\scan_'+str(scanNo)+'\\I_vs_2th_0.txt')
      		# plt.plot(fichier[:,0], fichier[:,1], label='scan_%s_%s'%(scanNo, calib))
      		# plt.xlim(30,32)
      		# plt.ylim(200,400)
      		# plt.legend()
	except:
		print('you')
	print("FINISHED %s" % (scanNo))


# ==========================
def funct_pearson7(x, backgr, slopeLin, amplitude, center, fwhmLike, exposant):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7
	return backgr+slopeLin*x+amplitude*(1+((x-center)/fwhmLike)**2.0)**(-exposant)


def funct_pearson7_2(x, backgr, slopeLin, amplitude, center, fwhmLike, exposant, amplitude2, center2, fwhmLike2, exposant2):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7 Hall et al. Journal of applied crystallography 1977 :http://journals.iucr.org/j/issues/1977/01/00/a15317/a15317.pdf
	return backgr+slopeLin*x+amplitude*(1+((x-center)/fwhmLike)**2.0)**(-exposant) + amplitude2*(1+((x-center2)/fwhmLike2)**2.0)**(-exposant2)


def funct_pearson7_3(x, backgr, slopeLin, amplitude, center, fwhmLike, exposant, amplitude2, center2, fwhmLike2, exposant2, amplitude3, center3, fwhmLike3, exposant3):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7 Hall et al. Journal of applied crystallography 1977 :http://journals.iucr.org/j/issues/1977/01/00/a15317/a15317.pdf
	return backgr+slopeLin*x+amplitude*(1+((x-center)/fwhmLike)**2.0)**(-exposant) + amplitude2*(1+((x-center2)/fwhmLike2)**2.0)**(-exposant2)++amplitude3*(1+((x-center3)/fwhmLike3)**2.0)**(-exposant3)


def funct_pearson7_fc2(x, backgr, slopeLin, a, amplitude, center, fwhmLike, exposant):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7
	return backgr+slopeLin*x+a*x**2+amplitude*(1+((x-center)/fwhmLike)**2.0)**(-exposant)


def funct_deux_pearson7_fc2(x, backgr, slopeLin, a, amplitude1, center1, fwhmLike1, exposant1, amplitude2, center2, fwhmLike2, exposant2):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7
	return backgr+slopeLin*x+a*x**2+amplitude1*(1+((x-center1)/fwhmLike1)**2.0)**(-exposant1)+amplitude2*(1+((x-center2)/fwhmLike2)**2.0)**(-exposant2)


def funct_deux_pearson7_fdfix(x, backgr, slopeLin, amplitude1, center1, fwhmLike1, exposant1, center2):
	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	PI = numpy.pi
	# p7 Ag et TiO2
	return backgr+slopeLin*x+amplitude1*(1+((x-center1)/fwhmLike1)**2.0)**(-exposant1)+157.026*(1+((x-center2)/0.04363)**2.0)**(-2.136)
	# return backgr+slopeLin*x+amplitude1*(1+((x-center1)/fwhmLike1)**2.0)**(-exposant1)+88.6*(1+((x-16.92)/0.367)**2.0)**(-2.585) #p7


def fittingSeq_p7(scanNo, pointIndex, peakCenGuess, range2th, guess, flagHeader, plotHeader, fileNameRoot, inputRootFolder):

	# backgr, slopeLin, amplitude, center, fwhmLike, exposant
	twoTh1 = peakCenGuess - range2th; twoTh2 = peakCenGuess + range2th;
	pathData = inputRootFolder+"DataNxs/"
	pathDataTXT = inputRootFolder+"exploited/scan_%s/" % (scanNo)
	pathSave = inputRootFolder+"fit/"

	try:
		os.stat(pathSave)
	except:
		os.mkdir(pathSave)

	fileSaveName = "fit_%s" % (peakCenGuess)
	line2write = ""
	if (flagHeader):
		# writing the header in the output file
		# amplitude2 center2 fwhmLike2 exposant2\n"
		line2write = "#temps temps_reel scanNo pointIndex omega chi phi f13 f24 monitor backgr slopeLin amplitude center fwhmLike exposant"

	logFileName = fileSaveName+".txt"
	logFile = open(pathSave+logFileName, "a")
	logFile.write(line2write)

	# open the NXS file for reading chi
	fileName = pathData + fileNameRoot + str(scanNo)+"_0001.nxs"
	try:
		with h5py.File(fileName,'r') as f: #will properly close it once the indented code is executed
			group1 =  list(f.keys())[0]
			gam = f[group1]['DIFFABS/d13-1-cx1__ex__dif.1-gamma/raw_value'][()]
			# xpadImage = f[f'scan_{scanNo:0>4d}/scan_data/data_{GLOBAL_xpad_dataset}'][()]
			deltaArray = f[f'scan_{scanNo:0>4d}/scan_data/data_08'][()]
			print('delta_array=', deltaArray)
			#xpadImage = f[group1]['scan_data/data_28'][pointIndex]
			monitor = f[group1]['scan_data/data_02'][pointIndex]
			delta = f[group1]['scan_data/data_11'][pointIndex]
			chi=f[group1]['scan_data/data_09'][pointIndex]
			phi=f[group1]['scan_data/data_10'][pointIndex]
			omega=f[group1]['scan_data/data_08'][pointIndex]
			f13=f[group1]['scan_data/data_22'][pointIndex]
			f24=f[group1]['scan_data/data_25'][pointIndex]
			temps=f[group1]['scan_data/sensors_rel_timestamps'][pointIndex]
			temps_reel=f[group1]['scan_data/sensors_timestamps'][pointIndex]#epoch
	
		print('omega=%s'%(omega))    

		data = numpy.genfromtxt(pathDataTXT + "I_vs_2th_%s.txt" %(pointIndex), skip_header = 1)
		thisChi = chi
		print("looking at file = I_vs_Two_%s.txt   (chi=%s)" %(pointIndex, thisChi))
    
		indexMini = 0; indexMaxi = 0;
		for i in range (data.shape[0]):
			if (data[i, 0] <= twoTh1):
				indexMini = i
			if (data[i, 0] <= twoTh2):
				indexMaxi = i
		
		# pour la région	
		xcoord = data[indexMini:indexMaxi, 0]
		ycoord = data[indexMini:indexMaxi, 1]
	
		'''
		# pour tout	
		xcoord = data[:, 0]
		# ycoord = data[:, 1]/monitor
		ycoord = data[:,1]/4#-I_2th_reference[:,1]
		'''

		'''
		# Nb(110° + TiO2

	
		guess=[250, 0, 350, 31.77, 0.1, 1, 100, 32.4, 0.04, 1]# Nb110_guess initial
		guess=[307, -1.5, 390, 31.77, 0.25, 1.2, 120, 32.4, 0.05, 2.7]# Nb110_guess initial    



		
		# DELETE A REGION 2 petits pics TiO2 dans les pieds pour chi=90 de (110)beta
		print(numpy.shape(xcoord))
		xcoord=np.hstack((xcoord[0:110],xcoord[124:-1]))
		ycoord=np.hstack((ycoord[0:110],ycoord[124:-1]))


		#FIN Nb 110 + TiO2
		''' 
		''' 
        #Mo(110)+ TiO2
        
		guess=[250, 0, 400, 33.2, 0.1, 1, 400, 34.1, 0.04, 1]# Mo110_guess initial
		guess=[370, -3.5, 332, 33.32, 0.24, 1.7, 460, 34.07, 0.04, 2.17]# Mo110_guess initial    
        
        
        #FIN Mo(110)+ TiO2
		'''     
        #Al(111)
        
		#guess=[500 , 0.05 , 20000, 35.848, 0.12, 2, 180, 36.438, 0.09, 6, 300, 36.590, 0.4, 3.5]# Mo110_guess initial
		#guess=[7.84133386e+02, -1.27645159e+01,  1.14491542e+02,  3.19922498e+01,
  #1.44061482e-01,  1.67108796e+00]

		#guess=[575 , 0 , 1200, 35.667, 0.2, 2, 150, 36.349, 0.1, 2, 25, 36.697, 0.2, 1]
        
        #TiO2(101)
		#guess=[8.38174024e+02, -1.27885688e+01, 4.18414422e+01,  3.25355502e+01,
 #-8.52830732e+03,  9.51391685e+09]# TiO2(101)_guess initial   
        
        #Cu(111)
        
		guess=[3.38895430e+02, 1.13361533e+00, 1.65020456e+02, 3.59395757e+01,
 1.75329545e-01, 1.03643986e+00]
  
        #TiO2(111)
		#guess=[4.58648611e+02, -5.09527418e+00,  4.33610407e+02,  3.65625524e+01,
 #-1.17647033e+04,  8.58649620e+10]# TiO2(101)_guess initial 
        
        #--- CHOOSING WHAT TO TAKE ---#
		
		print(numpy.shape(xcoord))
        
        # delete TiO2 region
		xcoord=np.hstack((xcoord[0:130],xcoord[160:-1])) #coord. in plot pixels
		ycoord=np.hstack((ycoord[0:130],ycoord[160:-1]))
        
        # delete Al / Cu region
		#xcoord=np.hstack((xcoord[0:25],xcoord[130:-1])) #coord. in plot pixels
		#ycoord=np.hstack((ycoord[0:25],ycoord[130:-1]))
		
        # delete Al region and extra TiO2 peak
		#xcoord=np.hstack((xcoord[0:60],xcoord[135:160],xcoord[180:-1])) #coord. in plot pixels
		#ycoord=np.hstack((ycoord[0:60],ycoord[135:160],ycoord[180:-1])) #take points from three different regions: 0:50, 130:155, and 180:-1
        
        
		# comment this out if you don't want the guess plot
        ###TRACE (= plot) avec les paramètres initiaux "Guess plot"
		plt.figure(1)
		# plt.plot(xcoord, ycoord, '-')#;plt.yscale('symlog')#, label='index_%s'%(this_ScanNo))
		plt.plot(xcoord, ycoord, '-o', label='index_%s'%(pointIndex))
	
		#plt.plot(xcoord, funct_pearson7(xcoord, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5]), 'r-')
	
		# guess[1]=((ycoord[-1]-ycoord[1])/(xcoord[-1]-xcoord[1]))
		plt.legend()
		print(guess)
		# plt.plot(xcoord, funct_pearson7_3(xcoord, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6], guess[7], guess[8], guess[9], guess[10], guess[11], guess[12], guess[13], guess[14]), 'r-')
		plt.plot(xcoord, funct_pearson7(xcoord, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5]), 'r-')
		# plt.plot(xcoord, funct_deux_pearson7_fdfix(xcoord, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5]), 'r-o')	
		plt.title('index_%s'%(pointIndex))	
		plt.show()
		
	
	
	
	
		# print('guess1=%s'%(guess))
		# DEBUT de l'ajustement 
		try:
			# plt.figure()
			# plt.plot(xcoord, ycoord, 'ro-',label='scanNo_%s_f13_%s'%(scanNo, np.round(f13,1)))
			# plt.plot(xcoord,funct_pearson7_fc2(xcoord,  guess[0], guess[1], guess[2], guess[3], guess[4], guess[5],guess[6]))
		

			#popt, pcov = curve_fit(funct_pearson7, xcoord, ycoord, guess) #fitting
			popt, pcov = curve_fit(funct_pearson7, xcoord, ycoord, guess) #fitting
	
			# plt.plot(xcoord, funct_pearson7_2(xcoord, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6], guess[7], guess[8], guess[9], guess[10]))
			# guess=popt
			# aju_debut=1
	
			print("        opti: %s" %(str(popt)))
			# print "        opti: %s" %(str(pcov))
			print(omega)#, monitor#, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
			# 1 pearson
			line2write = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n" %(temps,temps_reel, scanNo, pointIndex, omega, chi, phi, f13, f24, monitor, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
				
			# 2 pearsons		
			# line2write = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" %(temps,temps_reel, scanNo, pointIndex, chi, phi, monitor, f13, f24, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10] )	#Deux PEARSONs
			# logFile.write(line2write)
	
			# 2 pearsons + fond poly ordre 2
			# line2write = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" %(temps, temps_reel, scanNo, pointIndex, thisChi, omega, phi, monitor, f13, f24, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])	
			logFile.write(line2write)
		
		
		
			# 3 pearsons
			# line2write = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" %(temps, temps_reel, scanNo, pointIndex, thisChi, omega, phi, monitor, f13, f24, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13], popt[14])	
			# logFile.write(line2write)


			plotHeader= True ## put this to False if you don't want the fitted plot (otherwise True)
			if (omega >10 and omega <19): #or change the omega
				plt.figure()
				plt.plot(xcoord, ycoord, 'ro-',label='scanNo_%s_delta_%s_chi_%s'%(scanNo, np.round(delta,2),np.round(chi)))
				# plt.plot(xcoord, funct_pearson7(xcoord, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), 'k-', label='initiale')
				# plt.plot(xcoord, popt[0]+popt[1]*xcoord+popt[2]*xcoord**2+popt[3]*(1+((xcoord-popt[4])/popt[5]**2)**(-popt[6])))
				# plt.plot(xcoord, funct_pearson7_2(xcoord, guess0[0], guess0[1], guess0[2], guess0[3], guess0[4], guess0[5], guess0[6], guess0[7], guess0[8], guess0[9], guess0[10]))
				
				'''plt.plot(scanNo, f13, 'ko', label='F13')
				plt.plot(scanNo, f24, 'kx', label='F24')
				plt.xlim(15450,15950)
				plt.ylim(0,145)
				plt.xlabel('Force(N)')
				plt.ylabel('scanNo')
				plt.title('Trilayer_Nb110') 
				plt.grid()

				if scanNo==15482:
					plt.legend()#['F13', 'F24'],loc='higher left')
				'''
			
				fit = funct_pearson7(xcoord, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])#, popt[7], popt[8], popt[9], popt[10])#fond continu ordre 2
				# fitinitial=	funct_pearson7_2(xcoord, guess0[0], guess0[1], guess0[2], guess0[3], guess0[4], guess0[5], guess0[6], guess0[7], guess0[8], guess0[9], guess0[10])#, popt[10])#fond continu ordre 2		
				# OPEN reference
				# I_2th_reference=np.loadtxt("D:\\DT\\Renault\\2110\\exploited\\scan_7922\\I_vs_2th_0.txt")
				# plt.plot(I_2th_reference[:,0],I_2th_reference[:,1],'-s', label='ref_#7922')     
				plt.plot(xcoord, fit, 'b-',label='fit')
				# plt.plot(xcoord, fitinitial, 'k-',label='fit0')
				#plt.ylim((350,3500))
				# plt.title("Cu191106b_%s"%(pointIndex))
				plt.xlabel('2theta(deg)')
				plt.ylabel('Intensite(u._arb.)')
				#plt.legend(['omega_%s'%(omega), 'ajust. PVII simpl.'],loc='higher left')
				plt.legend()			
				# plt.xlim(16.2,17.8)
				#plt.ylim(200,600)
				plt.title('Al(111)_%s'%(scanNo))
				plt.grid()
				plt.show()
				plt.savefig('FIT_%s.png'%(scanNo))
				#plt.close()
		
		except:
			print("    *** no fit found %s" %(scanNo))
	
		logFile.close()
		# FIN de l'ajustement
		# print('guess2=%s'%(guess))
		# return guess, aju_debut
 	
		print("FINISHED")	
	except:
		print("pas de fichier")
# ==========================


'''
# VISUALISATION DES IMAGES SCANNEES
# visualising RAW images in the scan
pathRoot = 'G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/' #folder (root like) in which all the data is recovered and saved
this_ScanNo =5084; this_imgIndex = 0
this_fileNameRoot = "scan_"
# for this_imgIndex in range(1,1200+1,1):
this_inputDataFolder = pathRoot + "DataNxs/" #this is the root name of the NXS file to be looked at (it includes the root part of the fileName)
# this_mini = -1; this_maxi = -1; this_logScale = True #auto scale in mini and maxi + LOG scale
# visu_one_rawImage(scanNo=this_ScanNo, imgIndex=this_imgIndex, fileNameRoot=this_fileNameRoot, inputDataFolder = this_inputDataFolder, logScale = this_logScale, mini=this_mini , maxi=this_maxi)
this_mini = 0.1; this_maxi = 3; this_logScale = True # LOG scale + manual mini/maxi
visu_one_rawImage(scanNo=this_ScanNo, imgIndex=this_imgIndex, fileNameRoot=this_fileNameRoot, inputDataFolder = this_inputDataFolder, logScale = this_logScale, mini=this_mini , maxi=this_maxi)
# ================================================
'''

'''
# CREATION ET VISUALISATION DU FLATSCAN
initial_scan = 6#10 ici le premier scan du flat
final_scan = 6#14 ici le dernier scan du flat
pathRoot = "G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/"
this_inputDataFolder = pathRoot + "flat/"
destinationFolder = pathRoot + "exploited/"
file_extension ="_0001.nxs"# ".nxs"#"_0001.nxs"
dataset = "31"#03 pour les scas 10 à 14

flat_img = flatscan(first_scan=initial_scan, last_scan=final_scan, path=this_inputDataFolder, file_extension=file_extension, dataset=dataset, plot_flag=False, save_flag=True, destination_save=destinationFolder)


# CREATION DES IMAGES DEPLIEES

#import glob

#liste_fichier = glob.glob('D:/Users/saltaf/Desktop/Synchrotron_SOLEIL/2022_07/python/DataNxs/*nxs')




for this_ScanNo in range (5311,5534+1,1):
#for this_ScanNo in liste_fichier:
	# making the geometrical correction (2th / psi / intensity) for all images in a scan
	pathRoot = "G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/" #folder (root like) in which all the data is recovered and saved
	# pathRoot = "/Volumes/CHOMMAUX/Soleil2007/Depouillement/"
	#this_ScanNo=this_ScanNo[71:75]
	this_ScanNo=int(this_ScanNo)
	if this_ScanNo>999 :
		this_fileNameRoot = "scan_"
	elif this_ScanNo>99 :
		this_fileNameRoot = "scan_0"
	elif this_ScanNo>9:
		this_fileNameRoot = "scan_00"     
	else:
           this_fileNameRoot = "scan_000"
	
	# this_fileNameRoot = "scan_"
	this_inputDataFolder = pathRoot + "DataNxs/" #this is the root name of the NXS file to be looked at (it includes the root part of the fileName)
	this_outputPath = pathRoot + "exploited/" #this is the Root folder name for saving . SUb folders with the scan noi. will be created
	try:
		extract_and_correct_1scan(scanNo= this_ScanNo,flatImg=flat_img, fileNameRoot  = this_fileNameRoot, inputDataFolder = this_inputDataFolder, outputPath=this_outputPath)
	except:
		print('Scan_No=%s n existe pas'%(this_ScanNo))
# ==============================================
'''


'''
# VISUALISATION DES IMAGES CORRIGEES
# visualising corrected images (after geometry corr)
# make sure the corrected images are generated, i.e. have already run the previous step
pathRoot = "G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/" #folder (root like) in which all the data is recovered and saved
this_ScanNo = 5534; this_imgIndex = 1
this_fileNameRoot = "raw_"
this_inputDataFolder = pathRoot + "exploited/" #this is the root name of the NXS file to be looked at (it includes the root part of the fileName)
this_compressionFactor = 1 #only one point over 10 will be considered. Note that the full data set is more than 140000points, so using a factor 5-10 is a good ideea for speeding display
this_mini = -1; this_maxi = -1; this_logScale = True #auto scale in mini and maxi + LOG scale
visu_one_corrImage(scanNo=this_ScanNo, imgIndex=this_imgIndex, compressionFactor =this_compressionFactor, fileNameRoot=this_fileNameRoot, inputDataFolder = this_inputDataFolder, logScale = this_logScale, mini=this_mini , maxi=this_maxi)
# this_mini = 1.5; this_maxi = 3; # LOG scale + manual mini/maxi
# visu_one_corrImage(scanNo=this_ScanNo, imgIndex=this_imgIndex, compressionFactor =this_compressionFactor, fileNameRoot=this_fileNameRoot, inputDataFolder = this_inputDataFolder, logScale = this_logScale, mini=this_mini , maxi=this_maxi)
# ==============================================
'''

'''
# EXTRACTION DES COURBES I(2O)
for this_ScanNo in range(5083, 5303+1,1):
# for this_ScanNo in range(1457, 1457+1,1):
	# extracting the intensity vs. 2th curves
	pathRoot = "G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/"
	this_index1 = 0; this_index2 = 1000
	this_psiAve = 0.0; this_deltaPsi = 4
	this_fileNameRoot = "raw_"
	this_inputDataFolder = pathRoot + "exploited/" 
	this_outputPath = this_inputDataFolder
	extract_intensity_vs_2th(scanNo = this_ScanNo, index1 = this_index1, index2 = this_index2, psiAve = this_psiAve, deltaPsi = this_deltaPsi, fileNameRoot = this_fileNameRoot, inputDataFolder = this_inputDataFolder, outputPath = this_outputPath)
	## extracting the intensity vs. 2th curves
	## pathRoot = "/Volumes/CHOMMAUX/Soleil2007/Depouillement/" #folder (root like) in which all the data is recovered and saved
	##TpathRoot = "C:/Users/Phymat/Documents/renault/soleil2207/"
	##this_index1 = 0; this_index2 = 4
	##this_psiAve = 0; this_deltaPsi = 50
	##this_fileNameRoot = "raw_"
	##this_inputDataFolder = pathRoot + "exploited/" 
	##this_outputPath = this_inputDataFolder
	##extract_intensity_vs_2th(scanNo = this_ScanNo, index1 = this_index1, index2 = this_index2, psiAve = this_psiAve, deltaPsi = this_deltaPsi, fileNameRoot = this_fileNameRoot, inputDataFolder = this_inputDataFolder, outputPath = this_outputPath)

# ================================================
'''



# FIT
###plt.figure()
for this_ScanNo in range (5311, 5534+1,3):#74, 851+1,1

	#for Nb and Mo index 0 but for Cu index 1
	for this_index1 in range(0,0+1,1):#scan_370,476+1,50
		# extracting the intensity vs. 2th curves
		pathRoot = "G:/Mechanik/PERSONAL/bylj/04_SOLEIL_2210/Synchrotron_SOLEIL/" #folder (root like) in which all the datared and saved

		#Nb
		#this_cenGuess = 36.45; this_widthGuess = 0.06; th
		#Al
		#this_cenGuess = 32; this_widthGuess = 0.06; this_range2th = 1       #Mo(110) + TiO2
		#TiO2
		# this_cenGuess = 30; this_widthGuess = 0.06; this_range2th = 0.5
		this_cenGuess = 36; this_widthGuess = 0.06; this_range2th = 1 


		this_guess = [-10, 0.0, -10, this_cenGuess, this_widthGuess, 1.0] #backgr [0] and amplit [2] will be auto detected

		if this_ScanNo>999 :
			this_fileNameRoot = "scan_"
		elif this_ScanNo>99 :
			this_fileNameRoot = "scan_0"
		elif this_ScanNo>9 :
			this_fileNameRoot = "scan_00"
		else:
			this_fileNameRoot = "scan_000"
		this_inputRootFolder = pathRoot
		this_flagHeader = False
		this_plotHeader = False	
		if this_index1 == 0:
			this_flagHeader = False
			this_plotHeader = True
		fittingSeq_p7(scanNo=this_ScanNo, pointIndex=this_index1, peakCenGuess=this_cenGuess, range2th=this_range2th, guess = this_guess, flagHeader = this_flagHeader, plotHeader = this_plotHeader, fileNameRoot = this_fileNameRoot, inputRootFolder = this_inputRootFolder)

#================================================##



