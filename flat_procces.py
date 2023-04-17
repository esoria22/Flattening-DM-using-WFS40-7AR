# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:30:56 2022
ds
@author: esoria
"""
import sys
import os
import time
import struct
import numpy as np
import time

import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np

a= 1
if a==1:    #activar el WFS

#cambio parámetros de medida
    #load the DLL:
    wfs = ct.windll.WFS_32
    byref = ct.byref
    #Set the data types compatible with C DLL
    count = ct.c_int32() 
    deviceID  = ct.c_int32()  
    instrumentListIndex  = ct.c_int32() 
    inUse = ct.c_int32() 
    instrumentName = ct.create_string_buffer(256)
    instrumentSN = ct.create_string_buffer(256)
    resourceName = ct.create_string_buffer(256)
    IDQuery = ct.c_bool()
    resetDevice = ct.c_bool()
    instrumentHandle = ct.c_ulong()
    pupilCenterXMm = ct.c_double()
    pupilCenterYMm = ct.c_double()
    pupilDiameterXMm = ct.c_double()
    pupilDiameterYMm = ct.c_double()
    exposureTimeAct = ct.c_double()
    exposureTimeSet = ct.c_double()
    masterGainAct = ct.c_double()
    masterGainSet = ct.c_double()
    dynamicNoiseCut = ct.c_int32() 
    calculateDiameters = ct.c_int32() 
    blackLevelOffsetSet = ct.c_int32() 
    cancelWavefrontTilt = ct.c_int32() 
    errorMessage = ct.create_string_buffer(512)
    errorCode = ct.c_int32()
    pixelFormat = ct.c_int32()
    pixelFormat.value = 0 #currently 8 bit only
    camResolIndex = ct.c_int32()
    spotsX = ct.c_int32()
    spotsY = ct.c_int32()
    wavefrontType = ct.c_int32() 
    limitToPupil = ct.c_int32()
    zernikeOrders=  ct.c_int32() 
    roCMm= ct.c_double()
    
    #Set the parameter values
    MAX_SPOTS_X = 50
    MAX_SPOTS_Y = 40
    MAX_ZERNIKE_MODES=66
    MAX_ZERNIKE_ORDERS = 10
    arrayZernikeOrdersUm=np.zeros(([MAX_ZERNIKE_ORDERS+1]),dtype = np.float32)
    arrayZernikeUm = np.zeros(([MAX_ZERNIKE_MODES+1]),dtype = np.float32)
    arrayWavefront = np.zeros((MAX_SPOTS_Y,MAX_SPOTS_X),dtype = np.float32)
    instrumentListIndex.value = 0 #0,1,2,, if multiple instruments connected
    arrayZernikeUm = np.zeros(([MAX_ZERNIKE_MODES+1]),dtype = np.float32)
    #Configure camera
    camResolIndex.value = 1
    zernikeOrders.value= 10
   
    # For WFS instruments: 
    # Index  Resolution 
    # 0    1280x1024          
    # 1    1024x1024          
    # 2     768x768            
    # 3     512x512            
    # 4     320x320 
    # For WFS10 instruments: 
    # Index  Resolution 
    # 0     640x480          
    # 1     480x480          
    # 2     360x360            
    # 3     260x260            
    # 4     180x180 
    # For WFS20 instruments: 
    # Index  Resolution 
    # 0    1440x1080             
    # 1    1080x1080             
    # 2     768x768               
    # 3     512x512               
    # 4     360x360               
    # 5     720x540, bin2 
    # 6     540x540, bin2 
    # 7     384x384, bin2 
    # 8     256x256, bin2 
    # 9     180x180, bin2
    
    #Set pupil
    pupilCenterXMm.value = 0 #mm
    pupilCenterYMm.value = 0.3 #mm
    pupilDiameterXMm.value = 4.2 #mm
    pupilDiameterYMm.value = 4.2 #mm
    
    #Set spot calculation params
    dynamicNoiseCut.value = 0
    calculateDiameters.value = 0
    cancelWavefrontTilt.value = 0
    
    
    wavefrontType.value = 0
    zernikeOrders.value = 10
    arrayZernikeOrdersUm=np.zeros(([MAX_ZERNIKE_ORDERS+1]),dtype = np.float32)
    # This parameter defines the type of wavefront to calculate. 
    # Valid settings for wavefrontType: 
    # 0   Measured Wavefront 
    # 1   Reconstructed Wavefront based on Zernike coefficients 
    # 2   Difference between measured and reconstructed Wavefront 
    # Note: Function WFS_CalcReconstrDeviations needs to be called prior to this function in case of Wavefront type 1 and 2.
    
    
    limitToPupil.value = 1
    # This parameter defines if the Wavefront should be calculated based on all detected spots or only within the defined pupil. 
    # Valid settings: 
    # 0   Calculate Wavefront for all spots 
    # 1   Limit Wavefront to pupil interior
    #Check how many WFS devices are connected
    wfs.WFS_GetInstrumentListLen(None,byref(count))
    print('WFS sensors connected: ' + str(count.value))
    
    #Select a device and get its info
    devStatus = wfs.WFS_GetInstrumentListInfo(None,instrumentListIndex, byref(deviceID), byref(inUse),
                                 instrumentName, instrumentSN, resourceName)
    if(devStatus != 0):
        errorCode.value = devStatus
        wfs.WFS_error_message(instrumentHandle,errorCode,errorMessage)
        print('error in WFS_GetInstrumentListInfo():' + str(errorMessage.value))
    else:
        print('WFS deviceID: ' + str(deviceID.value))
        print('in use? ' + str(inUse.value))
        print('instrumentName: ' + str(instrumentName.value))
        print('instrumentSN: ' + str(instrumentSN.value))
        print('resourceName: ' + str(resourceName.value))
    
    #devStatus = wfs.WFS_close (instrumentHandle)
    
    if not inUse.value:
        devStatus = wfs.WFS_init(resourceName, IDQuery, resetDevice, byref(instrumentHandle))
        if(devStatus != 0):
            errorCode.value = devStatus
            wfs.WFS_error_message(instrumentHandle,errorCode,errorMessage)
            print('error in WFS_init():' + str(errorMessage.value))
        else:
            print('WFS has been initialized. Instrument handle: ' +str(instrumentHandle.value))
    else:
        print('WFS already in use')
    
    
    
    #Configure camera
    devStatus = wfs.WFS_ConfigureCam(instrumentHandle, 
                                     pixelFormat, camResolIndex, byref(spotsX), byref(spotsY))
    if(devStatus != 0):
        errorCode.value = devStatus
        wfs.WFS_error_message(instrumentHandle,errorCode,errorMessage)
        print('error in WFS_ConfigureCam():' + str(errorMessage.value))
    else:
        print('WFS camera configured')
        print('SpotsX: ' + str(spotsX.value))
        print('SpotsY: ' + str(spotsY.value))
    
    #Set pupil
    devStatus = wfs.WFS_SetPupil(instrumentHandle,
                                 pupilCenterXMm, pupilCenterYMm, pupilDiameterXMm, pupilDiameterYMm)
    if(devStatus != 0):
        errorCode.value = devStatus
        wfs.WFS_error_message(instrumentHandle,errorCode,errorMessage)
        print('error in WFS_SetPupil():' + str(errorMessage.value))
    else:
        print('WFS pupil set')
exposureTimeSet.value=4.5 #ms
desvStatus=wfs.WFS_SetExposureTime(instrumentHandle,exposureTimeSet,byref(exposureTimeAct))
print("exp_act:"+str(exposureTimeAct.value))

masterGainSet.value=1.03
desvStatus=wfs.WFS_SetMasterGain(instrumentHandle,masterGainSet,byref(masterGainAct))
print("gain_act:"+str(masterGainAct.value))

blackLevelOffsetSet.value= 10 #entre 0 y 255
desvStatus=wfs.WFS_SetBlackLevelOffset (instrumentHandle, blackLevelOffsetSet)


num_modes= 66

sys.path.append('C:/Users/esoria.DOMAINT/Documents/testTWFs/Lib')
#me comunico con el DM

from asdk38 import DM

serialName = 'BOL105'
    
print("Connect the mirror")
dm = DM( serialName )
    
print("Retrieve number of actuators")
num_act = int( dm.Get('NBOfActuator') )
print( "Number of actuator for " + serialName + ": " + str(num_act) )

dm.Reset()
values = [0.] * num_act

#interation matrix

probe_amp = 0.4

influ_M = np.zeros((num_modes+1,num_act))
for ind in np.arange(0,num_act):
           z = 0  
       # Probe the phase response
           for s in [1, -1]:
               amp = np.zeros((num_act,))
               amp[ind] = s * probe_amp
               dm.Send(amp)              
               devStatus = wfs.WFS_TakeSpotfieldImage(instrumentHandle)
               devStatus = wfs.WFS_CalcSpotsCentrDiaIntens(instrumentHandle, dynamicNoiseCut, calculateDiameters)
               devStatus = wfs.WFS_CalcSpotToReferenceDeviations(instrumentHandle, cancelWavefrontTilt)
               devStatus= wfs.WFS_ZernikeLsf(instrumentHandle, byref(zernikeOrders), arrayZernikeUm.ctypes.data, arrayZernikeOrdersUm.ctypes.data, byref(roCMm))
               z1= arrayZernikeUm
               z += s * z1 / (2 * probe_amp)
               time.sleep(0.5)
               
           print(f"Actuador:{ind}") 
           influ_M[:,ind]=z
#elimino la fila sobrante y el pistón
influ_M2=influ_M[2:,:]
from hcipy import *
j= j[1:,:]
Z2A=inverse_tikhonov(influ_M2, rcond=0.01, svd=None)
Z2A=np.linalg.pinv(j)

#grafico el WF incidente
devStatus = wfs.WFS_CalcWavefront(instrumentHandle, wavefrontType, limitToPupil, arrayWavefront.ctypes.data)
lamb = 0.633 #um, HeNe laser
array= arrayWavefront[:spotsY.value,:spotsX.value].copy()
meanWavefront = np.nanmean(array)
PV = np.nanmax(meanWavefront) - np.nanmin(meanWavefront)
RMS = np.sqrt(np.nanmean(meanWavefront**2,axis=(0,1)))
plt.imshow(array)

#mando con la misma amplitud diferentes modos puros
print('Send Zernike to the mirror: #XX')      
for zern in range(35):
    dm.Send(Z2A[zern]*0.02)
    print(str(zern)+" zern")
    time.sleep(0.5) # Wait for 1 second

print("Reset")
dm.Reset()

print("Exit")
tdevStatus = wfs.WFS_close(instrumentHandle)


import numpy as np
import h5py
f = h5py.File("influ.mat", "r")
data = f.get('data/matrix')
data = np.array(data) # For converting to a NumPy array
import scipy.io
mat = scipy.io.loadmat('influ.mat')
j = np.load("influ.npy")
j= np.resize(j,[66,88])
