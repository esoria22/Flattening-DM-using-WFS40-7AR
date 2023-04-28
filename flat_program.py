# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:30:56 2022

@author: esoria
"""
import sys
import time
import numpy as np
import time
import scipy.io
import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np
import hcipy
import math
from skimage import draw

def flattening (pupil_x, pupil_y, pupil_diameter, exposure_time, gain):

    #Genero las constantes que voy a utilizar en formato c
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
    exposureTimeSet=ct.c_double()
    masterGainAct = ct.c_double()
    masterGainSet=ct.c_double()
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
    #defino el numero de spots maximos, dependerá de mi array
    MAX_SPOTS_X = 50
    MAX_SPOTS_Y = 40
    MAX_ZERNIKE_MODES=66
    MAX_ZERNIKE_ORDERS = 10
    #genero los vectores donde alojaré los datos medidos
    arrayZernikeOrdersUm=np.zeros(([MAX_ZERNIKE_ORDERS+2]),dtype = np.float32)
    arrayZernikeUm = np.zeros(([MAX_ZERNIKE_MODES+2]),dtype = np.float32)
    arrayWavefront = np.zeros((MAX_SPOTS_Y,MAX_SPOTS_X),dtype = np.float32)
    Wavefront = np.zeros((MAX_SPOTS_X,MAX_SPOTS_Y),dtype = np.float32)
    instrumentListIndex.value = 0 #0,1,2,, if multiple instruments connected
    #Configure camera
    camResolIndex.value = 1
    zernikeOrders.value= 10
    
    #Parámetros de la pupila
    pupilCenterXMm.value = pupil_x #mm
    pupilCenterYMm.value = pupil_y #mm
    pupilDiameterXMm.value = pupil_diameter #mm
    pupilDiameterYMm.value = pupil_diameter #mm
    
    
    
    #Set spot calculation params
    dynamicNoiseCut.value = 0
    calculateDiameters.value = 0
    cancelWavefrontTilt.value = 0  
    wavefrontType.value = 0
    zernikeOrders.value = 10
    arrayZernikeOrdersUm=np.zeros(([MAX_ZERNIKE_ORDERS+2]),dtype = np.float32)
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
        sys. exit()
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
            
    
    exposureTimeSet.value= exposure_time
    devStatus = wfs.WFS_SetExposureTime(instrumentHandle,exposureTimeSet,byref(exposureTimeAct))
    print('exposureTimeAct, ms: ' + str(exposureTimeAct.value))
    
    masterGainSet.value= gain
    devStatus = wfs.WFS_SetMasterGain(instrumentHandle,masterGainSet,byref(masterGainAct))
    print('Gain: ' + str(masterGainAct.value))
    
    blackLevelOffsetSet.value=10
    desvStatus=wfs.WFS_SetBlackLevelOffset (instrumentHandle,blackLevelOffsetSet)
    
    
    num_modes= 66
    
    #me comunico con el espejo deformable:
    sys.path.append('C:/Users/esoria.DOMAINT/Documents/testTWFs/Lib')
    
    from asdk38 import DM
    
    serialName = 'BOL105'
        
    print("Connect the mirror")
    dm = DM( serialName )
        
    print("Retrieve number of actuators")
    num_act = int( dm.Get('NBOfActuator') )
    print( "Number of actuator for " + serialName + ": " + str(num_act) )
    
    array = np.zeros((12,12))
    circmask = draw.circle(5.5,5.5,5.5,(12,12))
    #interation matrix
    probe_amp =0.1 #amplitud con la que hago la caracterización
    dm.Reset() #reseteo el DM
     #aplano el DM
    
    influ_M = np.zeros((num_modes-1,num_act)) #genero la matriz donde voy a alojar los valores
    num_med=2
    
    for ind in np.arange(0,num_act):
               z = 0  
           # Probe the phase response
               for s in [1, -1]:
                   r=0
                   for r in np.arange(0,num_med):
                       res=np.zeros((65,num_med))
                       amp = np.zeros((num_act,))
                       amp[ind] = s * probe_amp
                       u=amp 
                       dm.Send(u)
                       
                       devStatus = wfs.WFS_TakeSpotfieldImage(instrumentHandle, exposureTimeAct, masterGainAct)
                       devStatus = wfs.WFS_CalcSpotsCentrDiaIntens(instrumentHandle, dynamicNoiseCut, calculateDiameters)
                       devStatus = wfs.WFS_CalcSpotToReferenceDeviations(instrumentHandle, cancelWavefrontTilt)
                       devStatus= wfs.WFS_ZernikeLsf(instrumentHandle, byref(zernikeOrders), arrayZernikeUm[1:].ctypes.data, arrayZernikeOrdersUm.ctypes.data, byref(roCMm))
                       res[:,r]= arrayZernikeUm[3:]
                       r+1
                       #array[circmask] =  u
                       #np.flip(array, axis=0)
                       #plt.imshow(array)
                       #plt.pause(0.5)
                       
                       time.sleep(0.1)
                   z1=np.mean(res,1)
                   z += s * z1 / (2 * probe_amp)
                   time.sleep(0.01) # Wait for 1 second
                   
               print(f"Actuador:{ind+1}") 
               influ_M[:,ind]=z
    
   
    Z2C=np.zeros([num_act,num_modes-1]) #genero la matriz donde voy a alojar los valores
    Z2C=hcipy.util.inverse_tikhonov(influ_M, rcond=0.05, svd=None) #realizo la pseudo-inversa
    # metodo 2 PI Z2C=np.linalg.pinv(influ_M)
    #Recorro los primeros 15 polinomios con amplitud 
    
    ### ANALISISIS MATRIZ DE INFLUENCIA##
    plt.imshow(influ_M)
    np.save('influ.npy', influ_M)
    u, s, vh = np.linalg.svd(influ_M, full_matrices=True)
    x=np.arange(1,66)
    plt.plot(x,np.log(s))
    plt.grid()
    plt.ylabel('Eigenvalues')
    plt.ylabel('Eigenmodes')
    plt.show()
    
    
    
    """#guardo la matriz
    import scipy.io
    scipy.io.savemat('plano_M.mat', {'mydata': plano})
    
    """
    
    #Flattening
    
    leakage=0.02 #pèrdida
    gain=0.2 #ganancia
    actuators = np.zeros(88)
    
    #actuator=plano
    for n in np.arange(0,20):
        dm.Send(actuators)
        devStatus = wfs.WFS_TakeSpotfieldImage(instrumentHandle, exposureTimeAct, masterGainAct)
        devStatus = wfs.WFS_CalcSpotsCentrDiaIntens(instrumentHandle, dynamicNoiseCut, calculateDiameters)
        devStatus = wfs.WFS_CalcSpotToReferenceDeviations(instrumentHandle, cancelWavefrontTilt)
        devStatus= wfs.WFS_ZernikeLsf(instrumentHandle, byref(zernikeOrders), arrayZernikeUm[1:].ctypes.data, arrayZernikeOrdersUm.ctypes.data, byref(roCMm))
        actuators = (1-leakage) * actuators - gain * Z2C.dot(arrayZernikeUm[3:])
        n+1
        time.sleep(1) # Wait for 1 second
    plano= actuators
    print("Aplanado")
    
    
    array = np.zeros((12,12))
    circmask = draw.circle(5.5,5.5,5.5,(12,12))
    array[circmask] =  plano
    
    plt.imshow(array) #grafico el estado del DM
    plt.colorbar()
    np.save('plano.npy', plano)
    scipy.io.savemat('planoA.mat', {'mydata': plano})
    dm.Reset()
    devStatus = wfs.WFS_close(instrumentHandle)
