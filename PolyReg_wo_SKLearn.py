import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
plt.style.use("seaborn")

def ConvertingCSVToListData (fileNameCsv):
    filecsv = open(fileNameCsv, 'r')
    filecsvUnformattedData = filecsv.readlines() # readlines() method returns a list containing each line in the file as a list item
    filecsvFormattedData = list() # list() function is used for creating a list object on a python language
    # del filecsvUnformattedData[0] # Deleting  the column titles
    # Spliting the data by column, and converting into a formatted data list
    for line in filecsvUnformattedData:
        initialLine = line.split(',')
        data = int(initialLine[0])
        voltage =  float(initialLine[1])
        filecsvFormattedData.append([data, voltage])
    filecsv.close()
    return filecsvFormattedData

def NoisyDataList(fileNameList):
    NoisyData = list()
    # len() function stands for returning the number of items in a a list 
    # # range(len(()) is used for together when iterating over a mutable sequence, and  to modify the list at certain positions.

    for i in range(len(fileNameList)):
        if i != 0: # if i is not equal with 0 (condition check)
            if fileNameList[i][1] - fileNameList[i-1][1] > 0.01: # 0.01, it means the voltage difference is 0.01 V
                NoisyData.append(i)
    return NoisyData

def NoisyDataCorrection(NoisyDataList, BatteryDataList):
    valueAverage = abs((BatteryDataList[NoisyDataList + 1][1] + BatteryDataList[NoisyDataList - 1][1])/2) # Simple statistic for correcting the noisy data point between each coloumn
    BatteryDataList[NoisyDataList][1] = valueAverage # The value of simple statistic is entered into its place (the coloumn)

def reverselist(BatteryDataList):
    reversedlist = list()
    for i in range(len(BatteryDataList)): # len(list) will indicate the number of contained list, and range () will count a loop from zero.
        reversedlist.append(BatteryDataList[len(BatteryDataList) - 1 - i])
    return reversedlist

def polynomialNumpyFit(xdata, ydata, degree):
    coefficientsvalue = np.polyfit(xdata, ydata, degree) # Standard polynomial equation by using Numpy Library
    polynomialmodel = list() # result of polynomial equation (model)
    for x in xdata:
        y = 0
        i = 0
        while degree - i >= 0:
            y = y + coefficientsvalue[i] * x ** (degree - i)
            i = i + 1
        polynomialmodel.append(y)
    return polynomialmodel

def ChiSquaredEquation (observeddata, expecteddata):
    TotalChiSquaredValue = 0
    for i in range(len(observeddata)):
        expecteddata = expecteddata[i]
        observeddata = observeddata[i]
        if expecteddata != 0:
            TotalChiSquaredValue = TotalChiSquaredValue + abs(((observeddata - expecteddata)**2) / expecteddata) # the equation / formula of Chi-Square Test
        elif abs(observeddata - expecteddata) <= 0.000001:
            TotalChiSquaredValue = TotalChiSquaredValue + 0
        else:
            TotalChiSquaredValue = TotalChiSquaredValue + (observeddata + expecteddata)**2
    return TotalChiSquaredValue

def MaximumDegreeFit (xdata, ydata):
    n = 1 
    chisSquaredValue = list()
    while n <= 8: # n = 8, It means only check order from n = 1 until 8, in order to minimize compute time
        ChiSquared = ChiSquaredEquation(ydata, polynomialNumpyFit(xdata, ydata, n))
        chisSquaredValue.append([n, ChiSquared])
        n = n + 1
    # Minimal degree
    minimalDegree = 0
    minimalChiSquared = 1000000000
    for i in range(len(chisSquaredValue)):
        if chisSquaredValue[i][1] < minimalChiSquared:
            minimalChiSquared = chisSquaredValue[i][1]
            minimalDegree = i
    return chisSquaredValue[minimalDegree]

def PolynomialRegression(X ,Y):
    mymodel = np.poly1d(np.polyfit(X, Y, 8)) # Degree of Polynomial Regression = 8
    Y_pred = mymodel(X)
    return Y_pred

def main():

    # Step 1 : Reading the data  from csv, and converting into list
    path = r"C:\Users\Admin\Desktop\IoT-BMS-Roy\BMS Project\PolyReg\Data\BMS_OCV.csv"
    batteryData = ConvertingCSVToListData(path)
    # print(batteryData)

    dataMeasurement = list()
    voltage = list()
    for data in batteryData:
        dataMeasurement.append(data[0])
        voltage.append(data[1])
    
    # print(dataMeasurement)
    # print("")
    # print(voltage)
    # print(batteryData)
    # print("")

    # Step 2-3 : Preparation the data
    # Step 2 : Checking the battery data list, if there is an empty data
    noisydata = NoisyDataList(batteryData)  # Noisy data point means when you are trying to discharge the battery (the data is not supposed to be like that) 
                                            # there is anomaly (spike or increasement voltage value) but it is supposed to be discharging

    # print(noisydata)

    # Step 3 : Fixing the NoisyData point 
    for noisydatapoint in noisydata:
        # fixingdata = NoisyDataCorrection(noisydatapoint - 1, batteryData)
        fixingdata = NoisyDataCorrection(noisydatapoint, batteryData)
    
    # Step 4 : Checking the fixing data is already on the list
    dataMeasurement = list()
    voltage = list()
    for data in batteryData:
        dataMeasurement.append(data[0])
        voltage.append(data[1])
    
    # print(batteryData)
    # print("")

    # Step 5 : In order to normalize the data into 100% scale, the fixing data is needed to be transform into reversed-order list
    # reversed-order list
    reversedDataMeasurement = reverselist(dataMeasurement)
    reversedVoltage = reverselist(voltage)

    # print(reversedDataMeasurement)
    # print("")
    # print(reversedVoltage)
    # print("")

    # Step 6 : After reversing the order of the list, normalize the data measurement into unit of 0 to 100% scale
    normalizedDataMeasurement = len(reversedDataMeasurement)
    incrementvalue = 100/normalizedDataMeasurement # 100 is the value of maximum soc
                                                   # IncrementValue = soc increment is divided by total of data measurement
    normalizedSoC = list() # Variable of Normalized SoC based on array
    i =  0 # Initiation for SoC
    while i < normalizedDataMeasurement:
        normalizedSoC.append(i*incrementvalue)
        i += 1 

    # print(normalizedSoC)
    # print("")
    # print(reversedVoltage)

    # Step 7 : After making the data measurement into percentage value vs Voltage
    #          The data is fitted into a polynomial function to get the state of charge as a function of open circuit voltage.
    #          Numpy polynomial method is recommended for new code as it is more stable numerically.
    #          p(x) = p[0]*x**deg + ... + p[deg] of degree deg to points(x,y)
    #          Syntax = nump.ply (X, Y, deg, rcond = None, full = False, w = None, cov = False)
    #          output of syntax is coefficients value based of quadratic polynomial           
    #          The output of coefficients is going to be tested with Chi Square Test (A low value of chi-square means there is a high correlation between two sets of data)
    #          By performing a Chi Square Test, it is able to get a optimal coefficient value of quadratic polynomial


    X = np.array(voltage)
    Y = np.array(normalizedSoC)
    
    Y_pred = PolynomialRegression(X, Y)

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    residual_Error = Y-Y_pred
    n = len(Y)

    # R Square Error Calculation
    SST = np.sum([((Y-Y_mean)**2)])
    SSE = np.sum([((Y-Y_pred)**2)])
    R_sqr = (SST - SSE)/SST
    print("R Square Value =", R_sqr)

    # Mean Absolute Error Calculation
    MAE = (np.sum([(abs(residual_Error))]))/n 
    print("Mean Absolute Error =", MAE)

    # Mean Square Error Calculation

    MSE = (np.sum([(residual_Error**2)]))/n
    RMSE = sqrt(MSE)
    print("Root Mean Square Error =", RMSE)

    # Plotting the Actual SoC and Predicted SoC Values
    plt.scatter(Y, X,s=1,c = "green", marker="X", label = "Actual SoC") # Actual SoC
    plt.scatter(Y_pred, X,s=1,c = "red", label = "Predicted SoC") # Predicted SoC
    plt.legend()
    plt.title('SoC Estimation using OCV')
    plt.xlabel('SoC(%)')
    plt.ylabel('Voltage (V)')
    plt.show()

main()