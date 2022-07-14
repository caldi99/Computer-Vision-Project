"""
    This file was entirely written by Francesco Caldivezzi
"""

import cv2

#Load model
net = cv2.dnn.readNet("../model.onnx")
blob = cv2.dnn.blobFromImage(cv2.imread("../image.jpg"), scalefactor = 1.0, size=(224, 224), swapRB = True, crop = False)

#Set Input
net.setInput(blob)

#Compute output
out = net.forward()

file = open("results.txt","w")

#Print Results
for i in range(224):
    for j in range(224):
        file.write("R : {} C : {} value : {}\n".format(i,j,out[0,0,i,j]))

file.close()
