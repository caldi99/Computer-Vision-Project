%Author : Daniela Cuza
% in this script we convert the model into .onnx

clear all
close all

load('model.mat')

filename = "model.onnx";
exportONNXNetwork(net,filename)