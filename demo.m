clear all
clc
close all

load example1
%% Train the HP classifier
Input.Data_Train=DTra1;   % Training data
Input.Label_Train=LTra1;  % Ground truth
Input.LayerNum=5;         % Layer number
[Output0]=HP(Input,'learning');
%% Validate the trained HP classifier with Mode A
Input1.Data_Test=DTes1;   % Validation data
Input1.Syst=Output0.Syst; % Trained HP classifier
[Output1]=HP(Input1,'testinga'); 
ConfusionMatrix=confusionmat(LTes1,Output1.Labels);  % Confusion matrix of the prediction result
Acc1=sum(sum(ConfusionMatrix.*eye(size(ConfusionMatrix,1))))/sum(sum(ConfusionMatrix)) % Accuracy of the prediction result 
%%  Validate the trained HP classifier with Mode B
Input1.Data_Test=DTes1;    % Validation data
Input1.Syst=Output0.Syst;  % Trained HP classifier
[Output1]=HP(Input1,'testingb');
ConfusionMatrix=confusionmat(LTes1,Output1.Labels);  % Confusion matrix of the prediction result
Acc2=sum(sum(ConfusionMatrix.*eye(size(ConfusionMatrix,1))))/sum(sum(ConfusionMatrix)) % Accuracy of the prediction result 