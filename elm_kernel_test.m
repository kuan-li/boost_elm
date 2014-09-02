function [TestingTime, TestingAccuracy,TY, confusion_matrix] = ...
                            elm_kernel_test(DATA, classifier)

% Input parameters
TV.T=DATA.TV.T;
TV.P=DATA.TV.P;

%%%%%%%%%%% Calculate the output of testing input
tic;
Omega_test = kernel_matrix((classifier.P)',classifier.Kernel_type, classifier.Kernel_para,TV.P');
%   TY: the actual output of the testing data
TY=(Omega_test' * classifier.OutputWeight)';                            
TestingTime=toc;

%%%%%%%%%% Calculate training & testing classification accuracy
[~, label_index_expected]=max(TV.T,[],1);
[~, label_index_actual]=max(TY,[],1);
[confusion_matrix, ~]=confusionmat(label_index_expected, label_index_actual);

TestingAccuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));