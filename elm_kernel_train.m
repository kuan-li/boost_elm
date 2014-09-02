function [TrainingTime, TrainingAccuracy, Y, classifier, confusion_matrix] = ...
    elm_kernel_train(DATA, Regularization_coefficient, Kernel_type, Kernel_para, WeightType, InputWeights)


% Input parameters
T=DATA.T;
classifier.P=DATA.P;

weights=DATA.weights;

if WeightType < 3
    W=diag(weights{WeightType+1});
else
    InputWeights=InputWeights/max(InputWeights);
    W = diag(InputWeights);
end
clear weights;

classifier.Kernel_type=Kernel_type;
classifier.Kernel_para=Kernel_para;
C = Regularization_coefficient;

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
n = size(T,2);
Omega_train = kernel_matrix((classifier.P)',classifier.Kernel_type, classifier.Kernel_para);
classifier.OutputWeight=((W * Omega_train+speye(n)/C)\(W * T'));
TrainingTime=toc;

%%%%%%%%%%% Calculate the training output
%   Y: the actual output of the training data

Y=(Omega_train * classifier.OutputWeight)';

%%%%%%%%%% Calculate training & testing classification accuracy
[~, label_index_expected]=max(T,[],1);
[~, label_index_actual]=max(Y,[],1);
[confusion_matrix, ~]=confusionmat(label_index_expected, label_index_actual);

TrainingAccuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));
