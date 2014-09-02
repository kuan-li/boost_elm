function [TestingTime, TestingAccuracy, TY, confusion_matrix] = elm_test(DATA, classifier)

% Input parameters

TV.T=DATA.TV.T;
TV.P=DATA.TV.P;
NumberofTestingData=DATA.NumberofTestingData;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=classifier.InputWeight*TV.P;
clear TV.P;             %   Release input of testing data
ind=ones(1,NumberofTestingData);
BiasMatrix=classifier.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(classifier.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here
end
TY=(H_test' * classifier.OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

%%%%%%%%%% Calculate training & testing classification accuracy
[~, label_index_expected]=max(TV.T,[],1);
[~, label_index_actual]=max(TY,[],1);
[confusion_matrix, ~]=confusionmat(label_index_expected, label_index_actual);

TestingAccuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));