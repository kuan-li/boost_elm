function [TrainingTime,TrainingAccuracy, Y, classifier, confusion_matrix] = ...
    elm_train(DATA, Regularization_coefficient, NumberofHiddenNeurons, ActivationFunction, WeightType, InputWeights)

% Input parameters
T=DATA.T;
P=DATA.P;
NumberofTrainingData=DATA.NumberofTrainingData;
NumberofInputNeurons=DATA.NumberofInputNeurons;

weights=DATA.weights;

if WeightType < 3
    W=diag(weights{WeightType+1});
else
    %InputWeights=InputWeights/max(InputWeights);
    W = diag(InputWeights);
end
clear weights;

C=Regularization_coefficient;
        
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
classifier.InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
classifier.BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=classifier.InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=classifier.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

classifier.ActivationFunction=ActivationFunction;
%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper

if NumberofHiddenNeurons < NumberofTrainingData
    % faster method 1 //refer to 2012 IEEE TSMC-B paper
    % classifier.OutputWeight=inv(eye(size(H,1))/C+ H  * W * H') * H * W * T';   
     classifier.OutputWeight=(eye(size(H,1))/C+ H  * W * H') \ ( H * W * T'); 
    %implementation; one can set regularizaiton factor C properly in classification applications 
else
    % classifier.OutputWeight=H * inv(eye(size(H',1))/C+ W * H' * H) * W * T';   
    classifier.OutputWeight=H * ( (eye(size(H',1))/C+ W * H' * H) \( W * T'));
    %implementation; one can set regularizaiton factor C properly in classification applications
end
%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 

%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;                  %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * classifier.OutputWeight)';                             %   Y: the actual output of the training data
clear H;

[~, label_index_expected]=max(T,[],1);
[~, label_index_actual]=max(Y,[],1);
[confusion_matrix, ~]=confusionmat(label_index_expected, label_index_actual);

TrainingAccuracy=sum(diag(confusion_matrix))/sum(confusion_matrix(:));