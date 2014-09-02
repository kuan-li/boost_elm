function DATA=parseTrainTestFile(TrainingData_File, TestingData_File)

%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
%   Release raw training data array
clear train_data;

NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%%% Preprocessing the data of classification
sorted_target=sort(T,2);
label=zeros(1,1);
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(NumberofTrainingData)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;
NumberofOutputNeurons=number_class;

%%%%%%%%%% Processing the targets of training
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
num_per_class=zeros(1,number_class); %%%%
class_belong=zeros(1,NumberofTrainingData); %%%%

for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break;
        end
    end
    temp_T(j,i)=1;
    num_per_class(j)=num_per_class(j)+1; %%%%
    class_belong(i)=j; %%%%
end
T=temp_T*2-1;

weights{1}=ones(1, NumberofTrainingData);

class_weights=1./num_per_class;
class_weights=class_weights./max(class_weights); %Normalization

weights{2}=class_weights( class_belong );
weights{3}=class_weights( class_belong );
weights{3}(num_per_class( class_belong) > NumberofTrainingData/number_class) = ...
    weights{3}(num_per_class( class_belong) > NumberofTrainingData/number_class) *0.618;


%weights initial for adaboost
distribution=(1/number_class)./num_per_class;
distribution_weights=distribution(class_belong);

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
%   Release raw testing data array
clear test_data;                                    

NumberofTestingData=size(TV.P,2);

%%%%%%%%%% Processing the targets of testing
temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TV.T(1,i)
            break;
        end
    end
    temp_TV_T(j,i)=1;
end
TV.T=temp_TV_T*2-1;


%prepare to output
DATA.T=T;
DATA.P=P;
DATA.NumberofTrainingData=NumberofTrainingData;
DATA.NumberofInputNeurons=NumberofInputNeurons;
DATA.NumberofOutputNeurons=NumberofOutputNeurons;

DATA.weights=weights;
DATA.distribution_weights=distribution_weights;

DATA.TV.T=TV.T;
DATA.TV.P=TV.P;
DATA.NumberofTestingData=NumberofTestingData;
end