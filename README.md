# Boosting Weighted-ELM
Boosting weighted ELM for imbalanced learning

Kuan Li, Xiangfei Kong, Zhi Lu, Liu Wenyin, Jianping Yin. Boosting weighted ELM for imbalanced learning. Neurocomputing 128: 15-21 (2014)

### Usage

Step1 read training and testing data information
	DATA=parseTrainTestFile('diabetes_train','diabetes_test');

Step2 Train
Regular version:
[TrainingTime,TrainingAccuracy, Y, classifier, confusion_matrix] = elm_train(DATA,2^20, 200, 'sig', 2, 0);


kernel version：
[TrainingTime, TrainingAccuracy, Y, classifier, confusion_matrix] =  elm_kernel_train(DATA, 2^20,'RBF_kernel', 0.1, 1, 0)

Step3 Test
Regular version:
[TestingTime, TestingAccuracy, TY, confusion_matrix] = elm_test(DATA, classifier)

kernel version：
[TestingTime, TestingAccuracy,TY, confusion_matrix] = elm_kernel_test(DATA, classifier)
