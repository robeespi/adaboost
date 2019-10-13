%Main Function which load and prepare data 
%Also, call train and testing function

%clear
clear all;
%import data
datacsv = importdata('wdbc_data.csv');
X = datacsv.data;
%Transforming data to a convenient data type
rawy = datacsv.textdata(:,2);
strrawy = string(rawy);
strrawy(strrawy=='M') = -1;
strrawy(strrawy=='B') = 1;
Y = str2double(strrawy);

%Splitting the data into training and test set
%As it is required
N=300;
xtrain = X(1:N, :);
ytrain = Y(1:N);
xtest = X(N+1:end, :);
ytest = Y(N+1:end);

%Setup number iterations
number_iterations = 100;

%Running Adaboost Training
%Getting train error and margin plot
[trainerror,modelvariables,traccuracy] = trainmodel_evaluation(xtrain,ytrain,number_iterations);

%Running Adaboost Testing
%Getting final classification, test error for each iteration and test
%accuracy
[resulttest,testaccuracy] = testing(modelvariables,xtest,ytest,number_iterations);
 



