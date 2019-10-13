%This function call the ada boost implementation function for obtain the
%model parameters as input
%Other inputs are the train data set, this include features of each sample
%and train labels as well as the number of iterations desired for train the
%model which is 100 number of boosting iterations.

function [trainerror,modelvariables,traccuracy] = trainmodel_evaluation(xtrain,ytrain,number_iterations)
	
trainerror=zeros(number_iterations,1);

M=[];
c=[];

%Getting model parameters 
modelvariables=ada_boost_implementation(xtrain,ytrain,number_iterations);

%Run over iterations to obtain parameters to predict a class and then
%calculate margin and finally cumulative distribution to plot
for k=1:min(number_iterations,size(modelvariables,1))
  
  [yb,b] = ada_boost_predict(modelvariables(1:k),xtrain);
  y_train_prediction = sign(ada_boost_predict(modelvariables(1:k),xtrain));
  trainerror(k) = sum(y_train_prediction ~= ytrain);
  c=yb./b;
  mar = c.* ytrain ;
  M = [M,mar]; 
end

traccuracy = 1 - trainerror(end)/size(ytrain,1);

f1 = figure;
f2 = figure;

%Plotting training error
figure(1)
plot(trainerror./size(ytrain,1),'g')
ylabel('Error');
xlabel('Number of Boosting Iterations'), hold on,

%Plotting cumulative distribution over margin
figure(2) 
n = length(ytrain);
hola=(1/n:1/n:n/n)';
plot(sort(M(1:n,10)),hola,'b',sort(M(1:n,100)),hola,'r');
legend('After twenty four iterations','After one hundred iterations');