%Function to predict the class value of according to the models parameters
%trained

%Input parameters model, test data set, number of iterations
%Output test error over the iterations and test accuracy

function [testerror,testaccuracy] = testing(modelvariables, xtest,ytest,number_iterations)

testerror=zeros(number_iterations,1);

for k=1:min(number_iterations,size(modelvariables,1))
    
  %Getting the class prediction  
  y_boost_test=sign(ada_boost_predict(modelvariables(1:k),xtest));
  
  %Storing the error for each iteration
  testerror(k) = sum(y_boost_test ~= ytest);  
  
end

%Storing test accuracy
testaccuracy = 1 - testerror(end)/size(ytest,1);

%Plotting test error over the number of iterations
figure(1)
plot(testerror./size(ytest,1),'r')
ylabel('Error');
xlabel('Number of Boosting Iterations'), hold on,
