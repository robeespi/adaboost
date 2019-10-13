%Function to predict class {+1,1}
function [ht,alphasum] = ada_boost_predict(modelvariables,X)

ht = zeros(size(X,1),1);
g = zeros(size(X,1),1);

alphasum = 0;
for i=1:length(modelvariables)
  g = sign(modelvariables{i}.s*(X(:,modelvariables{i}.ind)-modelvariables{i}.x0)); 
  
  %Getting the final classification
  ht = ht + modelvariables{i}.var_alpha*g;
  alphasum = alphasum + modelvariables{i}.var_alpha;
end
