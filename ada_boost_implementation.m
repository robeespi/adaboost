%Input train features, train lables and number iterations
%Output model each stump data (alpha, threshold, index stump classification, class stump classification(direction) 

function [modelvariables] = ada_boost_implementation(X,y,number_iterations)

%Setting initial weights distribution
weights = ones(size(X,1),1); 
weights = weights/sum(weights);
var_alpha = 1;

%Cell to store model parameters
modelvariables = cell(number_iterations,1);

Ht = zeros(size(X,1),1);

for i=1:number_iterations
  
  %Storing decision stump for each iteration
  stumpvar = build_stump(X,y,weights);
   
  if ( stumpvar.werr < 0.5 && stumpvar.werr > 0.000001)
    
    %Weak hypothesis
    ht = sign(stumpvar.s*(X(:,stumpvar.ind)-stumpvar.x0));
    
    % insert alpha calculation here
    var_alpha = 0.5*log((1-stumpvar.werr)/stumpvar.werr);
    
    Ht = Ht + var_alpha*ht; % update the combined predictions
    
    % weight update
    weights=exp(-Ht.*y);
    weights=weights/sum(weights);
    
    modelvariables{i} = stumpvar;
    modelvariables{i}.var_alpha = var_alpha; 
  else  
      if i > 1
        i = i-1; break; 
      else
          modelvariables{i} = stumpvar;
          modelvariables{i}.var_alpha = 1;
          break;
      end
  end

end

%Return model parameters
modelvariables = modelvariables(1:i); 

