function theta = trainNeuralNet(X,Y,input_layer,hidden_layer,output_layer,lambda)

	theta1 = rand(hidden_layer,input_layer+1)*2*0.12-0.12;
	theta2 = rand(output_layer,hidden_layer+1)*2*0.12-0.12;
	initial_nn_params = [theta1(:);theta2(:)];

	options = optimset('MaxIter', 100);

	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer, ...
                                   hidden_layer, ...
                                   output_layer, X, Y, lambda);

	[theta, cost] = fminunc(costFunction, 			initial_nn_params, options);

endfunction