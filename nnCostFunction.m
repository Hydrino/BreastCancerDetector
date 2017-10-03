function [J grad] = nnCostFunction(nn_params,input_layer,hidden_layer,output_layer,X,Y,lambda)
	
	m = length(X);

	theta1 = reshape(nn_params(1:hidden_layer * 	(input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

	theta2 = reshape(nn_params((1 + (hidden_layer * 	(input_layer + 1))):end), ...
                 output_layer, (hidden_layer + 1));

	[h a2] = forwardProp(X,theta1,theta2);

	a = log(h)*Y;
	b = log(1-h)*(1-Y);
	
	temp1 = theta1;
	temp1(:,1)=0;

	temp2 = theta2;
	temp2(:,1) = 0;

	reg = (lambda/(2*m))*(sum(sum(temp1.^2)) + sum(sum	(temp2.^2)));

	J = (-a-b)/m + reg;
	
	del3 = h - Y';
	
	del2 = theta2'*del3;
	del2 = del2.*a2.*(1-a2);
	del2 = del2(2:end,:);
	
	delta2 = del3*a2';
	delta1 = del2*X;

	Theta1_grad = (1/m)*(delta1+lambda*temp1);
	Theta2_grad = (1/m)*(delta2+lambda*temp2);

	grad = [Theta1_grad(:);Theta2_grad(:)];
	
endfunction