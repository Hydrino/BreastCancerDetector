%The main file!
%Let's get started bro!!

%Load data 
data = dlmread('data.csv',',');

X_train = data(170:569,1:30);
Y_train = data(170:569,31);

[m n] = size(X_train);

%normalise the data first
X_train = normalise(X_train);

%add ones
X_train = [ones(m,1) X_train];

%regularisation parameter
lambda = 0.01;

% first layer = 30,hidden layer = 6, output layer = 1
input_layer = 30;
hidden_layer = 6;
output_layer = 1;

[theta] = trainNeuralNet(X_train,Y_train,input_layer,hidden_layer,output_layer,lambda);

theta1 = reshape(theta(1:hidden_layer*(input_layer+1)),hidden_layer,(input_layer+1));

theta2 = reshape(theta((1 + (hidden_layer * (input_layer + 1))):end), ...
                 output_layer, (hidden_layer + 1));

%now it's time to validate!
X_val = data(1:169,1:30);
Y_val = data(1:169,31);

%normalise the data
X_val=normalise(X_val);

%add ones
X_val = [ones(size(X_val,1),1) X_val];

pred = forwardProp(X_val,theta1,theta2);
pred = pred';
pred(pred>=0.4)=1;
pred(pred<0.4)=0;

accuracy = mean(double(Y_val==pred))*100

error=sum(pred!=Y_val)



