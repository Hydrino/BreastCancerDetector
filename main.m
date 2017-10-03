%The main file!
%Let's get started bro!!

%Load data 
data = dlmread('data.csv',',');

X_train = data(1:340,1:30);
Y_train = data(1:340,31);

%normalise the data first
X_train = normalise(X_train);

%add ones
X_train = [ones(m,1) X_train];

init_theta = zeros(n+1,1);

%regularisation parameter
lambda=0;

%initial cost 
J = computeCost(X_train,init_theta,Y_train,lambda);

%using advanced optimisation
options =  optimset('GradObj', 'on', 'MaxIter', 400);
[theta cost] = fminunc(@(t)computeCost(X_train,t,Y_train,lambda),init_theta,options);


% test this theta on cross validation set

X_val = data(341:456,1:30);
Y_val = data(341:456,31);

%normalise this first
X_val = normalise(X_val);

%add ones
X_val = [ones(length(X_val),1) X_val];

predictions = h(X_val,theta);

predictions(predictions>=0.4) = 1;
predictions(predictions<0.4) = 0;

accuracy = mean(double(predictions==Y_val))*100

true_pos=[];

for i=1:length(Y_val)
	if Y_val(i,1)==1 && predictions(i,1)==1
		true_pos = [true_pos; i];
	endif
endfor

precision = length(true_pos)/sum(predictions)

recall = length(true_pos)/sum(Y_val)

error = sum(Y_val!=predictions)

F = 2*precision*recall/(precision+recall)


