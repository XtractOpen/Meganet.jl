
export getCIFAR10
function getCIFAR10(n, CIFAR10dataPath::String = "",num::T = one(Float32)) where {T} ## num is just for indicating the type. Maybe there's a better solution?
nImg = [32,32,3];
ALL_IMG_TRAIN = zeros(T,prod(nImg),50000);
ALL_LABELS_TRAIN = zeros(T,10,50000);
IMG_TEST = zeros(T,prod(nImg),10000);
LABELS_TEST = zeros(T,10,10000);
TEMP = zeros(T,10,10000);
for i=1:6
	if i < 6
		filename = string(CIFAR10dataPath,"data_batch_",i,".mat");
	else
		filename = string(CIFAR10dataPath,"test_batch.mat");
	end
	file = matopen(filename);
	data = read(file, "data");
	labels = read(file, "labels");
	close(file);
	C = convert(Array{Int64,1},vec(labels+1));
    C = C + ((0:(10000-1))*10);
    TEMP[C] = 1;
	if i<6
		II = ((i-1)*10000 + 1):(i*10000);
		ALL_IMG_TRAIN[:,II] = data';
		ALL_LABELS_TRAIN[:,II] = TEMP;
	else
		IMG_TEST[:,:] = data'; 
		LABELS_TEST[:,:] = TEMP;
	end
    TEMP[:] = 0;
end


# # figure; imshow(reshape(IMG_TEST(:,75),32,32,3))

p = n/size(ALL_IMG_TRAIN,2);
n_train = n;
n_test = ceil(Int64,size(IMG_TEST,2)*p);

ptrain = randperm(size(ALL_IMG_TRAIN,2));
Y_train = ALL_IMG_TRAIN[:,ptrain[1:n_train]];
C_train = ALL_LABELS_TRAIN[:,ptrain[1:n_train]];

ptest = randperm(size(IMG_TEST,2));
Y_test = IMG_TEST[:,ptest[1:n_test]];
C_test = LABELS_TEST[:,ptest[1:n_test]];


Y_test = normalizeData(Y_test);
Y_train = normalizeData(Y_train);

return Y_train,C_train,Y_test,C_test;
end
# using MAT
# using PyPlot
# (Y_train,C_train,Y_test,C_test) = getCIFAR10(500);

# a = 1;


