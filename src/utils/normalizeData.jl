export normalizeData
function normalizeData(X)
# image normalization as described in
# https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
#
m = mean(X,1);
s = std(X,1);
adjusted_s = max.(s, 1.0/sqrt(size(X,1)));
for k=1:size(X,2)
	X[:,k] = (X[:,k] - m[k])./adjusted_s[k];
end
return X;
end
