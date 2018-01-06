
function numel(Y)
    return length(Y)
end

function lastOne(n)
    ei = zeros(n)
    ei[end] = 1
    return ei
end

function fft2(Y)
    return fft(fft(Y,1),2)
end

function ifft2(Y)
    return ifft(ifft(Y,2),1)
end
    