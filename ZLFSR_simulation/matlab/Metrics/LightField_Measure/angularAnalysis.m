function score =  angularAnalysis(feat)

featFFT = abs(fft(feat));

peak_index = find(diff(sign(diff(featFFT)))<0)+1;
if isempty(peak_index)
    score = 0;
    return;
end

featFFT_peakMax = max(featFFT(peak_index));
targetIndex = min(find(featFFT==featFFT_peakMax));

if featFFT(targetIndex)/featFFT(1) < 0.15
    score = 1000./((0.01*255)^2+mean(feat));
else
    score = targetIndex;
end

end
