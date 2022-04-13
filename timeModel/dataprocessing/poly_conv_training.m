clear all;
data_raw = dlmread('/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/res_conv.txt');

data = [];

for i = 1:size(data_raw, 1) % 行数
    if data_raw(i, 2) ~= 1
        data = [data; data_raw(i, :)];
    end
end



result = data(:,end: end);
ipt_all = data(:,1: end - 1);
[row, col] = size(ipt_all);

X_runtime = [];

for i = 1:row
    ipf = ipt_all(i, :);
    tmp1 = [ipf(1:3),ipf(6:8),ipf(14)]; 
    tmp2 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            tmp2 = [tmp2, tmp1(j)*tmp1(k)];
        end
    end
    tmp3 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            for l = k: length(tmp1)
                tmp3 = [tmp3, tmp1(j)*tmp1(k)*tmp1(l)];
            end
        end
    end    
    tmp = [tmp1, tmp2, tmp3]; %is the best 40%
    tmp = [tmp, ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)]; %output data
    tmp = [tmp, ipf(5)*ipf(6)*ipf(7)*ipf(8)]; %filter data
    tmp = [tmp, ipf(13)*ipf(14)*ipf(15)*ipf(16)]; %input data
    tmp = [tmp, ipf(13)*ipf(15)*ipf(16)]; %input data (padding)
    tmp = [tmp, ipf(13)*ipf(14)*ipf(16)]; %input data (padding)
    tmp = [tmp, ipf(17)]
    tmp = [tmp, ipf(18)]
    tmp = [tmp, ipf(17) * ipf(18)]
    X_runtime = [X_runtime; tmp];
end

runtime = result(:,1); % runtime
rmse = [];

%% lasso
y1 = runtime;
[B1,FitInfo1] = lasso(X_runtime, y1, 'CV', 10);
fprintf('Runtime model complexity: %d\n', sum(B1(:,FitInfo1.IndexMinMSE) ~= 0) + 1)

y_runtime = X_runtime * B1(:,FitInfo1.IndexMinMSE) + FitInfo1.Intercept(FitInfo1.IndexMinMSE);
mspe_runtime = sqrt(mean(((y_runtime - y1)./y1) .^ 2));
  mse_runtime = sqrt(mean(((y_runtime - y1)) .^ 2));
fprintf('%.4f, %.4f\n', mspe_runtime, mse_runtime);

coeffi_runtime =  [B1(:,FitInfo1.IndexMinMSE)', FitInfo1.Intercept(FitInfo1.IndexMinMSE)];
csvwrite('/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/coeff_conv.txt', coeffi_runtime);

