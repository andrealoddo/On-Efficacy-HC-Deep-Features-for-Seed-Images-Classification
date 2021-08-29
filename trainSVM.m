function [trainedClassifier, validationAccuracy, ...
    bestBoxConstraint, bestKernelScale, bestKernelFunction, bestPolynomialOrder, CMN] = trainSVM(trainingData, types)

BoxConstraint = [1];%[1e-3:1e3];
KernelScale = {'auto'};%[1e-3:1e3];
KernelFunction = {'gaussian', 'linear', 'polynomial'};
PolynomialOrder = [2:4];

BoxConstraint = [1];%[1e-3:1e3];
KernelScale = {'auto'};%[1e-3:1e3];
KernelFunction = {'polynomial'};
PolynomialOrder = [2];


bestBoxConstraint = 1e-3;
bestKernelScale = 1e-3;
bestKernelFunction = 'gaussian';
bestPolynomialOrder = 2;
bestAcc = 0;
AVGtype = "macroAVG";

if strcmp(types, 'BW')
    inputTable = trainingData.FeaturesBW;
elseif strcmp(types, 'BWGray')
    inputTable = trainingData.FeaturesBWGray;
elseif strcmp(types, 'BWColor')
    inputTable = trainingData.FeaturesBWRGB;
elseif strcmp(types, 'Color')
    inputTable = trainingData.FeaturesColour;
elseif strcmp(types, 'Gray')
    inputTable = trainingData.FeaturesGray;
elseif strcmp(types, 'GrayColor')
    inputTable = trainingData.FeaturesGrayRGB;
elseif strcmp(types, 'All')
    inputTable = trainingData.AllFeaturesFixed;
elseif contains(types, 'CNN')
    inputTable = trainingData.features;
    inputTable = array2table(inputTable);
end

predictorNames = inputTable.Properties.VariableNames(1:end-1);
inputTable.Properties.VariableNames{end} = 'Label';
classes = unique( inputTable.Label(:) );

predictors = inputTable(:, predictorNames);
response = inputTable.Label;

for bc = 1:numel(BoxConstraint)
    for ks = 1:size(KernelScale,1)
        for kf = 1:numel(KernelFunction)
            for po = 1:numel(PolynomialOrder)
                sprintf('Computing kNN with BC = %f, KS = %f, KF = %s, PO = %d', ...
                    BoxConstraint(bc), KernelScale{ks}, KernelFunction{kf}, PolynomialOrder(po));
                
                if( ~strcmp( KernelFunction{kf}, 'polynomial' ) && po > 1)
                    continue;
                end
                
                if( ~strcmp( KernelFunction{kf}, 'polynomial' ) )
                    template = templateSVM(...
                        'KernelFunction', KernelFunction{kf}, ...
                        'KernelScale', KernelScale{ks}, ...
                        'BoxConstraint', BoxConstraint(bc), ...
                        'Standardize', true);
                    
                else
                    template = templateSVM(...
                        'KernelFunction', KernelFunction{kf}, ...
                        'PolynomialOrder', PolynomialOrder(po), ...
                        'KernelScale', KernelScale{ks}, ...
                        'BoxConstraint', BoxConstraint(bc), ...
                        'Standardize', true);
                end

                classification = fitcecoc(...
                    predictors,...
                    response,...
                    'Learners', template, ...
                    'Coding', 'onevsall');

                % Create the result struct with predict function
                predictorExtractionFcn = @(t) t(:, predictorNames);
                knnPredictFcn = @(x) predict(classification, x);
                trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));
                
                trainedClassifier.Classification = classification;
                
                % Perform cross-validation
                partitionedModel = crossval(trainedClassifier.Classification, 'KFold', 10);
                
                % Compute validation predictions
                [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
                
                % Compute validation accuracy
                validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
                
                %cm = confusionchart(response, validationPredictions);
                %CMN = cm.NormalizedValues;
                CMN = confusionmat(validationPredictions,response);
                %CMN = CMN(1:end-1, 1:end-1);
                
                [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
                
                if( 100*stats{5, AVGtype} > bestAcc) % ACC
                    bestAcc = 100*stats{5, AVGtype};
                    bestBoxConstraint = BoxConstraint(bc);
                    bestKernelScale = KernelScale(ks);
                    bestKernelFunction = KernelFunction{kf};
                    bestPolynomialOrder = PolynomialOrder(po);
                end
            end
        end
    end
end
end

%100*stats{5, AVGtype}  % Pre
%100*stats{15, AVGtype} % Spe
%100*stats{11, AVGtype} % Sen
%100*stats{20, AVGtype} % F1
