function [trainedClassifier, validationAccuracy, bestAcc, bestKernel, bestDistribution, CMN] = trainBayes(trainingData, types)

bestAcc = 0;
bestKernel = '';
bestDistribution = '';
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

      
classification = fitcnb(...
    predictors, ...
    response, ...
    'Kernel', 'normal', ...
    'Support', 'Unbounded', ...
    'DistributionNames', 'kernel');% Create the result struct with predict function

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

if( size(CMN,1) > numel(classes) )
    CMN = CMN(1:end-1, 1:end-1);
end

[microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);

bestAcc = 100*stats{5, AVGtype};
    
end

