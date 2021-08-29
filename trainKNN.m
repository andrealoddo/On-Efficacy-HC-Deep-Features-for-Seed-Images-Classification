function [trainedClassifier, validationAccuracy, bestAcc, bestDistance, bestK, CMN] = trainKNN(trainingData, types, k, distance)

%nn = [1 : 7];
%distances = {'cityblock', 'chebychev', 'correlation', 'cosine', 'euclidean', ...
%    'hamming', 'jaccard', 'minkowski', 'seuclidean', 'spearman'};

nn = [6];
distances = {'cityblock'};

bestAcc = 0;
bestDistance = '';
bestK = 0;
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

if nargin == 4
    classification = fitcknn(...
        predictors, ...
        response, ...
        'Distance', distance, ...
        'NumNeighbors', k, ...
        'DistanceWeight', 'Equal', ...
        'Standardize', true);% Create the result struct with predict function
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
    bestDistance = distance;
    bestK = k;
    
else
    for k = 1:numel(nn)
        for dis = 1:numel(distances)
            sprintf('Computing kNN with k = %d, distance = %s', nn(k), distances{dis});
            % Train a classifier
            classification = fitcknn(...
                predictors, ...
                response, ...
                'Distance', distances{dis}, ...
                'NumNeighbors', nn(k), ...
                'DistanceWeight', 'Equal', ...
                'Standardize', true);
            
            %     classification = fitcknn(predictors,response,'OptimizeHyperparameters','auto',...
            %     'HyperparameterOptimizationOptions',...
            %     struct('AcquisitionFunctionName','expected-improvement-plus'), 'ClassNames', classes);
            
            %     classification = fitcecoc(predictors,response,'OptimizeHyperparameters','auto',...
            %     'HyperparameterOptimizationOptions',...
            %     struct('AcquisitionFunctionName','expected-improvement-plus'), 'ClassNames', classes);
            
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
            CMN = CMN(1:end-1, 1:end-1);
            
            [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
            
            if( 100*stats{5, AVGtype} > bestAcc) % ACC
                bestAcc = 100*stats{5, AVGtype};
                bestDistance = distances{dis};
                bestK = nn(k);
            end
        end
    end
end
end

%100*stats{5, AVGtype}  % Pre
%100*stats{15, AVGtype} % Spe
%100*stats{11, AVGtype} % Sen
%100*stats{20, AVGtype} % F1
