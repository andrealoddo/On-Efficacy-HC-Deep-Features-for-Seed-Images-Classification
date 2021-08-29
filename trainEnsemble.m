function [trainedClassifier, validationAccuracy, ...
    bestAcc, bestMethod, bestLearner, CMN] = trainEnsemble(trainingData, types, method, learner)

methods = {'Bag', 'Subspace', 'AdaBoostM2', 'LPBoost', 'RUSBoost', 'TotalBoost'};
%learners = {'discriminant', 'knn', 'tree'};
learners = {'knn', 'tree'};

methods = {'Bag'};
learners = {'tree'};

bestAcc = 0;
bestMethod = '';
bestLearner = '';
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

isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
%distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
%distributionNames(isCategoricalPredictor) = {'mvmn'};

if nargin == 4
        
%     template = templateTree(...
%         'MaxNumSplits', 33, ...
%         'NumVariablesToSample', 9);
    
    template = learner;
    classification = fitcensemble(...
        predictors, ...
        response, ...
        'Method', method, ...
        'NumLearningCycles', 30, ...
        'Learners', template);
    
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

    
    [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);

    bestAcc = 100*stats{5, AVGtype};
    %bestDistance = distance;
    %bestK = k;
    
else
    for k = 1:numel(methods)
        for dis = 1:numel(learners)  
            sprintf('Computing Ensemble with method = %s, learner= %s', methods{k}, learners{dis});

            if ((strcmp(methods{k},'Subspace')) && not(strcmp(learners{dis},'knn'))) continue; end

            if (not((strcmp(methods{k},'Subspace'))) && (strcmp(learners{dis},'knn'))) continue; end

            if (strcmp(methods{k},'Subspace')) && (strcmp(learners{dis},'tree')) continue; end
            
            %if not(strcmp(learners{dis},'knn')) not_knn=1; end

            %if (strcmp(methods{k},'Subspace')) && (not(strcmp(learners{dis},'knn'))) continue; end
                      
            classification = fitcensemble(...
                predictors, ...
                response, ...
                'Method', methods{k}, ...
                'NumLearningCycles', 30, ...
                'Learners', learners{dis});

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
            
            CMN = confusionmat(validationPredictions,response);
            %CMN = CMN(1:end-1, 1:end-1);
            
            [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
            
            if( 100*stats{5, AVGtype} > bestAcc) % ACC
                bestAcc = 100*stats{5, AVGtype};
                bestMethod = methods{k};
                bestLearner = learners{dis};
            end
        end
    end
end
end