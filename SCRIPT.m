addpath('C:\Users\loand\Google Drive\Ricerca\Codes\MATLAB\Utilities');
warning('off');
TEST = 0;
TEST = 5; % 2 = KNN, 3 = SVM, 4 = Bayes, 5 = Ensemble

CNN = 0;

labelsPath = 'Labels';
dataset = 'Cagliari';
dataset = 'Canada';

if CNN == 1
    featPath = 'Features\CNN';
elseif CNN == 0
    featPath = 'Features';
end

features = dir( fullfile(featPath, dataset, '*.mat') );

types = {'All', 'BW', 'BWGray', 'BWColor', 'Color', 'Gray', 'GrayColor'};

AVGtype = "microAVG";

%CNN
if( CNN == 1 )
    features = dir( fullfile(featPath, dataset, '*.mat') );
    labels = load( fullfile(labelsPath, strcat(dataset, '.mat') ) );
    labels = labels.labels;
    types = {'CNNvgg16', 'CNNvgg19', 'CNNalex', 'CNNgoogle', 'CNNincv3', 'CNNresnet101', 'CNNresnet18', 'CNNresnet50', 'CNNseednet'};
end

%%% TEST
if TEST == 1
    
    t = 2;
    if( CNN == 0 )
        trainingData = load(fullfile(featPath, dataset, features(t).name));
    else
        trainingData = load(fullfile(featPath, dataset, features(t).name));
        trainingData.features(:, end+1) = labels;
    end
    [trainedClassifier, validationAccuracy, bestAcc, bestDistance, bestK, CMN] = trainBis(trainingData, types{t}, 6, 'cityblock');
    [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);

elseif TEST == 2
    
    for t = 1:numel(types)
        
        sprintf('Loading... %s (type = %s)', features(t).name, types{t});
        if( CNN == 0 )
            trainingData = load(fullfile(featPath, dataset, features(t).name));
        else
            trainingData = load(fullfile(featPath, dataset, features(t).name));
            trainingData.features(:, end+1) = labels;
        end
        
        [trainedClassifier, validationAccuracy, bestAcc, bestDistance, bestK, CMN] = trainKNN(trainingData, types{t});
        
        [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
        
        fprintf(' & ')  % &
        fprintf([types{t}, ' & '])  % Type
        %fprintf([bestDistance, ' & '])  % Dis
        %fprintf([num2str(bestK), ' & '])  % Type
        fprintf([num2str(100*stats{5, AVGtype}, "%.2f") , ' & '])  % Acc
        fprintf([num2str(100*stats{6, AVGtype}, "%.2f") , ' & '])  % Pre
        fprintf([num2str(100*stats{15, AVGtype}, "%.2f") , ' & ']) % Spe
        fprintf([num2str(100*stats{11, AVGtype}, "%.2f") , ' & ']) % Sen
        fprintf([num2str(100*stats.macroAVG(23), "%.2f") , ' & '])  % Mavg
        fprintf([num2str(100*stats{20, AVGtype}, "%.2f") , ' \\\\ \n']) % F1
    end
    
elseif TEST == 3
    MFM_base = 0;
    for t = 1:numel(types)
        
        sprintf('Loading... %s (type = %s)', features(t).name, types{t});
        if( CNN == 0 )
            trainingData = load(fullfile(featPath, dataset, features(t).name));
        else
            trainingData = load(fullfile(featPath, dataset, features(t).name));
            trainingData.features(:, end+1) = labels;
        end        
        [trainedClassifier, validationAccuracy, ...
            bestBoxConstraint, bestKernelScale, bestKernelFunction, bestPolynomialOrder, CMN] = trainSVM(trainingData, types{t});        
        [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
        
        fprintf( ' & ')  % Type
        fprintf([types{t}, ' & '])  % Type
        %fprintf([num2str(bestBoxConstraint), ' & '])  
        %fprintf([num2str(bestKernelScale), ' & ']) 
        %fprintf([bestKernelFunction, ' & '])  
        %fprintf([num2str(bestPolynomialOrder), ' & '])  
        fprintf([num2str(100*stats{5, AVGtype}, "%.2f") , ' & '])  % Acc
        fprintf([num2str(100*stats{6, AVGtype}, "%.2f") , ' & '])  % Pre
        fprintf([num2str(100*stats{15, AVGtype}, "%.2f") , ' & ']) % Spe
        fprintf([num2str(100*stats{11, AVGtype}, "%.2f") , ' & ']) % Sen
        fprintf([num2str(100*stats.macroAVG(23), "%.2f") , ' & ']) % Mavg
        fprintf([num2str(100*stats{20, AVGtype}, "%.2f") , ' \\\\ \n']) % F1
        
        if(100*stats{20, AVGtype} > MFM_base)
            MFM_base = 100*stats{20, AVGtype};
            CM_best = CMN;
            stats_best = stats;
        end
    end
    
    
elseif TEST == 4
    
    for t = 1:numel(types)
        
        sprintf('Loading... %s (type = %s)', features(t).name, types{t});
        if( CNN == 0 )
            trainingData = load(fullfile(featPath, dataset, features(t).name));
        else
            trainingData = load(fullfile(featPath, dataset, features(t).name));
            trainingData.features(:, end+1) = labels;
        end        
        [trainedClassifier, validationAccuracy, bestAcc, bestKernel, bestDistribution, CMN] = trainBayes(trainingData, types{t});        
        [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
        
        fprintf( ' & ')  % Type
        fprintf([types{t}, ' & '])  % Type
        %fprintf([num2str(bestBoxConstraint), ' & '])  
        %fprintf([num2str(bestKernelScale), ' & ']) 
        %fprintf([bestKernelFunction, ' & '])  
        %fprintf([num2str(bestPolynomialOrder), ' & '])  
        fprintf([num2str(100*stats{5, AVGtype}, "%.2f") , ' & '])  % Acc
        fprintf([num2str(100*stats{6, AVGtype}, "%.2f") , ' & '])  % Pre
        fprintf([num2str(100*stats{15, AVGtype}, "%.2f") , ' & ']) % Spe
        fprintf([num2str(100*stats{11, AVGtype}, "%.2f") , ' & ']) % Sen
        fprintf([num2str(100*stats.macroAVG(23), "%.2f") , ' & '])  % Mavg
        fprintf([num2str(100*stats{20, AVGtype}, "%.2f") , ' \\\\ \n']) % F1

    end
    
elseif TEST == 5
    MFM_base = 0;
    for t = 1:numel(types)
        
        sprintf('Loading... %s (type = %s)', features(t).name, types{t});
        if( CNN == 0 )
            trainingData = load(fullfile(featPath, dataset, features(t).name));
        else
            trainingData = load(fullfile(featPath, dataset, features(t).name));
            trainingData.features(:, end+1) = labels;
        end        
        [trainedClassifier, validationAccuracy, bestAcc, bestKernel, bestDistribution, CMN] = trainEnsemble(trainingData, types{t});        
        [microAVG, macroAVG, wAVG, stats] = computeMetrics(CMN);
        
        fprintf( ' & ')  % Type
        fprintf([types{t}, ' & '])  % Type 
        fprintf([num2str(100*stats{5, AVGtype}, "%.2f") , ' & '])  % Acc
        fprintf([num2str(100*stats{6, AVGtype}, "%.2f") , ' & '])  % Pre
        fprintf([num2str(100*stats{15, AVGtype}, "%.2f") , ' & ']) % Spe
        fprintf([num2str(100*stats{11, AVGtype}, "%.2f") , ' & ']) % Sen
        fprintf([num2str(100*stats.macroAVG(23), "%.2f") , ' & '])  % Mavg
        fprintf([num2str(100*stats{20, AVGtype}, "%.2f") , ' \\\\ \n']) % F1
        
        if(100*stats{20, AVGtype} > MFM_base)
            MFM_base = 100*stats{20, AVGtype};
            CM_best = CMN;
            stats_best = stats;
        end
    end
    
end

