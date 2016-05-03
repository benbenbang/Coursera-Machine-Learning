function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


    pred = pval < epsilon;
    tp = zeros(size(pred,1),1);
    fp = zeros(size(pred,1),1);
    fn = zeros(size(pred,1),1);

    % calc True Positive, y = 1 && pval = 1
    %for i = 1:length(pred)
    %    if pred(i) == 1 && yval(i) == 1
    %        tp(i) = 1;
    %    end
    %end

    %tp = sum(tp);

    % calc False Positive, y = 0 && pval = 1
    %for i = 1:length(pred)
    %    if pred(i) == 1 && yval(i) == 0
    %        fp(i) = 1;
    %    end
    %end

    %fp = sum(fp);

    % calc False Negative, y = 1 && pval = 0
    %for i = 1:length(pred)
    %    if pred(i) == 0 && yval == 1
    %        fn(i) = 1;
    %    end
    %end

    %fn = sum(fn);

    % calc tp, fp, fn
    tp = sum((pred == 1) & (yval == 1));
    fp = sum((pred == 1) & (yval == 0));
    fn = sum((pred == 0) & (yval == 1));
    
    % calc Precision and Recall 
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);

    F1 = 2 * (precision * recall / (precision + recall));

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
