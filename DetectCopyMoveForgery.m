function DetectCopyMoveForgery
    clc
    clear
    blocksize = 8;
    overlap = 1;
    Nd = 16;
    Th = 0.9999;
    s_threshold = 2;

    % Get input image
    [filename, path] = uigetfile('*.*', "Select an Image");
    img = imread(fullfile(path, filename));
    imshow(img);
    title('Original image');
    [r, c, n] = size(img);
    if n > 1
        im = rgb2gray(img);
    else
        im = img;
    end
    figure;
    imshow(im), title('Gray image');

    % Display block size
    disp(['Block Size: ', num2str(blocksize)]);

    % Divide into overlapping blocks
    a = 1;
    for j = 1:overlap:(c - blocksize) + 1
        for i = 1:overlap:(r - blocksize) + 1
            sondos(a).block = im(i:i + blocksize - 1, j:j + blocksize - 1);
            sondos(a).position = [i, j];
            sondos(a).index = a;
            a = a + 1;
        end
    end

    % Measure execution time
    tic;

    % Apply DCT for each block
    sz = numel(sondos);
    DC = zeros(sz, 1);
    FDCT = zeros(sz, blocksize^2);  % Assuming blocksize x blocksize DCT
    for a = 1:sz
        [feature, ~] = featureExtraction(sondos(a).block);
        DC(a) = feature(1);
        FDCT(a, :) = feature(:)';
    end

    % Divide into groups
    numloss = 20;
    try
        % Increase the maximum number of iterations
        maxIterations = 500;
        opts = statset('MaxIter', maxIterations);
        [idx, centers] = kmeans(FDCT, numloss, 'Options', opts);

    catch
        error('Failed to converge. Consider adjusting parameters or preprocessing the data.');
    end

    G = cell(numloss, 1);
    for n = 1:numloss
        G{n} = find(idx == n);
    end

    % Draw segmentation
    col = jet(numloss);
    figure;
    imshow(im), title('Clustered image');
    for e = 1:numloss
        color = col(e, :);
        for ee = 1:numel(G{e})
            idx = G{e}(ee);
            rectangle('Position', [sondos(idx).position(2), sondos(idx).position(1), blocksize, blocksize], 'EdgeColor', color);
        end
    end

    % Detect CM and identify duplicates
    figure;
    imshow(im), title('Result image');
    detectedDuplicates = false(sz, 1);
    for n0 = 1:numloss
        emp = find(G{n0} == 0);
        if ~isempty(emp)
            A = zeros(numel(emp), blocksize^2 + 2);
            for a = 1:numel(emp)
                idx = G{n0}(emp(a));
                [f, ~] = featureExtraction(sondos(idx).block);
                A(a, 1:blocksize^2) = f;
                A(a, end-1) = sondos(idx).position(1);
                A(a, end) = sondos(idx).position(2);
            end
        else
            A = zeros(numel(G{n0}), blocksize^2 + 2);
            for a = 1:numel(G{n0})
                idx = G{n0}(a);
                [f, ~] = featureExtraction(sondos(idx).block);
                A(a, 1:blocksize^2) = f;
                A(a, end-1) = sondos(idx).position(1);
                A(a, end) = sondos(idx).position(2);
            end
        end

        Asorted = sortrows(A, 1:9);

        for i = 1:size(Asorted, 1) - 1
            similar = abs(Asorted(i + 1, 1:9) - Asorted(i, 1:9)) < s_threshold;
            if all(similar)
                x1 = Asorted(i, end-1);
                x2 = Asorted(i + 1, end-1);
                y1 = Asorted(i, end);
                y2 = Asorted(i + 1, end);
                D = sqrt((x1 - x2)^2 + (y1 - y2)^2);
                if D > Nd
                    detectedDuplicates(G{n0}(i)) = true;
                    detectedDuplicates(G{n0}(i + 1)) = true;
                    rectangle('Position', [y1, x1, blocksize, blocksize], 'EdgeColor', 'r');
                    rectangle('Position', [y2, x2, blocksize, blocksize], 'EdgeColor', 'r');
                end
            end
        end
    end

    % Display original and duplicate markers
    figure;
    imshow(im), title('Original and Duplicate Markers');
    hold on;

    originalIndices = find(~detectedDuplicates);
    duplicateIndices = find(detectedDuplicates);

    % Extract coordinates for original markers
    originalX = zeros(1, numel(originalIndices));
    originalY = zeros(1, numel(originalIndices));
    for i = 1:numel(originalIndices)
        idx = originalIndices(i);
        originalX(i) = sondos(idx).position(2) + blocksize / 2;
        originalY(i) = sondos(idx).position(1) + blocksize / 2;
    end

    % Extract coordinates for duplicate markers
    duplicateX = zeros(1, numel(duplicateIndices));
    duplicateY = zeros(1, numel(duplicateIndices));
    for i = 1:numel(duplicateIndices)
        idx = duplicateIndices(i);
        duplicateX(i) = sondos(idx).position(2) + blocksize / 2;
        duplicateY(i) = sondos(idx).position(1) + blocksize / 2;
    end

    % Plot original markers in green
    scatter(originalX, originalY, 50, 'g', 'filled');

    % Plot duplicate markers in red
    scatter(duplicateX, duplicateY, 50, 'r', 'filled');

    hold off;

    % Display execution time
    elapsedTime = toc;
    disp(['Execution Time: ', num2str(elapsedTime), ' seconds']);
end

function [feature, vector] = featureExtraction(block)
    dctCoefficients = dct2(block);
    feature = dctCoefficients(:)';
    vector = feature;
end
