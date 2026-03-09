%% Pipeline Step 1: Extract Color Chart Data & Create Masks
% GOAL: Select chart corners -> Extract RGBs (Correctly Ordered) -> Save
% Data. (Similar to Exercise 2)
% INPUT: Linearized Raw Image (PNG).
% OUTPUT: 'chart_data.mat' containing sorted RGBs, Centers, and Binary Mask.

% REMEMBER TO CHANGE "chart_dataX.mat" at the end, replacing X for the # of
% the chart you are crearting a mask for. Go from closest to furthest.

clc; clear; close all;

% --- 1. SETTINGS ---
% Update this to match your current image file
imageFilename = 'IMG_2253.png'; 

% Chart Definition (Standard DGK / Macbeth)
numRows = 3;   
numCols = 6;   
boxSize = 10;  % Pixels to average around center

% --- 2. LOAD IMAGE ---
if ~isfile(imageFilename)
    error('File not found: %s', imageFilename);
end

rgbImg = imread(imageFilename);
rgbDouble = double(rgbImg); % Work in Double precision

% Display
f = figure(1); 
imshow(rgbImg);
title('Step 1: Locate Color Chart', 'FontSize', 14);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0.1 0.8 0.8]); 

% --- 3. CORNER SELECTION (Zoom-Friendly) ---
disp('--- INTERACTIVE MODE ---');
disp('Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left');

cornerNames = {'Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'};
cornerPoints = zeros(4, 2);

for k = 1:4
    % A. Navigation Phase
    title(sprintf('Zoom to %s corner, then press ENTER to lock view', cornerNames{k}), 'Color', 'k');
    zoom on; 
    pause; % Wait for Enter
    
    % B. Clicking Phase
    title(sprintf('Click %s corner NOW', cornerNames{k}), 'Color', 'r', 'FontWeight', 'bold');
    zoom off;
    [xk, yk] = ginput(1);
    cornerPoints(k, :) = [xk, yk];
    
    % Visual Feedback
    hold on;
    plot(xk, yk, 'ro', 'LineWidth', 2, 'MarkerSize', 10);
    text(xk+15, yk, num2str(k), 'Color', 'y', 'FontSize', 14, 'FontWeight', 'bold');
    drawnow;
end
close(1);

% --- 4. GRID GENERATION & REORDERING ---
% Define ideal flat chart (Arbitrary units)
w_patch = 100; h_patch = 100;
w_total = numCols * w_patch;
h_total = numRows * h_patch;

% Ideal points (TL, TR, BR, BL)
fixedPoints  = [0, 0;  w_total, 0;  w_total, h_total;  0, h_total];
movingPoints = cornerPoints; 

% Calculate Perspective Transform
tform = fitgeotrans(fixedPoints, movingPoints, 'projective');

% Create Grid of Centers
% NOTE: meshgrid creates column-major arrays. We transpose (') immediately 
% to force Reading Order (Row 1 Left->Right, then Row 2...).
[X_cols, Y_rows] = meshgrid( ...
    (0:numCols-1) * w_patch + w_patch/2, ...
    (0:numRows-1) * h_patch + h_patch/2);

X_flat = X_cols'; % Transpose to fix order
Y_flat = Y_rows'; % Transpose to fix order

idealCenters = [X_flat(:), Y_flat(:)];

% Transform centers back to Image Space
imageCenters = transformPointsForward(tform, idealCenters);

% --- 5. EXTRACT RGB & CREATE BINARY MASK ---
numPatches = size(imageCenters, 1);
rgb_values = zeros(numPatches, 3);
[imgH, imgW, ~] = size(rgbDouble);

% Create a logical mask for the WHOLE chart (useful for excluding it later)
chartMask = poly2mask(cornerPoints(:,1), cornerPoints(:,2), imgH, imgW);

fprintf('Extracting %d patches... ', numPatches);

for k = 1:numPatches
    cx = round(imageCenters(k, 1));
    cy = round(imageCenters(k, 2));
    
    % Safe Bounds
    r_start = max(1, cy - floor(boxSize/2));
    r_end   = min(imgH, cy + floor(boxSize/2));
    c_start = max(1, cx - floor(boxSize/2));
    c_end   = min(imgW, cx + floor(boxSize/2));
    
    % Extract Mean RGB
    patchPixels = rgbDouble(r_start:r_end, c_start:c_end, :);
    rgb_values(k, :) = squeeze(mean(mean(patchPixels, 1), 2));
end
fprintf('Done.\n');

% --- 6. VALIDATION PLOT ---
figure(2);
imshow(rgbImg); hold on;
title('Validation: Numbering should read like a book (white BR, black BL) (L->R, Top->Down)');
plot(imageCenters(:,1), imageCenters(:,2), 'g+', 'MarkerSize', 10);
visboundaries(chartMask, 'Color', 'b', 'LineWidth', 1);

for k = 1:numPatches
    text(imageCenters(k,1), imageCenters(k,2)-15, num2str(k), ...
        'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
hold off;

% --- 7. SAVE PIPELINE DATA ---
% We save 'chartMask' too—you'll need it for Backscatter Estimation!
save('chart_data1.mat', 'rgb_values', 'imageCenters', 'cornerPoints', 'chartMask', 'boxSize');
disp('Data saved to chart_data1.mat');

%% Pipeline Step 2: Depth Extraction & Backscatter Estimation (Grayscale Regression Method)
% GOAL: Combine depth reading (from TIF) and Backscatter calc (Regression) into one step.
% INPUT: 'chart_dataX.mat' (RGBs) + Depth Map (TIF).
% OUTPUT: 'step2_results.mat' containing z_depths and Bc_values.

clc; clear; close all;

% --- 1. SETTINGS ---
depthFilename = 'IMG_2253.tif'; 
numCharts = 1; % How many charts did you click in Step 1? (e.g. 1 or 5)

% GRAYSCALE CONFIGURATION
% Check your chart! 
% Indices 13=Black ... 18=White (or vice versa).
gray_indices = 13:18; 

% MANUFACTURER REFLECTANCE VALUES (Y) from DGK csv
% These are the known x-axis values for regression (Ground Truth).
% Ensure these match the order of your 'gray_indices'.
% Assuming Order: Darkest -> Brightest
manufacturer_Y = [0.0322, 0.0790, 0.1245, 0.1857, 0.2429, 0.6987];

% --- 2. LOAD DEPTH MAP ---
if ~isfile(depthFilename)
    error('Depth map not found: %s', depthFilename);
end
depthImg = imread(depthFilename);
depthDouble = double(depthImg);

% --- 3. INITIALIZE STORAGE ---
all_depths = zeros(1, numCharts);
all_Bc     = zeros(numCharts, 3); % Columns: R, G, B

figure(1); clf;
% Changed to 'flow' so the tiles automatically fill and center
tiledlayout('flow', 'TileSpacing', 'compact'); 
sgtitle('Step 2: Backscatter Estimation (Regression)', 'FontSize', 14);

fprintf('--- Processing %d Charts ---\n', numCharts);

% --- 4. MAIN PROCESSING LOOP ---
for k = 1:numCharts
    fname = sprintf('chart_data%d.mat', k);
    
    if isfile(fname)
        % Load RGB Data
        data = load(fname);
        
        % === PART A: EXTRACT DEPTH ===
        % Use the centers from Step 1 to find depth in the TIF
        centers = data.imageCenters;
        boxSize = data.boxSize;
        numPatches = size(centers, 1);
        patch_depths = zeros(numPatches, 1);
        
        for p = 1:numPatches
            cx = round(centers(p, 1));
            cy = round(centers(p, 2));
            
            % Boundary checks for Depth Map
            r_start = max(1, cy - floor(boxSize/2));
            r_end   = min(size(depthDouble,1), cy + floor(boxSize/2));
            c_start = max(1, cx - floor(boxSize/2));
            c_end   = min(size(depthDouble,2), cx + floor(boxSize/2));
            
            % Mean depth of this patch
            patch_depths(p) = mean(mean(depthDouble(r_start:r_end, c_start:c_end)));
        end
        
        % Average all patches to get Chart Depth
        current_z = mean(patch_depths);
        all_depths(k) = current_z;
        
        % === PART B: ESTIMATE BACKSCATTER (Regression) ===
        % Extract Grayscale Patches
        rgb_gray = data.rgb_values(gray_indices, :);
        
        % Normalize if needed (Handling 8-bit, 16-bit, or 0-1)
        if max(rgb_gray(:)) > 255
            rgb_gray = double(rgb_gray) / 65535.0; 
        elseif max(rgb_gray(:)) > 1
            rgb_gray = double(rgb_gray) / 255.0;   
        end
        
        % Loop Channels (R, G, B)
        colors = ['r', 'g', 'b'];
        
        % Plotting setup for this chart
        nexttile; hold on;
        title(sprintf('Chart %d (z=%.2fm)', k, current_z));
        
        for c = 1:3
            y_vals = rgb_gray(:, c); % Measured Intensity
            x_vals = manufacturer_Y'; % Known Reflectance
            
            % Linear Regression: y = mx + c (c = Backscatter)
            p = polyfit(x_vals, y_vals, 1);
            
            intercept = p(2);
            
            % Logic Check: Backscatter cannot be negative.
            if intercept < 0
                intercept = 0; 
            end
            
            all_Bc(k, c) = intercept;
            
            % Plot Fit
            plot(x_vals, y_vals, 'o', 'Color', colors(c), 'MarkerFaceColor', colors(c));
            plot(x_vals, polyval(p, x_vals), '-', 'Color', colors(c));
        end
        grid on; xlabel('True Y'); ylabel('Measured RGB');
        
        fprintf('Chart %d: Depth=%.2fm | Bc (RGB): %.3f, %.3f, %.3f\n', ...
            k, current_z, all_Bc(k,1), all_Bc(k,2), all_Bc(k,3));
        
    else
        fprintf('Skipping %s (Not found)\n', fname);
    end
end

% --- 5. SAVE CONSOLIDATED RESULTS ---
% This file now contains EVERYTHING you need for the Attenuation Step
save('Backscatter_Depths.mat', 'all_depths', 'all_Bc');

disp(' ');
disp('Step 2 Complete. Results saved to "Backscatter_Depths.mat".');
disp('You are now ready to calculate the Attenuation Coefficient (Beta).');

%% Pipeline Step 3: Backscatter Modeling & Subtraction

% GOAL: Fit a physical model to the Backscatter data (from Step 2) 
%       and subtract it from the original image pixel-by-pixel.
% INPUT: 'Backscatter_Depths.mat', 'DSC01452.tif' (Depth), 'DSC01452.png' (RGB)
% OUTPUT: 'step3_data.mat' (containing the Backscatter-Free Image)

clc; clear; close all;

% --- 1. LOAD DATA ---
if ~isfile('Backscatter_Depths.mat')
    error('Missing Backscatter_Depths.mat. Please run Step 2.');
end
load('Backscatter_Depths.mat', 'all_depths', 'all_Bc');

rgbFilename = 'IMG_2253.png'; % Ensure this matches your Step 1 file
depthFilename = 'IMG_2253.tif';

fprintf('Loading Images...\n');
rgbImg = im2double(imread(rgbFilename)); % Load as 0-1 double
depthImg = double(imread(depthFilename));

% --- 2. FIT MODEL (Saturating Exponential) ---
% Model: B(z) = B_inf * (1 - exp(-beta * z))
% We need to find B_inf (Saturation) and beta (Rate) for R, G, B.

% Define the function to optimize: Sum of Squared Errors
% p(1) = B_inf, p(2) = beta
model_fun = @(p, z) p(1) * (1 - exp(-p(2) * z));
cost_fun = @(p, z, y) sum((model_fun(p, z) - y).^2);

fitted_params = zeros(3, 2); % 3 Channels x 2 Params
colors = ['r', 'g', 'b'];
channel_names = {'Red', 'Green', 'Blue'};

figure(1); clf;
tiledlayout(1, 3, 'TileSpacing', 'compact');
sgtitle('Backscatter Model Fit: B(z) = B_{\infty}(1 - e^{-\beta z})', 'FontSize', 14);

fprintf('--- Fitting Backscatter Model ---\n');

for c = 1:3
    z_data = all_depths;      % X-axis: Depth
    b_data = all_Bc(:, c)';   % Y-axis: Backscatter Intensity
    
    % Initial Guess: [B_inf, beta]
    % Guess B_inf is slightly higher than max observed B
    % Guess beta is small (e.g., 0.1)
    p0 = [max(b_data)*1.2, 0.1];
    
    % Optimize (fminsearch is standard MATLAB, no toolbox needed)
    p_opt = fminsearch(@(p) cost_fun(p, z_data, b_data), p0);
    
    % Constraint: Params must be positive
    p_opt = abs(p_opt); 
    fitted_params(c, :) = p_opt;
    
    % --- PLOTTING FIT ---
    nexttile; hold on;
    plot(z_data, b_data, 'o', 'MarkerFaceColor', colors(c), 'MarkerEdgeColor', 'k');
    
    z_smooth = linspace(0, max(depthImg(:)), 100);
    b_smooth = model_fun(p_opt, z_smooth);
    plot(z_smooth, b_smooth, 'k--', 'LineWidth', 1.5);
    
    title(channel_names{c});
    xlabel('Depth (m)'); ylabel('Backscatter Intensity');
    grid on;
    legend('Data', sprintf('Fit: \\beta=%.3f', p_opt(2)), 'Location', 'se');
    
    fprintf('%s Channel: B_inf = %.4f, beta = %.4f\n', ...
        channel_names{c}, p_opt(1), p_opt(2));
end

% --- 3. GENERATE BACKSCATTER MAP ---
fprintf('Generating Synthetic Backscatter Map...\n');

[H, W, ~] = size(rgbImg);
Bc_Map = zeros(H, W, 3);

for c = 1:3
    B_inf = fitted_params(c, 1);
    beta  = fitted_params(c, 2);
    
    % Apply model to every pixel in the Depth Map
    Bc_Map(:, :, c) = B_inf * (1 - exp(-beta * depthImg));
end

% --- 4. SUBTRACT BACKSCATTER ---
fprintf('Subtracting Backscatter...\n');

% J_direct = J_measured - J_backscatter
J_BS_Removed = rgbImg - Bc_Map;

% CLAMPING: We cannot have negative light. 
% (Negative values mean our model predicted more fog than existed).
J_BS_Removed(J_BS_Removed < 0) = 0;

% --- 5. VISUALIZATION & SAVE ---
% Enhance visibility for display ONLY using imadjust. 
display_orig = imadjust(rgbImg, stretchlim(rgbImg), []);
display_img  = imadjust(J_BS_Removed, stretchlim(J_BS_Removed), []);

figure(2); clf;
subplot(1,3,1); imshow(display_orig); title('1. Original (Display Enhanced)');
subplot(1,3,2); imshow(Bc_Map);       title('2. Calculated Backscatter Map');
subplot(1,3,3); imshow(display_img);  title('3. BS Removed (Display Enhanced)');

save('step3_data.mat', 'J_BS_Removed', 'Bc_Map', 'fitted_params');
disp(' ');
disp('Step 3 Complete.');
disp('The "Haze" has been removed based on the depth of every pixel.');
disp('Result saved to "step3_data.mat".');

%% Pipeline Step 4: Wideband Attenuation Estimation (Beta) - Theoretical Method
% GOAL: Calculate color attenuation (Beta) using a single color chart.
% METHOD: Uses Eq 9 with W_c extracted from the 24% gray patch (Patch 17).
%         Instead of a land image, it uses the manufacturer's known reflectance.
%         Beta = -ln(Dc_patch17 / Theoretical_patch17) / z
% INPUT: 'chart_data1.mat', 'Backscatter_Depths.mat'
% OUTPUT: 'step4_data.mat' (Beta values)

clc; clear; close all;

% --- 1. CONFIGURATION ---
patch24Idx = 17; 

% Theoretical reflectance of the DGK 24% patch
% Since it is neutral gray, R = G = B = Y
theoretical_24 = [0.2429, 0.2429, 0.2429]; 

% --- 2. LOAD DATA ---
if ~isfile('Backscatter_Depths.mat') || ~isfile('chart_data1.mat')
    error('Missing Backscatter_Depths.mat or chart_data1.mat.');
end
load('Backscatter_Depths.mat', 'all_depths', 'all_Bc'); 
underwater_data = load('chart_data1.mat');

% --- 3. EXTRACT 24% GRAY PATCH DATA ---
fprintf('--- Calculating Attenuation (Beta) from 24%% Gray Patch ---\n');

raw_water_24 = underwater_data.rgb_values(patch24Idx, :);
Bc = all_Bc(1, :); 
z  = all_depths(1); 

water_val = double(raw_water_24);
if max(underwater_data.rgb_values(:)) > 255
    water_val = water_val / 65535.0;
elseif max(underwater_data.rgb_values(:)) > 1
    water_val = water_val / 255.0;
end

% --- 4. CALCULATE ATTENUATION ---
Dc = water_val - Bc;
Dc(Dc <= 0) = 1e-4; % Safety clamp

% Calculate W_c using the theoretical target instead of a land image
Wc = Dc ./ theoretical_24;

% Calculate Beta
Beta = -log(Wc) / z;

% --- 5. PLOTTING ---
colors = ['r', 'g', 'b'];
channel_names = {'Red', 'Green', 'Blue'};

figure(1); clf;
t = tiledlayout(1, 3, 'TileSpacing', 'compact');
sgtitle('Single-Chart Attenuation (\beta) vs Theoretical Target', 'FontSize', 14);

for c = 1:3
    nexttile; hold on;
    plot([0, z], [theoretical_24(c), Dc(c)], 'o-', 'Color', colors(c), 'MarkerFaceColor', colors(c), 'LineWidth', 2);
    
    title(channel_names{c});
    xlabel('Depth (m)'); ylabel('Direct Signal (0-1)');
    grid on; xlim([-0.2, z + 0.5]); ylim([0, 0.3]);
    
    text(z/2, theoretical_24(c)/2, sprintf('\\beta = %.3f', Beta(c)), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', colors(c), 'HorizontalAlignment', 'center');
    
    fprintf('%s Channel: Target=%.3f, Water(Dc)=%.3f -> Wc=%.3f, Beta = %.4f m^-1\n', ...
        channel_names{c}, theoretical_24(c), Dc(c), Wc(c), Beta(c));
end

save('step4_data.mat', 'Beta');
disp('Step 4 Complete. Beta values saved.');


%% Pipeline Step 5: Attenuation Correction & Radiometric Calibration (Theoretical)
% GOAL: Restore color using Beta, then calibrate to the Manufacturer's Y values.
% INPUT: Step 3 (No Fog), Step 4 (Beta), Chart 1 Data (Water).
% OUTPUT: 'Final_Calibrated_Image.mat'

clc; clear; close all;

% --- 1. SETTINGS ---
depthFilename = 'IMG_2253.tif'; 
gray_indices = 13:18; % Black -> White

% Manufacturer Y values (Target for radiometric calibration)
manufacturer_Y = [0.0322, 0.0790, 0.1245, 0.1857, 0.2429, 0.6987];

% --- 2. LOAD DATA ---
if ~isfile('step3_data.mat') || ~isfile('step4_data.mat') || ~isfile('chart_data1.mat')
    error('Missing pipeline data.');
end
load('step3_data.mat', 'J_BS_Removed');     
load('step4_data.mat', 'Beta');             
load('Backscatter_Depths.mat', 'all_Bc', 'all_depths'); 
chart1_data = load('chart_data1.mat');      

depthImg = double(imread(depthFilename));

% --- 3. APPLY ATTENUATION CORRECTION (Unscaled) ---
[H, W, C] = size(J_BS_Removed);
J_Recovered = zeros(H, W, C);

for c = 1:3
    gain_map = exp(Beta(c) * depthImg);
    J_Recovered(:, :, c) = J_BS_Removed(:, :, c) .* gain_map;
end

% --- 4. CALCULATE SCALING FACTORS (Using Theoretical Data) ---
raw_water_grays = chart1_data.rgb_values(gray_indices, :);
if max(raw_water_grays(:)) > 1
    raw_water_grays = double(raw_water_grays) / 65535.0;
end

z_c1 = all_depths(1);
Bc_c1 = all_Bc(1, :);
corrected_water_grays = zeros(size(raw_water_grays));

for c = 1:3
    val_no_fog = raw_water_grays(:, c) - Bc_c1(c);
    val_no_fog(val_no_fog < 0) = 0; 
    gain_factor = exp(Beta(c) * z_c1);
    corrected_water_grays(:, c) = val_no_fog * gain_factor;
end

Scaling_Factors = zeros(1, 3);
colors = ['r', 'g', 'b'];
channels = 'RGB'; 

figure(1); clf;
tiledlayout(1, 3, 'TileSpacing', 'compact');
sgtitle('Radiometric Calibration: Water vs. Theoretical Target');

for c = 1:3
    x_val = corrected_water_grays(:, c); 
    y_val = manufacturer_Y'; % Theoretical Target
    
    K = x_val \ y_val; 
    Scaling_Factors(c) = K;
    
    nexttile; hold on;
    plot(x_val, y_val, 'o', 'Color', colors(c), 'MarkerFaceColor', colors(c));
    plot(x_val, x_val * K, 'k--');
    xlabel('Recovered Underwater'); ylabel('Theoretical Target');
    title(sprintf('%s Gain: %.2f', channels(c), K));
    grid on;
end

% --- 5. APPLY & SAVE ---
J_Final = zeros(size(J_Recovered));
for c = 1:3
    J_Final(:, :, c) = J_Recovered(:, :, c) * Scaling_Factors(c);
end
J_Final(J_Final < 0) = 0; 
J_Display = J_Final;
J_Display(J_Display > 1) = 1;

%display_BS_Removed = imadjust(J_BS_Removed, stretchlim(J_BS_Removed), []);
%J_Display_Enhanced = imadjust(J_Display, stretchlim(J_Display), []);

figure(2); clf;
subplot(1,2,1); imshow(J_BS_Removed); title('Fog Removed (Non-Enhanced)');
subplot(1,2,2); imshow(J_Display); title('Restored Colors (Non-Enhanced)');

save('Final_Scientific_Image.mat', 'J_Final', 'Scaling_Factors');
imwrite(J_Display, 'Restored_Reflectance_Theoretical.png');

%% Pipeline Step 6: Universal XYZ Transformation (Water-Corrected -> Theoretical Values)
% Note: standard Observer swapped for camera's spectral signature 
% GOAL: Calculate 3x3 Matrix mapping the Water-Corrected RGB directly to CIE XYZ (D65).
%       This version calculates the target XYZ mathematically from first principles 
%       using the actual spectral reflectances, Canon Spectral signature, and illuminant.
% INPUT: 'DGKcolorchart_reflectances.csv', 'Canon-PowerShot-G7-X-Mark-III.csv', 'illuminant-D65.csv'
%        Plus Pipeline Data (chart_data1.mat, Backscatter_Depths.mat, step4_data.mat, Final_Scientific_Image.mat)
% OUTPUT: 'Final_XYZ_Image.mat', 'CCM_Matrix.mat'

clc; clear; close all;

% --- 1. LOAD & CALCULATE GROUND TRUTH XYZ FROM CSV FILES ---
fprintf('--- Calculating Target XYZ from First Principles ---\n');

% 1.1 Load Data
% Canon Spectral Signature (No headers in the CSV)
obs_data = readmatrix('Canon-PowerShot-G7-X-Mark-III.csv');
wave_obs = obs_data(:, 1);
x_bar = obs_data(:, 2);
y_bar = obs_data(:, 3);
z_bar = obs_data(:, 4);

% D65 Illuminant (Has headers, ignore NaN rows read by readmatrix)
ill_data = readmatrix('illuminant-D65.csv');
ill_data(any(isnan(ill_data), 2), :) = []; 
wave_ill = ill_data(:, 1);
I_ill = ill_data(:, 2);

% DGK Color Chart Reflectances (Has headers, ignore NaN rows)
ref_data = readmatrix('DGKcolorchart_reflectances.csv');
ref_data(any(isnan(ref_data), 2), :) = [];
wave_ref = ref_data(:, 1);
reflectances = ref_data(:, 2:19); % 18 Patches

% 1.2 Interpolate to Common Wavelength Range (400nm to 700nm at 5nm intervals)
lambda = (400:5:700)';
x_bar_interp = interp1(wave_obs, x_bar, lambda, 'spline');
y_bar_interp = interp1(wave_obs, y_bar, lambda, 'spline');
z_bar_interp = interp1(wave_obs, z_bar, lambda, 'spline');

I_interp = interp1(wave_ill, I_ill, lambda, 'spline');

% 1.3 Calculate Normalization Factor (N)
% N = sum(I(lambda) * y_bar(lambda) * d_lambda)
N_factor = sum(I_interp .* y_bar_interp);

% 1.4 Calculate XYZ for each of the 18 Patches
Ref_XYZ_List = zeros(18, 3);
for p = 1:18
    R_interp = interp1(wave_ref, reflectances(:, p), lambda, 'spline');
    
    % Discrete integration
    X = sum(R_interp .* I_interp .* x_bar_interp) / N_factor;
    Y = sum(R_interp .* I_interp .* y_bar_interp) / N_factor;
    Z = sum(R_interp .* I_interp .* z_bar_interp) / N_factor;
    
    Ref_XYZ_List(p, :) = [X, Y, Z];
end

% --- 2. MAP UNDERWATER PATCHES TO CSV PATCHES ---
% Image layout: 1-12 Colors, 13-18 Grayscale (Black -> White)
% CSV layout: 1-6 Grayscale (White -> Black), 7-18 Colors
map = zeros(1, 18);
map(13:18) = 6:-1:1; 
map(1:12) = 7:18; 

Target_XYZ = Ref_XYZ_List(map, :);

% --- 3. LOAD PIPELINE DATA ---
if ~isfile('chart_data1.mat') || ~isfile('Final_Scientific_Image.mat')
    error('Missing required pipeline data. Make sure Step 5 ran successfully.');
end

load('chart_data1.mat');            % Raw RGB of Underwater Chart 1
load('Backscatter_Depths.mat');     % Bc and Depth
load('step4_data.mat', 'Beta');     % Attenuation Coeffs
load('Final_Scientific_Image.mat'); % Scaling Factors & J_Final

% --- 4. RECONSTRUCT WATER-CORRECTED RGB FOR CHART 1 ---
numPatches = size(rgb_values, 1);
RGB_Camera_Reflectance = zeros(numPatches, 3);

z1 = all_depths(1);    
Bc1 = all_Bc(1, :);    
rgb_raw = double(rgb_values);
if max(rgb_raw(:)) > 1
    rgb_raw = rgb_raw / 65535.0; 
end

fprintf('--- Processing Underwater Chart Patches ---\n');
for k = 1:numPatches
    % A. Remove Backscatter
    patch_no_fog = max(0, rgb_raw(k,:) - Bc1);
    
    % B. Attenuation Gain
    patch_att = patch_no_fog .* exp(Beta .* z1);
    
    % C. Radiometric Scaling (The K factors from Step 5)
    RGB_Camera_Reflectance(k, :) = patch_att .* Scaling_Factors;
end

% --- 5. CALCULATE CCM (Corrected Water RGB -> Universal XYZ) ---
fprintf('--- Calculating Color Correction Matrix (3x3) ---\n');

% Solve M such that: Water_RGB * M = Target_XYZ
CCM = RGB_Camera_Reflectance \ Target_XYZ; 

disp('Calculated Matrix M:');
disp(CCM);

% --- 6. APPLY MATRIX TO WATER-CORRECTED IMAGE ---
fprintf('--- Standardizing Final Image to XYZ Space ---\n');

[H, W, ~] = size(J_Final);
img_flat = reshape(J_Final, [], 3);

% Apply Matrix: XYZ = RGB * CCM
xyz_flat = img_flat * CCM;

% Reshape back to image
J_XYZ = reshape(xyz_flat, H, W, 3);

% --- 7. SAVE DATA ---
save('Final_XYZ_Image.mat', 'J_XYZ', 'CCM');

% --- 8. VALIDATION PLOT ---
figure(1); clf;
subplot(1, 2, 1); hold on;

% Plot Target (Black Circles)
plot3(Target_XYZ(:,1), Target_XYZ(:,2), Target_XYZ(:,3), 'wo', 'MarkerSize', 8, 'LineWidth', 2);

% Plot Your Corrected Result (Red Dots)
Model_XYZ = RGB_Camera_Reflectance * CCM;
plot3(Model_XYZ(:,1), Model_XYZ(:,2), Model_XYZ(:,3), 'r.', 'MarkerSize', 15);

grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Theoretical Values Alignment (Black=True, Red=Water Corrected)');
view(45, 45);

% Simple sRGB Preview (Enhanced for visibility)
subplot(1, 2, 2);
rgb_preview = xyz2rgb(J_XYZ);
rgb_preview(rgb_preview < 0) = 0; % Clamp negatives before display
display_preview = imadjust(rgb_preview, stretchlim(rgb_preview), []);
imshow(display_preview); 
title('Final Scientific Image (sRGB Preview)');

disp('Step 6 Complete.');

%% Pipeline Step 7: Color Consistency Error Analysis (Theoretical XYZ)
% GOAL: Calculate the Angular Error (in degrees) between the 
%       Final Corrected Underwater XYZ and the True CSV XYZ standard.
% INPUT: Pipeline Data (Steps 1-6) + CSV files for ground truth.
% OUTPUT: Error Plots and Statistics.

clc; clear; close all;

% --- 1. CALCULATE GROUND TRUTH XYZ FROM CSV FILES ---
fprintf('--- 1. Calculating Target XYZ from First Principles ---\n');

% Load Data
obs_data = readmatrix('Canon-PowerShot-G7-X-Mark-III.csv');
wave_obs = obs_data(:, 1);
x_bar = obs_data(:, 2); y_bar = obs_data(:, 3); z_bar = obs_data(:, 4);

ill_data = readmatrix('illuminant-D65.csv');
ill_data(any(isnan(ill_data), 2), :) = []; 
wave_ill = ill_data(:, 1); I_ill = ill_data(:, 2);

ref_data = readmatrix('DGKcolorchart_reflectances.csv');
ref_data(any(isnan(ref_data), 2), :) = [];
wave_ref = ref_data(:, 1); reflectances = ref_data(:, 2:19);

% Interpolate to 400nm-700nm at 5nm intervals
lambda = (400:5:700)';
x_bar_interp = interp1(wave_obs, x_bar, lambda, 'spline');
y_bar_interp = interp1(wave_obs, y_bar, lambda, 'spline');
z_bar_interp = interp1(wave_obs, z_bar, lambda, 'spline');
I_interp = interp1(wave_ill, I_ill, lambda, 'spline');

% Normalize and Integrate
N_factor = sum(I_interp .* y_bar_interp);
Ref_XYZ_List = zeros(18, 3);
for p = 1:18
    R_interp = interp1(wave_ref, reflectances(:, p), lambda, 'spline');
    X = sum(R_interp .* I_interp .* x_bar_interp) / N_factor;
    Y = sum(R_interp .* I_interp .* y_bar_interp) / N_factor;
    Z = sum(R_interp .* I_interp .* z_bar_interp) / N_factor;
    Ref_XYZ_List(p, :) = [X, Y, Z];
end

% Map patches to match your clicking order (13=Black, 18=White)
map = zeros(1, 18);
map(13:18) = 6:-1:1; 
map(1:12) = 7:18;
Target_XYZ = Ref_XYZ_List(map, :);

% --- 2. LOAD & RECONSTRUCT MEASURED XYZ ---
fprintf('--- 2. Reconstructing Pipeline Data ---\n');

if ~isfile('Final_XYZ_Image.mat')
    error('Please run Step 6 first.');
end
load('chart_data1.mat');            % Raw RGB Underwater
load('Backscatter_Depths.mat');     % Bc, Depth
load('step4_data.mat', 'Beta');     % Attenuation
load('Final_Scientific_Image.mat', 'Scaling_Factors'); % Radiometric K
load('Final_XYZ_Image.mat', 'CCM'); % Matrix to XYZ

z1 = all_depths(1);
Bc1 = all_Bc(1, :);
rgb_raw = double(rgb_values);
if max(rgb_raw(:)) > 1
    rgb_raw = rgb_raw / 65535.0; 
end

numPatches = size(rgb_raw, 1);
Measured_XYZ = zeros(numPatches, 3);

for k = 1:numPatches
    % A. Water Correction (Remove Fog)
    patch_no_fog = max(0, rgb_raw(k,:) - Bc1);
    
    % B. Attenuation Gain
    patch_att = patch_no_fog .* exp(Beta .* z1);
    
    % C. Radiometric Calibration
    patch_ref = patch_att .* Scaling_Factors;
    
    % D. XYZ Transformation
    Measured_XYZ(k, :) = patch_ref * CCM;
end

% --- 3. CALCULATE ANGULAR ERROR ---
errors_deg = zeros(numPatches, 1);

fprintf('\n--- 3. Error Analysis (Degrees: Water vs Theoretical Standard) ---\n');
fprintf('%5s | %10s | %10s | %s\n', 'Patch', 'True Norm', 'Meas Norm', 'Error(Deg)');

for k = 1:numPatches
    v_true = Target_XYZ(k, :);
    v_meas = Measured_XYZ(k, :);
    
    n_true = norm(v_true);
    n_meas = norm(v_meas);
    
    if n_true > 0 && n_meas > 0
        % Dot Product Formula: a . b = |a||b| cos(theta)
        cosine_val = dot(v_true, v_meas) / (n_true * n_meas);
        cosine_val = max(min(cosine_val, 1.0), -1.0); % Clamp
        errors_deg(k) = rad2deg(acos(cosine_val));
    else
        errors_deg(k) = NaN; 
    end
    
    fprintf('%5d | %10.4f | %10.4f | %.2f\n', k, n_true, n_meas, errors_deg(k));
end

% --- 4. STATISTICS & PLOTTING ---
mean_err = mean(errors_deg, 'omitnan');
fprintf('\nGlobal Mean Angular Error: %.2f degrees\n', mean_err);

figure(1); clf;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2 0.2 0.6 0.5]);

b = bar(1:numPatches, errors_deg);
b.FaceColor = [0.2 0.6 0.8];
grid on;

xlabel('Patch Index (1-12 Color, 13-18 Gray)');
ylabel('Angular Error (Degrees)');
title(sprintf('Color Reproduction Error vs CSV Standard (Mean: %.2f^o)', mean_err));

% Add Threshold Lines
yline(5, '--g', 'Good (<5^o)', 'LineWidth', 2);
yline(10, '--y', 'Acceptable (<10^o)', 'LineWidth', 2);
yline(20, '--r', 'Poor (>20^o)', 'LineWidth', 2);

xticks(1:18);
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12', ...
             '13(Blk)','14','15','16','17','18(W)'});

disp('Analysis Complete.');


%% Pipeline Step 8: Coral Masking and Single-Point Chlorophyll Estimation
% GOAL: Isolate the coral from the background and apply a single-point 
%       proportional scaling to map chlorophyll concentration.
% INPUT: 'Final_Scientific_Image.mat'
% OUTPUT: 'Chlorophyll_Result_RAR.mat', 'Calibration_Mask.mat'

clc; clear; close all;

% --- 1. SETTINGS ---
% Single Laboratory Data Point, will swap for better calibration curve in
% the future
lab_chl_value = 2.35; % ug/cm^2

% --- 2. LOAD RADIOMETRIC DATA ---
if ~isfile('Final_Scientific_Image.mat')
    error('Missing Final_Scientific_Image.mat. Make sure Step 5 ran successfully.');
end
load('Final_Scientific_Image.mat', 'J_Final');

% Create a brightened version purely for drawing the mask easily
J_Display = J_Final;
J_Display(J_Display > 1) = 1;
display_img = imadjust(J_Display, stretchlim(J_Display), []);

% --- 3. INTERACTIVE CORAL MASKING ---
figure(1); clf;
imshow(display_img);
title('Step 8: Draw a polygon around the coral to mask it. Double-click inside to finish.', 'FontSize', 12);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1 0.1 0.8 0.8]);

disp('--- MASKING MODE ---');
disp('Click to draw vertices around the coral tissue.');
disp('Double-click the center of your polygon to lock the mask.');

% Use roipoly to let the user outline the coral
mask = roipoly();
close(1);

if isempty(mask) || ~any(mask(:))
    error('No mask was drawn. Please re-run and select the coral area.');
end

% --- 4. CALCULATE CHLOROPHYLL INDEX ---
fprintf('--- Calculating Chlorophyll Distribution ---\n');

% Extract individual channels from the calibrated reflectance image
R_ref = J_Final(:, :, 1);
G_ref = J_Final(:, :, 2);

% Calculate the Base Index (Green/Red Ratio)
% We add a tiny epsilon (1e-6) to prevent division by zero in dark shadows
Chl_Index = G_ref ./ (R_ref + 1e-6);

% --- 5. APPLY SINGLE-POINT CALIBRATION ---
% Extract the index values only for the coral tissue
valid_index_pixels = Chl_Index(mask & ~isnan(Chl_Index));
mean_index = mean(valid_index_pixels);

% Calculate the proportional scaling factor (assuming intercept = 0)
% lab_chl_value = scaling_factor * mean_index
scaling_factor = lab_chl_value / mean_index;

fprintf('Single-Point Scaling Factor Calculated: %.4f\n', scaling_factor);

% Apply the scaling factor to the entire index map
Chl_Map_Raw = Chl_Index .* scaling_factor;

% Apply the mask: set non-coral background pixels to NaN
Chl_Map = Chl_Map_Raw;
Chl_Map(~mask) = NaN;

% --- 6. CALCULATE STATISTICS ---
valid_chl_pixels = Chl_Map(~isnan(Chl_Map));
mean_chl = mean(valid_chl_pixels);
std_chl  = std(valid_chl_pixels);
min_chl  = min(valid_chl_pixels);
max_chl  = max(valid_chl_pixels);

fprintf('\n--- ROI RESULTS ---\n');
fprintf('Mean Chlorophyll: %.4f ug/cm^2 (Anchored to Lab Value)\n', mean_chl);
fprintf('Std Dev:          %.4f\n', std_chl);
fprintf('Min:              %.4f\n', min_chl);
fprintf('Max:              %.4f\n', max_chl);

% --- 7. VISUALIZE HEATMAP ---
figure(2); clf;
imagesc(Chl_Map);
colormap('jet');
c = colorbar;
c.Label.String = 'Chlorophyll a (\mu g / cm^2)';
axis image; axis off;
title(sprintf('Chlorophyll a Distribution (Mean: %.2f)', mean_chl), 'FontSize', 14);

% Draw a clean white boundary around the masked coral for aesthetics
hold on;
visboundaries(mask, 'Color', 'w', 'LineWidth', 2);
hold off;

% --- 8. SAVE DATA ---
save('Chlorophyll_Result_RAR.mat', 'Chl_Map', 'Chl_Index', 'scaling_factor');
save('Calibration_Mask.mat', 'mask');
disp(' ');
disp('Step 8 Complete. Heatmap and Mask saved successfully.');

%% Pipeline Step 9: Chlorophyll Data Extraction & Export
% GOAL: Extract the average chlorophyll values from the generated map,
%       visualize the data with error bars and text, and export to CSV.
% INPUT: 'Chlorophyll_Result_RAR.mat'
% OUTPUT: Bar Chart Figure, 'Coral_Chlorophyll_Averages.csv'

clc; clear; close all;

% --- 1. LOAD DATA ---
if ~isfile('Chlorophyll_Result_RAR.mat')
    error('Missing Chlorophyll_Result_RAR.mat. Please run Step 8 first.');
end
load('Chlorophyll_Result_RAR.mat', 'Chl_Map', 'scaling_factor');

% --- 2. EXTRACT VALUES ---
fprintf('--- Extracting Chlorophyll Statistics ---\n');

% Isolate only the valid coral pixels (ignore the NaN background)
valid_pixels = Chl_Map(~isnan(Chl_Map));

if isempty(valid_pixels)
    error('No valid chlorophyll pixels found in the map. Check your Step 8 mask.');
end

% Calculate primary statistics
avg_chl = mean(valid_pixels);
median_chl = median(valid_pixels);
std_chl = std(valid_pixels);
min_chl = min(valid_pixels);
max_chl = max(valid_pixels);
pixel_count = length(valid_pixels);

% Display results in the command window
fprintf('Average Chlorophyll-a: %.4f ug/cm^2\n', avg_chl);
fprintf('Median Chlorophyll-a:  %.4f ug/cm^2\n', median_chl);
fprintf('Standard Deviation:    %.4f\n', std_chl);
fprintf('Valid Pixel Count:     %d\n', pixel_count);

% --- 3. VISUALIZE STATISTICS (Bar Chart with Error Bars & Text) ---
figure(1); clf;
hold on;

% Plot the average as a bar
b = bar(1, avg_chl, 0.4, 'FaceColor', [0.2 0.6 0.5], 'EdgeColor', 'k', 'LineWidth', 1.2);

% Add the error bar representing standard deviation
errorbar(1, avg_chl, std_chl, 'k', 'LineWidth', 2, 'CapSize', 15);

% Formatting the figure for a clean, scientific look
ylabel('Chlorophyll a (\mu g / cm^2)', 'FontSize', 12, 'FontWeight', 'bold');
title('Mean Chlorophyll a Concentration', 'FontSize', 14);
xticks(1);
xticklabels({'Coral Sample'});
xlim([0.5, 1.5]);

% Dynamically set the Y-axis limit to leave room above the error bar
y_max = (avg_chl + std_chl) * 1.3;
ylim([0, y_max]); 

% --- ADD STATISTICS TEXT BOX ---
% Use a cell array {} for multi-line text to avoid \n interpreter errors
stats_cell = {
    sprintf('Average: %.4f \\mu g/cm^2', avg_chl), ...
    sprintf('Median: %.4f \\mu g/cm^2', median_chl), ...
    sprintf('Std Dev: %.4f', std_chl)
};

% Place the text box in the upper-left corner of the plot
text(0.55, y_max * 0.9, stats_cell, 'FontSize', 11, 'FontWeight', 'bold', ...
     'BackgroundColor', 'k', 'EdgeColor', 'k', 'Margin', 5);

grid on;
box on;
hold off;

% --- 4. EXPORT TO CSV ---
% Create a structured table for easy export
ResultsTable = table(avg_chl, median_chl, std_chl, min_chl, max_chl, pixel_count, ...
    'VariableNames', {'Average_ug_cm2', 'Median_ug_cm2', 'StdDev', 'Min_ug_cm2', 'Max_ug_cm2', 'Pixel_Count'});

% Write to CSV file in the current directory
csvFilename = 'Coral_Chlorophyll_Averages.csv';
writetable(ResultsTable, csvFilename);

disp(' ');
fprintf('Step 9 Complete. Data visualized and saved to "%s".\n', csvFilename);