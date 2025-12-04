clear
close all
clc

Tinfo = readtable('/Users/sunny/Desktop/Data/20250522_HEK_Sono3/cell_counts.csv'); 

pIn = '/Users/sunny/Desktop/Data/20250522_HEK_Sono3/'; %% input file folder
pOut = '/Users/sunny/Desktop/Data/20250522_HEK_Sono3/analysis1/'; %% the folder for output results.

if ~exist(pOut, 'dir')
    mkdir(pOut);
end

files_tif = dir(fullfile(pIn, '*.tif'));
files_tiff = dir(fullfile(pIn, '*.tiff'));    % use .tif/.tiff for 2D+time data
files = [files_tif; files_tiff];                          

for f = 1:numel(files)
    fname = fullfile(pIn, files(f).name);
    
    info = imfinfo(fname);

    T = numel(info);
    d1 = info(1).Height;
    d2 = info(1).Width;


    n = 2;  % take every n-th frame

    % how many frames will we actually load?
    idx_keep = 1:n:T;
    T_ds = numel(idx_keep);

    Y = zeros(d1, d2, T_ds, 'uint16');

    for j = 1:T_ds
        t = idx_keep(j);
        Y(:,:,j) = imread(fname, t, 'Info', info);
    end

%% Temporal and Spatial Downsampling before further analysis

    [d1, d2, T] = size(Y);

    if ~isa(Y,'single');    
        Y = single(Y);  
    end         % convert to single
    d = d1*d2;                                          % total number of pixels

%% Set parameters

    K = Tinfo.count(f);    
    tau = 10; % std of gaussian kernel (half size of neuron) 
    p = 2;

    options = CNMFSetParms(...
        'd1',d1,'d2',d2,...                         % dimensionality of the FOV
        'p',p,...                                   % order of AR dynamics
        'gSig',tau,...                              % half size of neuron
        'merge_thr',0.80,...                        % merging threshold
        'nb',2,...                                  % number of background components
        'min_SNR',3,...                             % minimum SNR threshold
        'space_thresh',0.5,...                      % space correlation threshold
        'cnn_thr',0.2);                             % threshold for CNN classifier

%% Data pre-processing

    [P,Y] = preprocess_data(Y,p); %Denoises the video and estimates the noise per pixel

%% fast initialization of spatial components using greedyROI and HALS

    [Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

%% update spatial components
    Yr = reshape(Y,d,T);
    [A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

%% update temporal components
    P.p = 0;    % set AR temporarily to zero for speed
    [C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);


%% SKIP classification section entirely

    A_keep = A;
    C_keep = C;

%% merge found components
    [Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);


%% refine estimates excluding rejected components

    Pm.p = p;    % restore AR value
    [A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
    [C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);

%% do some plotting

    [A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
    %K_m = size(C_or,1);
    [C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)
    
    Cn = correlation_image(Y);

    
    
%% Save ΔF/F + ROI numbers + centers as one CSV

    numROIs = size(C_df,1);
    roi_table = table((1:numROIs)', center(:,1), center(:,2), 'VariableNames', {'ROI','y','x'});
    for t_col = 1:size(C_df,2)
        roi_table.(sprintf('t%03d',t_col)) = C_df(:,t_col);
    end
    csvfile = fullfile(pOut, [files(f).name(1:end-4), '_df.csv']);
    writetable(roi_table, csvfile);
    fprintf('Saved ΔF/F CSV: %s\n', csvfile);    
    
   %% Save contour figure
    fig1 = figure('Visible','off');
    plot_contours(A_or, Cn, options, 1);
    contourfile = fullfile(pOut, [files(f).name(1:end-4), '_contours.png']);
    saveas(fig1, contourfile);
    close(fig1);
    fprintf('Saved contour figure: %s\n', contourfile);


  %% Save components GUI figure
    fig2 = figure('Visible','off');
    plot_components_GUI(Yr, A_or, C_or, b2, f2, Cn, options);
    guifile = fullfile(pOut, [files(f).name(1:end-4), '_components.png']);
    saveas(fig2, guifile);
    close(fig2);
    fprintf('Saved components GUI figure: %s\n', guifile);

%% Save full MAT file
    matfile = fullfile(pOut, [files(f).name(1:end-4), '_caiman.mat']);
    save(matfile, 'A2','C2','b2','f2','P2','S2','C_df','center','options','Cn','Yr');
    fprintf('Saved MAT file: %s\n', matfile);
    
    %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)


end


