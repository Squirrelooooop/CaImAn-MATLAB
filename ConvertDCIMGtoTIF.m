clear
close all
clc

dcimg_file = 'Z:\Callum\Data00001.dcimg';
out_file = 'Z:\Callum\Data00001.tif';

r = bfGetReader(dcimg_file);
r.setSeries(0);                               % use first series

d1 = r.getSizeY();                            % image height
d2 = r.getSizeX();                            % image width
T  = r.getSizeT();                            % number of time points
n = 2;  % take every n-th frame

% how many frames will we actually load?
T_ds = floor(T / n);   % or: numel(1:n:T)

Y = zeros(d1, d2, T_ds, 'uint16');

frame_idx = 0;
for t = 1:n:T
    frame_idx = frame_idx + 1;

    planeIndex = r.getIndex(0, 0, t-1) + 1;
    Y(:,:,frame_idx) = bfGetPlane(r, planeIndex);

    if mod(frame_idx,10) == 0 || t == T
        pct = 100 * t / T;
        fprintf('\rLoaded source frame %d / %d (kept %d, %.1f%%%% of time)', ...
                t, T, frame_idx, pct);
    end
end
fprintf('\nFinished loading (downsampled in time by %d).\n', n);

%% Temporal and Spatial Downsampling before further analysis

 [d1, d2, T] = size(Y);
% 
% sbin = 2;   % 2x2 spatial binning
% tbin = 2;   % average every 2 frames
% 
% % --- crop so dimensions are multiples of the bin factors ---
% d1_ds = floor(d1 / sbin) * sbin;
% d2_ds = floor(d2 / sbin) * sbin;
% T_ds  = floor(T  / tbin) * tbin;
% 
% Yc = Y(1:d1_ds, 1:d2_ds, 1:T_ds);   % cropped movie
% 
% % --- spatial binning (block averaging) ---
% d1_s = d1_ds / sbin;
% d2_s = d2_ds / sbin;
% 
% % reshape into [sbin, d1_s, sbin, d2_s, T_ds]
% Yc = reshape(Yc, sbin, d1_s, sbin, d2_s, T_ds);
% % permute to [d1_s, d2_s, sbin, sbin, T_ds]
% Yc = permute(Yc, [2 4 1 3 5]);
% % average over the sbin x sbin block
% Y_spatial = squeeze(mean(mean(Yc, 3), 4));  % [d1_s x d2_s x T_ds]
% 
% % --- temporal binning ---
% T_bin = T_ds / tbin;
% % reshape to [d1_s, d2_s, tbin, T_bin]
% Y_spatial = reshape(Y_spatial, d1_s, d2_s, tbin, T_bin);
% % average over the tbin frames
% Y_ds = squeeze(mean(Y_spatial, 3));         % [d1_s x d2_s x T_bin]
% 
% % keep as same class
% if ~isa(Y_ds, class(Y))
%     Y_ds = cast(Y_ds, class(Y));
% end
% 

if ~isa(Y,'single');    Y = single(Y);  end         % convert to single
d = d1*d2;                                          % total number of pixels

%% Set parameters

K = 20; % number of components (cells) to be found
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
% 
% % display centers of found components
% Cn =  correlation_image(Y);
% figure;imagesc(Cn);
%     axis equal; axis tight; hold all;
%     scatter(center(:,2),center(:,1),'mo');
%     title('Center of ROIs found from initialization algorithm');
%     drawnow;

%% update spatial components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

%% update temporal components
P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

% %% classify components
% % Make sure there is no parallel pool running
% delete(gcp('nocreate'));   % closes pool if one exists
% % This part takes the full video and and tests for each component how well
% % the footprint matches the data spatially. 
% % rval_space is the r-value for the spatial corellation
% rval_space = classify_comp_corr(Y,A,C,b,f,options);
% % Indexes those with spatial correlation over the threshold set.
% ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
%                                         % this test will keep processes
% %% further classification with cnn_classifier
% try  % matlab 2017b or later is needed
%     [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
% catch
%     ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
% end     
%                             
% %% event exceptionality
% 
% fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
% ind_exc = (fitness < options.min_fitness);
% 
% %% select components
% 
% keep = (ind_corr | ind_cnn) & ind_exc;
% 
% %% display kept and discarded components
% A_keep = A(:,keep);
% C_keep = C(keep,:);
% figure;
%     subplot(121); montage(extract_patch(A(:,keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15]);
%         title('Kept Components');
%     subplot(122); montage(extract_patch(A(:,~keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15])
%         title('Discarded Components');

%% SKIP classification section entirely

A_keep = A;
C_keep = C;
keep   = true(size(C,1),1);

%% merge found components
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);

%%
% display_merging = 1; % flag for displaying merging example
% if and(display_merging, ~isempty(merged_ROIs))
%     i = 1; %randi(length(merged_ROIs));
%     ln = length(merged_ROIs{i});
%     figure;
%         set(gcf,'Position',[300,300,(ln+2)*300,300]);
%         for j = 1:ln
%             subplot(1,ln+2,j); imagesc(reshape(A_keep(:,merged_ROIs{i}(j)),d1,d2)); 
%                 title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
%         end
%         subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
%                 title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
%         subplot(1,ln+2,ln+2);
%             plot(1:T,(diag(max(C_keep(merged_ROIs{i},:),[],2))\C_keep(merged_ROIs{i},:))'); 
%             hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
%             title('Temporal Components','fontsize',16,'fontweight','bold')
%         drawnow;
% end

%% refine estimates excluding rejected components

Pm.p = p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);

%% do some plotting

[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

figure;
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
%savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)

%% display components

plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

