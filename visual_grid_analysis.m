%% Voxel-wise visual-grid analysis on simulated data
% (1) This code simulates beta estimates with a grid-like modulation (i.e.
% a 6-fold rotational symmetry as a function of direction). (2) It then estimates
% the orientation of the grid-modulation and (3) contrasts the corresponding beta
% estimates in independent data. This procedure is cross-validated and control
% symmetries are tested in addition. Each beta estimate represents one direction.
% M.Nau, last updated - July 2021

% If you use this code or parts of it please cite the following paper:
% Nau, M., Navarro Schröder, T., Bellmund, J.L.S., Doeller, C.F.
% Hexadirectional coding of visual space in human entorhinal cortex.
% Nat Neurosci 21, 188–190 (2018). https://doi.org/10.1038/s41593-017-0050-8

%% (1) Simulate ground-truth data to illustrate pipeline

% simulation settings
n_partitions    = 4; % number of scanning runs
n_voxels        = 200; % at a voxel size of 2mm, entorhinal cortex has ~200 voxels
step_sz         = 10; % steps in which directions were sampled (in degree)
sampled_dirs    = deg2rad(0:step_sz:360-step_sz); % all sampled directions
n_dirs          = numel(sampled_dirs); % total number of directions
sim_grid_orient = randi(30,[1,n_voxels]); % simulated grid orientations

% create simulated beta estimates (4 data paritions, 200 voxels, 36 directions)
sim_betas    = arrayfun(@(cDP) cell2mat(arrayfun(@(cVoxel) ...
    sin((sampled_dirs - deg2rad(sim_grid_orient(cVoxel)))*6) + 3*randn(1,n_dirs), ...
    1:round(n_voxels), 'uni', 0)'), 1:n_partitions, 'uni', 0); % grid effect + 3std noise

%% start pipeline
symmetries  = [4:8]; % (e.g. 6 = 6-fold)
for cSym    = symmetries
    
    stats   = cell(1,n_partitions); grid_betas = cell(n_partitions,1); contrast = cell(n_partitions,1);
    for cDP = 1:size(sim_betas,2) % loop over data partitions (DP)
        
        % select training data
        train_dp   = 1:size(sim_betas,2); train_dp(cDP) = [];
        trainSet   = nanmean(cat(3,sim_betas{train_dp}),3);
        
        % build design matrix (DM)
        DM(:,1)    = sin(deg2rad(1:step_sz:360)*cSym); % Sine
        DM(:,2)    = cos(deg2rad(1:step_sz:360)*cSym); % cosine
        
        %% (2) estimate grid orientation in training partitions
        for cVoxel = 1:size(trainSet,1)
            stats{cDP} = regstats(trainSet(cVoxel, :)', DM, 'linear', {'tstat'});
            grid_betas{cDP}(cVoxel, :) = stats{cDP}.tstat.beta(2:3);
        end
        est_orient = (bsxfun(@atan2, grid_betas{cDP}(:,1), grid_betas{cDP}(:,2))) ./ cSym;
        
        %% (3) constrast aligned vs. misaligned directions in test partition
        testSet    = sim_betas{cDP};
        for cVoxel = 1:size(testSet,1)
            
            % round estimated grid orientation (continuous) to sampled directions (discrete steps)
            if est_orient(cVoxel)<0; est_orient(cVoxel) = est_orient(cVoxel)+deg2rad(360); end
            [~, idx]       = min(abs(sampled_dirs - est_orient(cVoxel)));
            
            % find indices of all beta estimates corresponding to directions
            % aligned and misaligned relative to putative grid orienation
            idx_steps      = (360/cSym/step_sz);
            aligned_idx    = unique([idx:(-1.*idx_steps):1, idx:idx_steps:n_dirs]);
            misaligned_idx = aligned_idx+idx_steps/2;
            
            % round again to sampled directions and wrap around the circle
            % (simplified, the paper code balances rounding up/down to avoid biases)
            aligned_idx = round(aligned_idx); misaligned_idx = round(misaligned_idx);
            misaligned_idx(misaligned_idx>n_dirs) = misaligned_idx(misaligned_idx>n_dirs) - n_dirs;
            
            % select correct beta estimates and make contrast
            aligned     = testSet(cVoxel, aligned_idx);
            misaligned  = testSet(cVoxel, misaligned_idx);
            contrast{cDP}(cVoxel) = mean(aligned) - mean(misaligned);
            %Note: Shuffle test set here to obtain null distribution
            
        end % voxel
    end % data paritition
    
    %% average and plot
    n_plot = cSym-(symmetries(1)-1);
    subplot(1,numel(symmetries),n_plot);
    boxplot(nanmean(cell2mat(contrast)));
    if n_plot==1; ylabel('aligned - misaligned'); end
    title(sprintf('%d fold', cSym)); box off; ylim([-5, 5])
    
end % symmetry