% Dimensions for plate and hole in centimetres (whole major and minor axis, not half)
L = 10;
W = 10;
t_lam = 1;
Sa = 2;
Sb = 4;

% No. of different non-zero loading conditions in X and Y direction
nx = 20;
ny = 20;

% Load variable initialisation - the actual load in centimetres is the load variable divided by 100
X = 0;
Y = 0;

% Full directory path might be required below
cd '\data-generation\damage-patterns\processed';

% Loop to carry out operations over the range of X and Y values
while X <= 100
    
    while Y <= 100
        
        modelName = strcat('Composite_L', int2str(L), '_W', int2str(W), '_t', int2str(t_lam), '_Sa', int2str(Sa), '_Sb', int2str(Sb), '_X', sprintf('%03d', X), '_Y', sprintf('%03d', Y));
        
        % To ensure that the operations are carried out only if this model
        % has not been processed already
        if exist(strcat('Empty_', modelName)) ~= 7
            
            for i = 1:4
                
                process(strcat('FC_Ply', int2str(i), '_', modelName));
                process(strcat('FT_Ply', int2str(i), '_', modelName));
                process(strcat('MC_Ply', int2str(i), '_', modelName));
                process(strcat('MT_Ply', int2str(i), '_', modelName));
                
            end
            
            mkdir(strcat('Empty_', modelName));
            
        end
        
        Y = Y + 100/ny;
        
    end
    
    Y = 0;
    X = X + 100/nx;
    
end         
        
        