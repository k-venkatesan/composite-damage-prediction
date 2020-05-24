function process(file)

% Full file path might be required below
[image, map] = imread(strcat('\data-generation\damage-patterns\raw\', file, '.png'));
image = ind2rgb(image, map);
image = RemoveWhiteSpace(image);

% Full file path might be required below
imwrite(image, strcat('\data-generation\damage-patterns\processed\', file, '.png')); 