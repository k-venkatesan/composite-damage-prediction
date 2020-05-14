function process(file)

% Full file path might be required below
[image, map] = imread(strcat('\DataGeneration\DamagePatterns\Raw\', file, '.png'));
image = ind2rgb(image, map);
image = RemoveWhiteSpace(image);

% Full file path might be required below
imwrite(image, strcat('\DataGeneration\DamagePatterns\Processed\', file, '.png')); 