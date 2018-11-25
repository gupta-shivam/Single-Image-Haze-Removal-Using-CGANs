#!/usr/bin/octave -qf

pkg load image
input = "data/test"
output = "data/downscaled"
liste = dir(input);
for i=3:size(liste,1)
 C = imresize(imread([input '/' liste(i).name]), [256,256]);
 imwrite(C, [output '/' liste(i).name])
end
