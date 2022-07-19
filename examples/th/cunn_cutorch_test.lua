#!/usr/bin/env th

require('cunn')

if cutorch then
  print('cutorch found!')
else
  print('cutorch not found!')
end