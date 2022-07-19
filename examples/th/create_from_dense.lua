#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-batch-normalization/?/init.lua'
require('oc')


local input = torch.Tensor(16, 16, 16):zero()
input:sub(4, 8, 4, 8, 4, 8):fill(1)

ranges = torch.Tensor(2)
ranges[1] = 0.5
ranges[2] = 1.5

-- create_from_dense does NOT directly operate on input_oc!
input_oc = oc.FloatOctree():create_from_dense(input:float(), ranges:float())

print(input_oc:size())