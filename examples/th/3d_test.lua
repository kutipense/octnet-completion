#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('optim')
require('oc')

local dataloader = require('dataloader')
local completion_model = require('completion_model')
local train = require('train')

torch.setdefaulttensortype('torch.FloatTensor')


local opt = {}
opt.batch_size = 16
opt.data_paths = { 
    "/root/vol/octnet-completion/benchmark/sdf",
    "/root/vol/octnet-completion/benchmark/df" 
}

opt.negative_slope = 0.2
opt.num_features = 80
opt.full_batches = true
opt.tr_dist = 3
opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 20
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = oc.OctreeCrossEntropyCriterion() -- TODO implement SmoothL1 and L1
opt.criterion:cuda()

-- create model
opt.net = completion_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")

train.worker(opt, train_data_loader, test_data_loader)

-- local shuffle = torch.randperm(2)
-- shuffle = shuffle:narrow(1, 1, opt.batch_size)
-- shuffle = shuffle:long()
-- shuffle = torch.LongTensor({1,2})

-- local input = inputs:index(1, shuffle) -- Important for Octree conversion!
-- local output = outputs:index(1, shuffle)

-- local tr_dist = 3;

-- local input_oc = oc.FloatOctree():octree_create_from_dense_features_batch(input)
-- input_oc = input_oc:cuda()
-- input_oc:clamp(tr_dist)

-- local f = input_oc:float()
-- local s = f:extract_n(1,2)
-- local ss = f:extract_n(2,3)
-- print(ss:max())
-- ss:print()
-- local output_oc = oc.FloatOctree():octree_create_from_dense_features_batch(output)
-- input_oc = input_oc:cuda()
-- output_oc = output_oc:cuda()
-- print('input size:')
-- print(input_oc:size())
-- -- print('--------')

-- local pred = model:forward(input_oc)
-- print('output size:')
-- print(pred:size())
-- print(input_oc:n_elems(), pred:n_elems(),output_oc:n_elems())
-- ss:write_to_bin("test.oc")

