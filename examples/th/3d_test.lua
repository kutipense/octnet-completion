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
opt.batch_size = 8
opt.data_paths = { 
    "/root/vol/octnet-completion/benchmark/sdf",
    "/root/vol/octnet-completion/benchmark/df" 
}

opt.negative_slope = 0.2
opt.num_features = 80
opt.full_batches = true
opt.tr_dist = 3
-- opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 250
opt.learningRate_steps = {}
-- opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = nn.SmoothL1Criterion():cuda() -- TODO implement and L1
opt.criterion_test = nn.AbsCriterion():cuda() -- TODO implement and L1

-- create model
opt.net = completion_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
-- completion_model.model_to_dot(opt.net)
-- local input, target = train_data_loader:getBatch()
-- -- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
-- local input, _target = train_data_loader:getBatch()

-- local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
-- local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()

-- model = torch.load('models/best.t7')
-- model:evaluate()

-- output = model:forward(input)
-- output = torch.exp(output)  - 1
-- output = output:transpose(2,3)
-- output = output:transpose(3,4)
-- output = output:transpose(4,5):float()

-- output = oc.FloatOctree():octree_create_from_dense_features_batch(output, opt.tr_dist):cuda()

-- print(output:size())

-- input:write_to_bin('junk/input.oc')
-- output:write_to_bin('junk/output.oc')
-- target:write_to_bin('junk/target.oc')

-- train.worker(opt, train_data_loader, test_data_loader)
