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
opt.batch_size = 4
opt.data_paths = { 
    "/root/vol/octnet-completion/benchmark/sdf",
    "/root/vol/octnet-completion/benchmark/df" 
}

opt.negative_slope = 0.2
opt.num_features = 80
opt.full_batches = true
opt.tr_dist = 3
-- opt.weightDecay = 0.0001
opt.learningRate = 1e-2
opt.n_epochs = 250
opt.learningRate_steps = {}
-- opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = oc.OctreeSmoothMAECriterion() -- TODO implement and L1
opt.criterion_prob1 = oc.OctreeBCECriterion(true, false, true)
opt.criterion_prob2 = oc.OctreeBCECriterion(true, false, true)
opt.criterion:cuda()

-- create model
opt.net = completion_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
-- completion_model.model_to_dot(opt.net)
-- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
local input, target = train_data_loader:getBatch()

local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
local target = oc.FloatOctree():octree_create_from_dense_features_batch(target, opt.tr_dist):cuda()

model = torch.load('models/net_epoch010.t7')
model:evaluate()

print(model.modules)

output = model:forward(input)
output2 = model.modules[10].output
print(output2:size())
-- print(output)
output_p1 = output[1]
output_p2 = output[2]
output = output[3]
output:log_scale_inv()

input:write_to_bin('junk/input.oc')
output_p1:write_to_bin('junk/output_p1.oc')
output_p2:write_to_bin('junk/output_p2.oc')
output:write_to_bin('junk/output.oc')
target:write_to_bin('junk/target.oc')

-- train.worker(opt, train_data_loader, test_data_loader)
