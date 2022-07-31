#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('optim')
require('oc')

local dataloader = require('dataloader')
local completion_model = require('completion_model')
local train = require('completion_train')

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
opt.n_epochs = 240
opt.learningRate_steps = {}
-- opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = {
    oc.OctreeBCECriterion(true, false, true):cuda(), oc.OctreeBCECriterion(true, false, true):cuda(),
    oc.OctreeBCECriterion(true, false, true):cuda(), oc.OctreeSmoothMAECriterion():cuda()
}

opt.criterion_test = oc.OctreeMSECriterion():cuda() --nn.AbsCriterion():cuda() -- TODO implement and L1

-- create model
opt.net = completion_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")


local input, _target = train_data_loader:getBatch()
local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()
-- local target2 = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist/2):cuda()

-- print(input:size(), target:size())
-- print(opt.criterion[4]:forward(target2, target))
-- local tmp_oc8, tmp_ocp8, tmp_oc16, tmp_ocp16, tmp_oc32, tmp_ocp32 = train.split_target(target);

opt.net = torch.load('models/net_epoch190.t7')
local output = opt.net:forward(input)
-- output[4]:log_scale_inv()
-- print(output)
-- output[4]:log_scale_inv()
-- print(opt.net.modules[15]:get(1), opt.net.modules[15]:get(1).output:n_leafs())
-- print(opt.net.modules[15], opt.net.modules[15].output:n_leafs())
-- net = nn.Sequential()
--     :add(oc.OctreeToCDHW(opt.tr_dist))
--     :add(oc.CDHWToOctreeByInput(opt.tr_dist))
-- net:cuda()
-- local output = net:forward(input)
-- print(output:size())
-- local output = opt.net:forward(input)
-- tmp_oc32:write_to_bin('junk/output.oc')
-- output[2]:write_to_bin('junk/output2.oc')
-- output[4]:write_to_bin('junk/output4.oc')
-- output[5]:write_to_bin('junk/output5.oc')
-- output[6]:write_to_bin('junk/output6.oc')
-- output[7]:write_to_bin('junk/output7.oc')
input:write_to_bin('junk/input.oc')
output[4]:write_to_bin('junk/output.oc')
target:write_to_bin('junk/target.oc')
-- -- completion_model.model_to_dot(opt.net)
print(opt.net:getParameters():size(1))
-- train.worker(opt, train_data_loader, test_data_loader)