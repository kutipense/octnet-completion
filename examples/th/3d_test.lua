#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/octnet/th/?/init.lua'

require('torch')
require('optim')
require('oc')

local dataloader = require('dataloader')
local completion_model = require('completion_model')
local train = require('train')

torch.setdefaulttensortype('torch.FloatTensor')


local opt = {}
opt.batch_size = 128
opt.data_paths = { 
    "/root/octnet/benchmark/sdf",
    "/root/octnet/benchmark/df" 
}

opt.negative_slope = 0.2
opt.num_features = 80
opt.full_batches = true
opt.tr_dist = 3
-- opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 5
opt.learningRate_steps = {}
-- opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = nn.SmoothL1Criterion():cuda() -- TODO implement and L1
opt.criterion_test = nn.AbsCriterion():cuda() -- TODO implement and L1

-- create model
opt.net = completion_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "train")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "val")
-- completion_model.model_to_dot(opt.net)
-- local input, target = train_data_loader:getBatch()
-- -- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
-- local input, _target = train_data_loader:getBatch()

-- local f_ops = require("f_ops")
-- local sdf_batch = torch.Tensor(1, 32, 32, 32, 2):zero()
-- local df_batch = torch.Tensor(1, 32, 32, 32, 1)
-- local sdf = f_ops.parse_sdf(f_ops.read_file("b286c9c136784db2af1744fdb1fbe7df__0__.sdf"))
-- local df = f_ops.parse_df(f_ops.read_file("b286c9c136784db2af1744fdb1fbe7df__0__.df"))
-- sdf_batch[1] = sdf
-- df_batch[1] = df:view(32, 32, 32, 1)

-- local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
-- local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()

-- model = torch.load('models/net_epoch100.t7')
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
-- print(model:getParameters():size(1))
train.worker(opt, train_data_loader, test_data_loader)
