#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('optim')
require('oc')

local dataloader = require('dataloader')
local struct_model = require('struct_model')
local struct_train = require('struct_train')

torch.setdefaulttensortype('torch.FloatTensor')


local opt = {}
opt.batch_size = 16
opt.data_paths = { 
    "/root/vol/octnet-completion/benchmark/sdf",
    "/root/vol/octnet-completion/benchmark/df" 
}

opt.negative_slope = 0.2
opt.num_features = 8
opt.full_batches = true
opt.tr_dist = 3
opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 250
opt.learningRate_steps = {}
-- opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.criterion = {
    oc.OctreeBCELevelCriterion(true, true, 0), oc.OctreeBCELevelCriterion(true, true, 1),
    oc.OctreeBCELevelCriterion(true, true, 2), oc.OctreeBCELevelCriterion(true, true, 3)
}
opt.criterion_test = oc.OctreeBCECriterion(true, true, true)

-- create model
opt.net = struct_model.create_model(opt)

-- create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")

-- struct_model.model_to_dot(opt.net)
struct_train.worker(opt, train_data_loader, test_data_loader)

-- local input, target = train_data_loader:getBatch()
-- -- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()

-- local input, _target = train_data_loader:getBatch()
-- local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
-- local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()

-- local f_ops = require("f_ops")
-- local sdf_batch = torch.Tensor(1, 32, 32, 32, 2):zero()
-- local df_batch = torch.Tensor(1, 32, 32, 32, 1)
-- local sdf = f_ops.parse_sdf(f_ops.read_file("b286c9c136784db2af1744fdb1fbe7df__0__.sdf"))
-- local df = f_ops.parse_df(f_ops.read_file("b286c9c136784db2af1744fdb1fbe7df__0__.df"))
-- sdf_batch[1] = sdf
-- df_batch[1] = df:view(32, 32, 32, 1)

-- local input = oc.FloatOctree():octree_create_from_dense_features_batch(sdf_batch, opt.tr_dist):cuda()
-- local target = oc.FloatOctree():octree_create_from_dense_features_batch(df_batch, opt.tr_dist):cuda()

-- local oo = oc.FloatOctree():cuda()
-- local oo2 = oc.FloatOctree():cuda()
-- oc.gpu.octree_gridpool2x2x2_max_gpu(input.grid, oo.grid) -- 16x16x16
-- oc.gpu.octree_gridpool2x2x2_max_gpu(oo.grid, oo2.grid) -- 8x8x8

-- local s = oc.OctreeSigmoid():cuda()
-- local sp1 = oc.OctreeSplitByProb(s, 0.5):cuda()
-- local sp2 = oc.OctreeSplitByProb(s, 0.5):cuda()
-- local gu1 = oc.OctreeGridUnpool2x2x2():cuda()
-- local gu2 = oc.OctreeGridUnpool2x2x2():cuda()
-- local n1 = oc.OctreeConvolutionMM(1,1):cuda()
-- local n2 = oc.OctreeConvolutionMM(1,1):cuda()
-- local n3 = oc.OctreeConvolutionMM(1,1):cuda()

-- output = n1:forward(oo2)

-- output = gu1:forward(output) -- 16x16x16
-- s:forward(output)
-- output = sp1:forward(output)
-- output = n2:forward(output)

-- print(output:size())

-- output = gu2:forward(output) -- 32x32x32
-- s:forward(output)
-- -- output = sp2:forward(output)
-- output = n3:forward(output)

-- print(output:size())
-- print(ss:size())

-- model = torch.load('models/net_epoch005.t7')
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
