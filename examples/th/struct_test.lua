#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('optim')
require('oc')

local dataloader = require('dataloader')
local struct_train = require('struct_train')
torch.setdefaulttensortype('torch.FloatTensor')


local opt = {}
opt.batch_size = 8
opt.data_paths = { 
    "/root/vol/octnet-completion/benchmark/sdf",
    "/root/vol/octnet-completion/benchmark/df" 
}
opt.full_batches = true
opt.tr_dist = 3
opt.criterion_test = oc.OctreeBCECriterion(true, true, true)
opt.net = torch.load('struct_models/net_epoch240.t7')

--create data loader
local train_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")
local test_data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit")

local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
-- local input, target = train_data_loader:getBatch()
local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
local target = oc.FloatOctree():octree_create_from_dense_features_batch(target, opt.tr_dist):cuda()


-- opt.net:evaluate()

-- output = opt.net:forward(input)
-- local tmp_oc = oc.FloatOctree():cuda()
-- oc.gpu.octree_split_by_prob_gpu(output[#output].grid, output[#output].grid, 0.5, true, tmp_oc.grid)
-- output[#output] = tmp_oc

-- local l1, l2, l3 = struct_train.split_target(target)
-- print(l1:n_leafs(), target:n_leafs())

input:write_to_bin('struct_junk/input.oc')
for i, v in ipairs(output) do
    v:write_to_bin('struct_junk/output' .. i .. '.oc')
end
-- l1:write_to_bin('struct_junk/target1.oc')
-- l2:write_to_bin('struct_junk/target2.oc')
-- l3:write_to_bin('struct_junk/target3.oc')
target:write_to_bin('struct_junk/target.oc')
-- print(opt.net:getParameters():size(1))
