#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'
require('nn')
require('nngraph')
require('torch')
require('optim')
require('cudnn')
require('graph')
require('cunn') -- require before oc!
require('oc')

local f_ops = require('f_ops')
local train = require('train')

torch.setdefaulttensortype('torch.FloatTensor')

function dataset(N)
  local inputs = torch.Tensor(N, 32, 32, 32, 2):zero()
  local outputs = torch.Tensor(N, 32, 32, 32, 1)
  local f = f_ops.read_file("10155655850468db78d106ce0a280f87__0__.sdf")
  local sdf_f = f_ops.parse_sdf(f, 0.1) 
  local df_f = f_ops.parse_df(f, 0.1)
  for i = 1, N do
    inputs[i] = sdf_f
    outputs[i] = df_f:view(32,32,32,1)
  end
  return inputs, outputs
end

-- Number of samples.
N = 2

-- Inputs, outputs and ranges used for conversion.
inputs, outputs = dataset(N)

local num_features = 80
local negative_slope = 0.2

-- 32x32x32
conv1 = nn.Sequential()
    :add(oc.OctreeConvolutionMM(2, num_features))
    :add(oc.OctreeGridPool2x2x2('max'))
    :add(oc.OctreeLeakyReLU(negative_slope, true))
-- 16x16x16
conv2 = nn.Sequential()
    :add(oc.OctreeConvolutionMM(num_features, num_features * 2))
    :add(oc.OctreeGridPool2x2x2('max'))
    :add(oc.OctreeBatchNormalizationSS(num_features * 2))
    :add(oc.OctreeLeakyReLU(negative_slope, true))
-- 8x8x8
conv3 = nn.Sequential()
    :add(oc.OctreeConvolutionMM(num_features * 2, num_features * 4))
    :add(oc.OctreeToCDHW())-- convert to dense
    :add(cudnn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
    :add(nn.LeakyReLU(negative_slope, true))
-- 4x4x4
conv4 = nn.Sequential()
    :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 8, 4, 4, 4))
    :add(cudnn.VolumetricBatchNormalization(num_features * 8))
    :add(nn.LeakyReLU(negative_slope, true))
-- 1x1x1
bottleneck = nn.Sequential()
    :add(nn.View(num_features * 8))
    :add(nn.Linear(num_features * 8, num_features * 8))
    :add(nn.ReLU(true))
    :add(nn.Linear(num_features * 8, num_features * 8))
    :add(nn.ReLU(true))
    :add(nn.View(num_features * 8, 1, 1, 1))
-- 1x1x1
deconv1 = nn.Sequential()
    :add(cudnn.VolumetricFullConvolution(num_features * 16, num_features * 4, 4, 4, 4, 1, 1, 1, 0, 0, 0))
    :add(cudnn.VolumetricBatchNormalization(num_features * 4))
    :add(cudnn.ReLU(true))
-- 4x4x4
deconv2 = nn.Sequential()
    :add(cudnn.VolumetricFullConvolution(num_features * 8, num_features * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    :add(cudnn.VolumetricBatchNormalization(num_features * 2))
    :add(cudnn.ReLU(true))
    :add(oc.CDHWToOctree(conv3:get(2))) --return from dense
--8x8x8
deconv3 = nn.Sequential()
    :add(oc.OctreeGridUnpool2x2x2('max'))
    :add(oc.OctreeConvolutionMM(num_features * 4, num_features))
    :add(oc.OctreeBatchNormalizationSS(num_features))
    :add(oc.OctreeReLU(true))
--16x16x16
deconv4 = nn.Sequential()
    :add(oc.OctreeGridUnpool2x2x2('max'))
    :add(oc.OctreeConvolutionMM(num_features * 2, 1))
    :add(oc.OctreeBatchNormalizationSS(1))
    :add(oc.OctreeReLU(true))
--32x32x32

local vol = -nn.Identity()
local L1 = vol - conv1
local L2 = L1 - conv2
local L3 = L2 - conv3
local L4 = L3 - conv4
local L5 = L4 - bottleneck
local L6 = nn.JoinTable(2)({ L5, L4 }) - deconv1
local L7 = nn.JoinTable(2)({ L6, L3 }) - deconv2
local L8 = oc.OctreeConcat()({ L7, L2 }) - deconv3
local L9 = oc.OctreeConcat()({ L8, L1 }) - deconv4

model = nn.gModule({ vol }, { L9 })
model = require('oc_weight_init')(model, 'xavier')
model:cuda()

-- parameter size
-- params, gradParams = m:getParameters()
-- n_params = params:size(1)
-- print(n_params)

-- cudnn.convert(m, cudnn)
-- graph.dot(m.fg, 'Forward Graph', 'fg')
-- graph.dot(m.bg, 'Backward Graph', 'bg')

-- Sample a random batch from the dataset.

local opt = {}
opt.batch_size = 2

opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 20
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']
opt.net = model
opt.criterion = oc.OctreeCrossEntropyCriterion() -- TODO implement SmoothL1 and L1
opt.criterion:cuda()

-- train.worker(opt, inputs, outputs)

local shuffle = torch.randperm(N)
shuffle = shuffle:narrow(1, 1, opt.batch_size)
shuffle = shuffle:long()

local input = inputs:index(1, shuffle) -- Important for Octree conversion!
local output = outputs:index(1, shuffle)
local input_oc = oc.FloatOctree():octree_create_from_dense_features_batch(input)
input_oc = input_oc:cuda()
output = output:cuda()
print('input size:')
print(input_oc:size())
print('--------')

local pred = model:forward(input_oc)
print('output size:')
print(pred:size())
input_oc:write_to_bin("test.oc")


--- Definition of the objective on the current mini-batch.
-- This will be the objective fed to the optimization algorithm.
-- @param x input parameters
-- @return object value, gradients

--[[
local feval = function(x)

    -- Get new parameters.
    if x ~= parameters then
      parameters:copy(x)
    end

    -- Reset gradients
    gradParameters:zero()
  
    -- Evaluate function on mini-batch.
    local pred = model:forward(input_oc)
    local f = criterion:forward(pred, output)
  
    -- Estimate df/dW.
    local df_do = criterion:backward(pred, output)
    model:backward(input, df_do)
    print(pred:resize(1, batch_size))
    print(output:resize(1, batch_size))
    
    -- return f and df/dX
    return f, gradParameters
end

sgd_state = sgd_state or {
    learningRate = learning_rate,
    momentum = momentum,
    learningRateDecay = 5e-7
}

-- Returns the new parameters and the objective evaluated
-- before the update.
p, f = optim.sgd(feval, parameters, sgd_state)

print('['..t..']: '..f[1])
--]]
