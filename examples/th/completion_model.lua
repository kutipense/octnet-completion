#!/usr/bin/env th
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('nn')
require('nngraph')
require('graph')
require('cudnn')
require('cunn') -- require before oc!
require('oc')

local function create_model(opt)
    local num_features = opt.num_features
    local negative_slope = opt.negative_slope

    -- 32x32x32
    conv1 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(2, num_features))
        :add(oc.OctreeGridPool2x2x2('max'))
        :add(oc.OctreeLeakyReLU(negative_slope, true))
    conversion16x16x16 = nn.Sequential()
        :add(oc.OctreeToCDHW(opt.tr_dist)) -- convert to dense
    -- 16x16x16
    conv2 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, num_features * 2))
        :add(oc.OctreeGridPool2x2x2('max'))
        :add(oc.OctreeBatchNormalizationSS(num_features * 2))
        :add(oc.OctreeLeakyReLU(negative_slope, true))
    conversion8x8x8 = nn.Sequential()
        :add(oc.OctreeToCDHW(opt.tr_dist)) -- convert to dense
    -- 8x8x8
    conv3 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features * 2, num_features * 4))
        :add(oc.OctreeToCDHW(opt.tr_dist)) --convert to dense
        :add(cudnn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
        :add(cudnn.VolumetricBatchNormalization(num_features * 4))
        :add(nn.LeakyReLU(negative_slope, true))
    -- 4x4x4

    conv4 = nn.Sequential()
        :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 8, 4, 4, 4))
        :add(cudnn.VolumetricBatchNormalization(num_features * 8))
        :add(nn.LeakyReLU(negative_slope, true))
    -- 1x1x1
    bottleneck = nn.Sequential()
        :add(nn.Reshape(num_features * 8, true))
        :add(nn.Linear(num_features * 8, num_features * 8))
        :add(nn.ReLU(true))
        :add(nn.Linear(num_features * 8, num_features * 8))
        :add(nn.ReLU(true))
        :add(nn.Reshape(num_features * 8, 1, 1, 1))
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
    --8x8x8
    deconv3 = nn.Sequential()
        :add(cudnn.VolumetricFullConvolution(num_features * 4, num_features, 4, 4, 4, 2, 2, 2, 1, 1, 1))
        :add(cudnn.VolumetricBatchNormalization(num_features))
        :add(cudnn.ReLU(true))
    --16x16x16
    deconv4 = nn.Sequential()        
        :add(cudnn.VolumetricFullConvolution(num_features * 2, 1, 4, 4, 4, 2, 2, 2, 1, 1, 1)) --additional
        :add(nn.Abs())
        :add(nn.AddConstant(1))
        :add(nn.Log())

    --32x32x32

    local vol = -nn.Identity()
    local L1 = vol - conv1
    local L1_2 = L1 - conversion16x16x16
    local L2 = L1 - conv2
    local L2_2 = L2 - conversion8x8x8
    local L3 = L2 - conv3
    local L4 = L3 - conv4
    local L5 = L4 - bottleneck
    local L6 = nn.JoinTable(2)({ L5, L4 }) - deconv1
    local L7 = nn.JoinTable(2)({ L6, L3 }) - deconv2
    local L8 = nn.JoinTable(2)({ L7, L2_2 }) - deconv3
    local L9 = nn.JoinTable(2)({ L8, L1_2 }) - deconv4

    model = nn.gModule({ vol }, { L9 })
    model = require('oc_weight_init')(model, 'xavier')
    model:cuda()
    return model
end

local function model_to_dot(m, path)
    cudnn.convert(m, cudnn)
    graph.dot(m.fg, 'Forward Graph', 'fg')
    graph.dot(m.bg, 'Backward Graph', 'bg')
end

return {
    create_model = create_model,
    model_to_dot = model_to_dot
}
