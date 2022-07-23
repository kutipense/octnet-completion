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
    -- 16x16x16
    conv2 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, num_features * 2))
        :add(oc.OctreeGridPool2x2x2('max'))
        :add(oc.OctreeBatchNormalizationSS(num_features * 2))
        :add(oc.OctreeLeakyReLU(negative_slope, true))
    -- 8x8x8
    conv3 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features * 2, num_features * 4))
        :add(oc.OctreeToCDHW()) --convert to dense
        :add(cudnn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
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
        :add(oc.CDHWToOctree(conv3:get(2))) --return from dense
    --8x8x8
    deconv3 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features * 4, num_features * 2)) --additional
        :add(oc.OctreeBatchNormalizationSS(num_features * 2)) --additional
        :add(oc.OctreeReLU(true)) --additional
        :add(oc.OctreeGridUnpoolGuided2x2x2(conv2:get(1)))
        -- :add(oc.OctreeSplitFull())
        :add(oc.OctreeConvolutionMM(num_features * 2, num_features)) --num_features edit
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
    --16x16x16
    deconv4 = nn.Sequential()        
        :add(oc.OctreeConvolutionMM(num_features * 2, num_features)) --additional
        :add(oc.OctreeBatchNormalizationSS(num_features)) --additional
        :add(oc.OctreeReLU(true)) --additional
        :add(oc.OctreeGridUnpoolGuided2x2x2(conv1:get(1)))
    
    deconv4_inter = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeSigmoid(false))

    deconv4_cont = nn.Sequential()
        :add(oc.OctreeSplitByProb(deconv4_inter, 0, true))
        :add(oc.OctreeConvolutionMM(num_features, 1)) --num_features edit
        :add(oc.OctreeLogScale(false))
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
    local L9_inter = L9 - deconv4_inter
    local L9_out = L9 - deconv4_cont

    model = nn.gModule({ vol }, { L9_inter, L9_out })
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
