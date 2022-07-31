#!/usr/bin/env th
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'

require('torch')
require('nn')
require('nngraph')
require('graph')
require('cudnn')
require('cunn') -- require before oc!
require('oc')

local function create_encoder_layer(in_f, out_f, negative_slope, bn, mp)
    local layer = nn.Sequential():add(oc.OctreeConvolutionMM(in_f, out_f))

    if mp == true then
        layer:add(oc.OctreeGridPool2x2x2('max'))
    end

    if bn == true then
        layer:add(oc.OctreeBatchNormalizationSS(out_f))
    end

    layer:add(oc.OctreeLeakyReLU(negative_slope, true))
    return layer
end


local function create_model(opt)
    local num_features = opt.num_features
    local negative_slope = opt.negative_slope


    local vol = -nn.Identity() -- 32x32x32
    local conv1 = create_encoder_layer(2, num_features, negative_slope, false, true) -- 16x16x16 output
    local L1 = vol - conv1
    
    local conv2 = create_encoder_layer(num_features, num_features * 2, negative_slope, true, true) -- 8x8x8 output
    local L2 = L1 - conv2

    local conv3 = nn.Sequential()
        :add(oc.OctreeToCDHW(opt.tr_dist))
        :add(cudnn.VolumetricConvolution(num_features * 2, num_features * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1)) -- 4x4x4
        :add(cudnn.VolumetricBatchNormalization(num_features * 4)):add(nn.LeakyReLU(negative_slope, true))
    local L3 = L2 - conv3

    -- 4x4x4
    local bottleneck = nn.Sequential()
        :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 4, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- 4x4x4
        :add(cudnn.VolumetricBatchNormalization(num_features * 4)):add(nn.ReLU(true))
        :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 4, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- 4x4x4
        :add(cudnn.VolumetricBatchNormalization(num_features * 4)):add(nn.ReLU(true))
    local L4 = L3 - bottleneck
    -- 4x4x4
    
    -- 4x4x4
    local deconv1 = nn.Sequential()
        :add(cudnn.VolumetricFullConvolution(num_features * 8, num_features * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1)) -- 8x8x8
        :add(cudnn.VolumetricBatchNormalization(num_features * 2))
        :add(cudnn.ReLU(true))
        :add(oc.CDHWToOctreeByInput(opt.tr_dist))
    local L5 = nn.JoinTable(2)({ L4, L3 }) - deconv1

    local deconv1_sig = L5 - oc.OctreeConvolutionMM(num_features*2, 1) - oc.OctreeSigmoid(false)
    -- local deconv1_df = L5 - oc.OctreeConvolutionMM(num_features*2, 1) - oc.OctreeLogScale(false)

    -- 8x8x8
    local deconv2 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features*4, num_features))
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
        :add(oc.OctreeGridUnpool2x2x2()) -- 16x16x16
    local L6 = oc.OctreeConcatDS()({ L5, L2 }) - deconv2

    local tmp_sig1 = oc.OctreeSigmoid(false)
    local deconv2_sig = L6 - oc.OctreeConvolutionMM(num_features, 1) - tmp_sig1
    -- local deconv2_df = L6 - oc.OctreeConvolutionMM(num_features, 1) - oc.OctreeLogScale(false)
  
    --16x16x16
    local deconv3 = nn.Sequential()
        :add(oc.OctreeSplitByProb(tmp_sig1, 0.5, true))
        :add(oc.OctreeConvolutionMM(num_features*2, num_features))
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
        :add(oc.OctreeGridUnpool2x2x2()) -- 32x32x32
    local L7 = oc.OctreeConcatDS()({ L6, L1 }) - deconv3

    local tmp_sig2 = oc.OctreeSigmoid(false)
    local deconv3_sig =  L7 - oc.OctreeConvolutionMM(num_features, 1) - tmp_sig2
    -- local deconv3_df = L7 - oc.OctreeConvolutionMM(num_features, 1) - oc.OctreeLogScale(false)

    --32x32x32
    local deconv4 = nn.Sequential()
        :add(oc.OctreeSplitByProb(tmp_sig2, 0.5, true))
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
        :add(oc.OctreeConvolutionMM(num_features, 1))
        -- :add(oc.OctreeLogScale(false))
    local L8 = L7 - deconv4


    model = nn.gModule({ vol }, { deconv1_sig, deconv2_sig, deconv3_sig, L8 })
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
