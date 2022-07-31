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

local function create_decoder_layer(in_f, out_f, negative_slope, bn, mp)
    local layer = nn.Sequential()
        :add(oc.OctreeConvolutionMM(in_f, out_f))

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

    -- 32x32x32
    conv1 = create_encoder_layer(2, num_features, negative_slope, false, true) -- 16x16x16 output
    conv2 = create_encoder_layer(num_features, num_features * 2, negative_slope, true, true) -- 8x8x8 output
    conv3 = create_encoder_layer(num_features * 2, num_features * 4, negative_slope, true, false) -- psuedo 4x4x4 (8x8x8 pre pool)

    conv3_convert = nn.Sequential()
        :add(oc.OctreeToCDHW(opt.tr_dist))
        :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 4, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- additional
        :add(cudnn.VolumetricBatchNormalization(num_features * 4)):add(nn.LeakyReLU(negative_slope, true)) -- additional

    -- volumetric start 4x4x4x320
    conv3_pool = nn.Sequential():add(cudnn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2)) -- 4x4x4

    conv4 = nn.Sequential()
        :add(cudnn.VolumetricConvolution(num_features * 4, num_features * 8, 4, 4, 4)) -- 1x1x1
        :add(cudnn.VolumetricBatchNormalization(num_features * 8)):add(nn.LeakyReLU(negative_slope, true))
    -- 1x1x1
    bottleneck = nn.Sequential()
        :add(nn.Reshape(num_features * 8, true))
        :add(nn.Linear(num_features * 8, num_features * 8)):add(nn.ReLU(true))
        :add(nn.Linear(num_features * 8, num_features * 8)):add(nn.ReLU(true))
        :add(nn.Reshape(num_features * 8, 1, 1, 1))
    -- -- 1x1x1
    deconv1 = nn.Sequential()
        :add(cudnn.VolumetricFullConvolution(num_features * 16, num_features * 4, 4, 4, 4, 1, 1, 1, 0, 0, 0)) -- 4x4x4
        :add(cudnn.VolumetricBatchNormalization(num_features * 4))
        :add(cudnn.ReLU(true))
    -- 4x4x4
    deconv2 = nn.Sequential()
        :add(cudnn.VolumetricFullConvolution(num_features * 8, num_features * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1)) -- 8x8x8
        :add(cudnn.VolumetricBatchNormalization(num_features * 2))
        :add(cudnn.ReLU(true))
    -- volumetric end 8x8x8x160

    --8x8x8
    deconv3 = nn.Sequential()
        :add(cudnn.VolumetricConvolution(num_features * 6, num_features, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- 8x8x8
        :add(cudnn.VolumetricBatchNormalization(num_features))
        :add(cudnn.ReLU(true))

    deconv3_convert = nn.Sequential()
        :add(oc.CDHWToOctreeByInput(opt.tr_dist))
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
        :add(oc.OctreeGridUnpool2x2x2())

    --16x16x16
    deconv4 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features*2, num_features))
        :add(oc.OctreeBatchNormalizationSS(num_features))
        :add(oc.OctreeReLU(true))
        :add(oc.OctreeGridUnpool2x2x2())
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeLogScale(false))
    --32x32x32

    local vol = -nn.Identity()
    local L1 = vol - conv1
    local L2 = L1 - conv2
    local L3 = L2 - conv3 - conv3_convert
    
    local L3_pool = L3 - conv3_pool
    local L4 = L3_pool - conv4
    local L5 = L4 - bottleneck
    local L6 = nn.JoinTable(2)({ L5, L4 }) - deconv1 --decoder entry
    local L7 = nn.JoinTable(2)({ L6, L3_pool }) - deconv2

    local L8 = nn.JoinTable(2)({ L7, L3 }) - deconv3 - deconv3_convert
    local L9 = oc.OctreeConcatDS()({ L8, L1 }) - deconv4

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
