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
        :add(oc.OctreeLeakyReLU(negative_slope, true))

    conv_p1 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeSigmoid(false))
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeSigmoid(false))

    conv2 = nn.Sequential()
        :add(oc.OctreeSplitByProb(conv_p1, 0.5, false))
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeLeakyReLU(negative_slope, true))

    conv_p2 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeSigmoid(false))
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeSigmoid(false))

    conv3 = nn.Sequential()
        :add(oc.OctreeSplitByProb(conv_p2, 0.5, false))
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeLeakyReLU(negative_slope, true))

    conv_p3 = nn.Sequential()
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeSigmoid(false))
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeSigmoid(false))

    conv4 = nn.Sequential()
        :add(oc.OctreeSplitByProb(conv_p3, 0.5, false))
        :add(oc.OctreeConvolutionMM(num_features, num_features))
        :add(oc.OctreeSigmoid(false))
        :add(oc.OctreeConvolutionMM(num_features, 1))
        :add(oc.OctreeSigmoid(false))
    -- 32x32x32

    local vol = -nn.Identity()
    local L1 = vol - conv1
    local L1_P = L1 - conv_p1
    local L2 = L1 - conv2
    local L2_P = L2 - conv_p2
    local L3 = L2 - conv3
    local L3_P = L3 - conv_p3
    local L4 = L3 - conv4

    model = nn.gModule({ vol }, { L1_P, L2_P, L3_P, L4 })
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
