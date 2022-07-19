-- Copyright (c) 2017, The OctNet authors
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--     * Redistributions of source code must retain the above copyright
--       notice, this list of conditions and the following disclaimer.
--     * Redistributions in binary form must reproduce the above copyright
--       notice, this list of conditions and the following disclaimer in the
--       documentation and/or other materials provided with the distribution.
--     * Neither the name of the <organization> nor the
--       names of its contributors may be used to endorse or promote products
--       derived from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

local OctreeBatchNormalizationSS, parent = torch.class('oc.OctreeBatchNormalizationSS', 'oc.OctreeModule')

function OctreeBatchNormalizationSS:__init(nInputPlane, inplace)
  parent.__init(self)
  
  self.nInputPlane = nInputPlane or error('need to specify nInputPlane')
  self.inplace = inplace or false
  
  self.gamma = torch.Tensor(nInputPlane)
  self.beta = torch.Tensor(nInputPlane)
  self.gradGamma = torch.Tensor(nInputPlane)
  self.gradBeta = torch.Tensor(nInputPlane)
  self:reset()
end

function OctreeBatchNormalizationSS:reset(stdv)
  stdv = stdv or 1.0/self.nInputPlane
  
  -- self.gamma:fill(1)
  -- self.beta:fill(0)
  
  self.gamma:uniform(-stdv, stdv)
  self.beta:uniform(-stdv, stdv)
end

function OctreeBatchNormalizationSS:updateOutput(input)
  if input:feature_size() ~= self.nInputPlane then error('invalid input size') end
  
  -- print('[INFO] OctreeBatchNormalizationSS updateOutput Start')
  if input._type == 'oc_float' then
    oc.cpu.octree_bn_ss_cpu(input.grid, self.gamma:data(), self.beta:data(), self.inplace, self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_bn_ss_gpu(input.grid, self.gamma:data(), self.beta:data(), self.inplace, self.output.grid)
  end
  -- print('[INFO] OctreeBatchNormalizationSS updateOutput End')
  
  return self.output
end 

function OctreeBatchNormalizationSS:updateGradInput(input, gradOutput)
  
  -- print('[INFO] OctreeBatchNormalizationSS updateGradInput Start')
  if input._type == 'oc_float' then
    oc.cpu.octree_bn_ss_bwd_cpu(gradOutput.grid, self.gamma:data(), self.inplace, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_bn_ss_bwd_gpu(gradOutput.grid, self.gamma:data(), self.inplace, self.gradInput.grid)
  end
  -- print('[INFO] OctreeBatchNormalizationSS updateGradInput End')
  
  return self.gradInput
end

function OctreeBatchNormalizationSS:accGradParameters(input, gradOutput)
  
  -- print('[INFO] OctreeBatchNormalizationSS accGradParameters Start')
  if input._type == 'oc_float' then
    oc.cpu.octree_bn_ss_wbwd_cpu(input.grid, gradOutput.grid, self.gradGamma:data(), self.gradBeta:data())
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_bn_ss_wbwd_gpu(input.grid, gradOutput.grid, self.gradGamma:data(), self.gradBeta:data())
  end
  -- print('[INFO] OctreeBatchNormalizationSS accGradParameters End')
end

