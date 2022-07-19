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

local OctreeBatchNormalization, parent = torch.class('oc.OctreeBatchNormalization', 'oc.OctreeModule')

function OctreeBatchNormalization:__init(nInputPlane)
  parent.__init(self)
  
  self.nInputPlane = nInputPlane or error('need to specify nInputPlane')

  self.avgs = torch.Tensor(nInputPlane)
  self.vars = torch.Tensor(nInputPlane)
  self:reset()
end

function OctreeBatchNormalization:reset(stdv)
  self.avgs:fill(0)
  self.vars:fill(0)
end

function OctreeBatchNormalization:updateOutput(input)
  if input:feature_size() ~= self.nInputPlane then error('invalid input size') end
  
  -- print('[INFO] OctreeBatchNormalization updateOutput Start')
  if input._type == 'oc_float' then
    oc.cpu.octree_bn_norm_cpu(input.grid, self.avgs:data(), self.vars:data(), self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_bn_norm_gpu(input.grid, self.avgs:data(), self.vars:data(), self.output.grid)
  end
  -- print('[INFO] OctreeBatchNormalization updateOutput End')

  return self.output
end 

function OctreeBatchNormalization:updateGradInput(input, gradOutput)
  
  -- print('[INFO] OctreeBatchNormalization updateGradInput Start')
  if input._type == 'oc_float' then
    oc.cpu.octree_bn_norm_bwd_cpu(input.grid, gradOutput.grid, self.avgs:data(), self.vars:data(), self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_bn_norm_bwd_gpu(input.grid, gradOutput.grid, self.avgs:data(), self.vars:data(), self.gradInput.grid)
  end
  -- print('[INFO] OctreeBatchNormalization updateGradInput End')
  
  return self.gradInput
end
