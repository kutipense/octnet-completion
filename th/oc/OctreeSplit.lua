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

local OctreeSplitByProb, parent = torch.class('oc.OctreeSplitByProb', 'oc.OctreeModule')

function OctreeSplitByProb:__init(prob_mod, threshold, check)
  parent.__init(self)

  self.prob_mod = prob_mod or error('need modules from which the split criterion can be computed')
  self.threshold = theshold or 0 -- 0 through a sigmoid is 0.5
  self.check = check or false
end

function OctreeSplitByProb:updateOutput(input)
  local prob = self.prob_mod.output or error('prob_mod.output is nil or false')

  if input._type == 'oc_float' then
    oc.cpu.octree_split_by_prob_cpu(input.grid, prob.grid, self.threshold, self.check, self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_by_prob_gpu(input.grid, prob.grid, self.threshold, self.check, self.output.grid)
  end

  return self.output 
end 

function OctreeSplitByProb:updateGradInput(input, gradOutput)
  if input._type == 'oc_float' then
    oc.cpu.octree_split_bwd_cpu(input.grid, gradOutput.grid, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_bwd_gpu(input.grid, gradOutput.grid, self.gradInput.grid)
  end

  return self.gradInput  
end



local OctreeSplitFull, parent = torch.class('oc.OctreeSplitFull', 'oc.OctreeModule')

function OctreeSplitFull:__init()
  parent.__init(self)
end

function OctreeSplitFull:updateOutput(input)
  if input._type == 'oc_float' then
    oc.cpu.octree_split_full_cpu(input.grid, self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_full_gpu(input.grid, self.output.grid)
  end

  return self.output 
end 

function OctreeSplitFull:updateGradInput(input, gradOutput)
  if input._type == 'oc_float' then
    oc.cpu.octree_split_bwd_cpu(input.grid, gradOutput.grid, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_bwd_gpu(input.grid, gradOutput.grid, self.gradInput.grid)
  end

  return self.gradInput  
end




local OctreeSplitRecSurf, parent = torch.class('oc.OctreeSplitRecSurf', 'oc.OctreeModule')

function OctreeSplitRecSurf:__init(rec_mod, threshold)
  parent.__init(self)

  self.rec_mod = rec_mod or error('need modules from which the split criterion can be computed')
  self.threshold = threshold or error('need to set threshold')
end

function OctreeSplitRecSurf:updateOutput(input)
  local rec = self.rec_mod.output or error('rec_mod.output is nil or false')

  if input._type == 'oc_float' then
    oc.cpu.octree_split_reconstruction_surface_cpu(input.grid, rec.grid, self.threshold, self.output.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_reconstruction_surface_gpu(input.grid, rec.grid, self.threshold, self.output.grid)
  end

  return self.output 
end 

function OctreeSplitRecSurf:updateGradInput(input, gradOutput)
  if input._type == 'oc_float' then
    oc.cpu.octree_split_bwd_cpu(input.grid, gradOutput.grid, self.gradInput.grid)
  elseif input._type == 'oc_cuda' then
    oc.gpu.octree_split_bwd_gpu(input.grid, gradOutput.grid, self.gradInput.grid)
  end

  return self.gradInput  
end



local OctreeDenseSplitSurf, parent = torch.class('oc.OctreeDenseSplitSurf', 'oc.OctreeModule')

function OctreeDenseSplitSurf:__init(rec_mod, threshold, structure_type)
  parent.__init(self)

  self.rec_mod = rec_mod or error('need modules from which the split criterion can be computed')
  self.threshold = threshold or error('need to set threshold')

  if structure_type == 'full' then
    self.structure_type = 0
  elseif structure_type == 'surface' then
    self.structure_type = 1
  elseif structure_type == 'octant' then
    self.structure_type = 2
  else
    error('invalid structure type')
  end

  self.gradInput = torch.Tensor()
end

function OctreeDenseSplitSurf:updateOutput(input)
  local rec = self.rec_mod.output or error('rec_mod.output is nil or false')

  if torch.type(input) == 'torch.CudaTensor' then
    oc.gpu.octree_split_dense_reconstruction_surface_gpu(input:data(), rec:data(), input:size(1), input:size(3), input:size(4), input:size(5), input:size(2), self.threshold, self.structure_type, self.output.grid)
  else
    error('unknow input type '..torch.type(input))
  end

  return self.output 
end 

function OctreeDenseSplitSurf:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)

  if torch.type(input) == 'torch.CudaTensor' then
    oc.gpu.octree_split_dense_reconstruction_surface_bwd_gpu(gradOutput.grid, self.gradInput:data())
  else
    error('unknow input type '..torch.type(input))
  end
  return self.gradInput  
end
