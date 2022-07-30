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

local ffi = require('ffi')

local OctreeBCELevelCriterion, parent = torch.class('oc.OctreeBCELevelCriterion', 'oc.OctreeCriterion')

function OctreeBCELevelCriterion:__init(size_average, check, level)
  parent.__init(self)

  if size_average ~= nil then
    self.size_average = size_average
  else
    self.size_average = true
  end
  self.check = check or false
  self.level = level or 0
end

function OctreeBCELevelCriterion:updateOutput(input, target)
  self.output = -1
  
  oc.validateShape(input, target)

  local out = torch.FloatTensor(1)
  local total_weight = torch.FloatTensor(1)
  oc.gpu.octree_bce_loss_level_gpu(input.grid, target.grid, self.size_average, self.level, out:data(), total_weight:data())
    
  self.output = out[1]
  self.total_weight = total_weight[1]

  return self.output
end 

function OctreeBCELevelCriterion:updateGradInput(input, target)
  oc.gpu.octree_bce_loss_level_bwd_gpu(input.grid, target.grid, self.size_average, self.level, self.total_weight, self.gradInput.grid)
  return self.gradInput
end
