#!/usr/bin/env th
-- Path to octnet module.
-- Needs to be adapted depending on the installation directory!
package.path = package.path .. ';/root/vol/octnet-completion/th/?/init.lua'
require('nn')
require('cunn') -- require before oc!
require('oc')
require('torch')
require('optim')

-- http://jucor.github.io/torch-doc-template/tensor.html
torch.setdefaulttensortype('torch.FloatTensor') 

--- Creating a 16x16x16 dataset containing cubes at different positions.
-- The task is to classify whether the cube is in the upper or the
-- lower half.
-- @param N dataset size
function dataset(N)
  local inputs = torch.Tensor(N, 16, 16, 16):zero()
  local outputs = torch.Tensor(N):fill(0)
  for i = 1, N do
    local depth = 4 -- math.random(4, 6)
    local height = 4 -- math.random(4, 6)
    local width = 4 -- math.random(4, 6)
    
    local anchor_depth = 1 -- math.random(1, 16 - depth)
    local anchor_height = 1 -- math.random(1, 8 - height)
    
    if math.random() > 0.5 then
      outputs[i] = 1
      anchor_height = anchor_height + 8
    end
    
    local anchor_width = 1 -- math.random(1, 16 - width)
    
    inputs:sub(i, i, anchor_depth, anchor_depth + depth, 
      anchor_height, anchor_height + height,
      anchor_width, anchor_width + width):fill(1)
  end
  return inputs, outputs
end

-- Number of samples.
N = 1000

-- Inputs, outputs and ranges used for conversion.
ranges = torch.Tensor(2)
ranges[1] = 0.5
ranges[2] = 1.5
inputs, outputs = dataset(N)

model = nn.Sequential()
model:add(oc.OctreeConvolution3x3x3(1, 8, 'avg'))
model:add(oc.OctreeReLU(true))
model:add(oc.OctreeConvolution3x3x3(8, 8, 'avg'))
model:add(oc.OctreeReLU(true))
model:add(oc.OctreeGridPool2x2x2('max'))
model:add(oc.OctreeToCDHW())
model:add(nn.View(8*8*8*8))
model:add(nn.Linear(8*8*8*8, 8, true))
model:add(nn.ReLU())
model:add(nn.Linear(8, 1, true))
model:add(nn.Sigmoid())

--- Get the path of the current script.
-- @return script path
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

-- For initialization:
package.path = package.path .. ";" .. script_path() .. '?.lua'
model = require('oc_weight_init')(model, 'xavier')
model = model:cuda() -- !

criterion = nn.BCECriterion()
criterion = criterion:cuda() -- !

batch_size = 16
learning_rate = 0.05
momentum = 0.9

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda() -- !
gradParameters = gradParameters:cuda() -- !

T = 2500
for t = 1, T do
  
  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batch_size)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle) -- Important for Octree conversion!
  local output = outputs:index(1, shuffle)
  
  -- input and output contain the dense batches with dimensions
  -- (batch, depth, height, width).
  -- The dense tensor needs to be converted to a batch of Octrees.
  local input_oc = oc.FloatOctree():create_from_dense_batch(input, ranges)
  
  -- !
  input_oc = input_oc:cuda()
  output = output:cuda()
  
  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
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
end