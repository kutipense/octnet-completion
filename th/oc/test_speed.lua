#!/usr/bin/env th

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

require('nn')
require('cunn')
require('cudnn')
require('sys')

local test_utils = require('test_utils')

local function speed_conv_fwd()
  local reps = 50
  local backend = cudnn
  -- local backend = nn
  local test_fcn = ''

  local col_buffer = oc.OTCudaBuffer('ot_data_t')

  local function test_vol_fwd(cin,cout, grid_h)
    local dense_d = oc.OctreeToCDHW():forward({grid_h}):cuda()
    local conv = backend.VolumetricConvolution(cin,cout, 3,3,3, 1,1,1, 1,1,1):cuda()

    local out = conv:forward(dense_d) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward(dense_d)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_dense_fwd(cin,cout, grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeDenseConvolution(cin,cout, 'avg', true, backend):cuda()

    local out = conv:forward({grid_d}) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward({grid_d})
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_mm_fwd(cin,cout, grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeConvolutionMM(cin,cout, col_buffer):cuda()

    local out = conv:forward({grid_d}) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward({grid_d})
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end



  local function test_vol_bwd(cin,cout, grid_h)
    local dense_d = oc.OctreeToCDHW():forward({grid_h}):cuda()
    local conv = backend.VolumetricConvolution(cin,cout, 3,3,3, 1,1,1, 1,1,1):cuda()

    local out = conv:forward(dense_d) 
    local grad_out = out:clone()
    local out = conv:backward(dense_d, grad_out) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:backward(dense_d, grad_out)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_dense_bwd(cin,cout, grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeDenseConvolution(cin,cout, 'avg', true, backend):cuda()

    local out = conv:forward({grid_d}) 
    local grad_out = { out[1]:clone() }
    local out = conv:backward({grid_d}, grad_out) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:backward({grid_d}, grad_out)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_mm_bwd(cin,cout, grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeConvolutionMM(cin,cout, col_buffer):cuda()

    local out = conv:forward({grid_d}) 
    local grad_out = { out[1]:clone() }
    local out = conv:backward({grid_d}, grad_out) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:backward({grid_d}, grad_out)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end



  local function test_vol_pool_fwd(grid_h)
    local dense_d = oc.OctreeToCDHW():forward({grid_h}):cuda()
    local conv = backend.VolumetricMaxPooling(2,2,2, 2,2,2):cuda()

    local out = conv:forward(dense_d) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward(dense_d)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_oc_pool_fwd(grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeGridPool2x2x2('max'):cuda()

    local out = conv:forward({grid_d}) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward({grid_d})
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_vol_pool_bwd(grid_h)
    local dense_d = oc.OctreeToCDHW():forward({grid_h}):cuda()
    local conv = backend.VolumetricMaxPooling(2,2,2, 2,2,2):cuda()

    local out = conv:forward(dense_d) 
    local grad_out = out:clone()
    local out = conv:backward(dense_d, grad_out) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:backward(dense_d, grad_out)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end

  local function test_oc_pool_bwd(grid_h)
    local grid_d = grid_h:cuda()
    local conv = oc.OctreeGridPool2x2x2('max'):cuda()

    local out = conv:forward({grid_d}) 
    local grad_out = { out[1]:clone() }
    local out = conv:backward({grid_d}, grad_out) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:backward({grid_d}, grad_out)
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end


  local function test_dense_to_octree(grid_h)
    local dense_d = oc.OctreeToCDHW():forward({grid_h}):cuda()
    local grid_d = grid_h:cuda()
    local conv = oc.CDHWToOctree('avg'):cuda()

    local out = conv:forward({{grid_d}, dense_d}) 
    sys.tic()
    for rep = 1, reps do 
      local out = conv:forward({{grid_d}, dense_d})
    end
    cutorch.synchronize()
    return sys.toc() / reps
  end


  local function select_function(cin, cout, grid_h, do_vol,do_dense,do_mm) 
    local timings = {-1, -1, -1}
    if test_fcn == 'conv_fwd' then
      if do_vol   then timings[1] = test_vol_fwd(cin,cout, grid_h) end
      if do_dense then timings[2] = test_dense_fwd(cin,cout, grid_h) end
      if do_mm    then timings[3] = test_mm_fwd(cin,cout, grid_h) end
    elseif test_fcn == 'conv_bwd' then
      if do_vol   then timings[1] = test_vol_bwd(cin,cout, grid_h) end
      if do_dense then timings[2] = test_dense_bwd(cin,cout, grid_h) end
      if do_mm    then timings[3] = test_mm_bwd(cin,cout, grid_h) end
    elseif test_fcn == 'gridpool_fwd' then
      if do_vol   then timings[1] = test_vol_pool_fwd(grid_h) end
      if do_mm    then timings[3] = test_oc_pool_fwd(grid_h) end
    elseif test_fcn == 'gridpool_bwd' then
      if do_vol   then timings[1] = test_vol_pool_bwd(grid_h) end
      if do_mm    then timings[3] = test_oc_pool_bwd(grid_h) end
    elseif test_fcn == 'dense_to_octree' then
      if do_mm    then timings[3] = test_dense_to_octree(grid_h) end
    else
      error('unknown test_fcn')
    end

    return timings
  end

  local function test_rand(cin,cout, gd,gh,gw, sp0,sp1,sp2, do_vol,do_dense,do_mm)
    local grid_h = test_utils.octree_rand(gd,gh,gw, cin, sp0,sp1,sp2)
    local timings = select_function(cin,cout, grid_h, do_vol,do_dense,do_mm)
    print(string.format('%d,%d, %d,%d,%d, %.2f,%.2f,%.2f: %d | %f | %f | %f', cin,cout, gd,gh,gw, sp0,sp1,sp2, grid_h:n_leafs(), timings[1], timings[2], timings[3]))
  end

  local function test_bin(cout, path, do_vol,do_dense,do_mm, fc)
    local grid_h = oc.FloatOctree()
    grid_h:read_from_bin(path)
    if fc then
      grid_h:resize(grid_h:grid_depth(), grid_h:grid_height(), grid_h:grid_width(), fc, grid_h:n_leafs())
      grid_h:magic_data_ptrs()
      
      local grid_data = grid_h:data()
      grid_data:apply(function() return torch.uniform(-1,1) end)
    end

    local cin = grid_h:feature_size()
    local timings = select_function(cin,cout, grid_h, do_vol,do_dense,do_mm)
    print(string.format('%d,%d, %s %d | %f | %f | %f', cin,cout, path, grid_h:n_leafs(), timings[1], timings[2], timings[3]))
  end


  test_fcn = 'conv_fwd'
  -- test_fcn = 'conv_bwd'
  -- test_fcn = 'gridpool_fwd'
  -- test_fcn = 'gridpool_bwd'
  -- test_fcn = 'dense_to_octree'


  print(test_fcn)
  print('grid parameters | conv_vol | conv_dense | conv_mm')
  print('----------------|----------|------------|--------')
  -- test_rand(2,3,   2,4,4,    0.5,0.5,0.5,      false,false,true)
  -- test_rand(32,64, 8,8,8,    0.0,0.0,0.0,      false,false,true)
  -- test_rand(32,64, 8,8,8,    0.5,0.5,0.5,      false,false,true)
  -- test_rand(32,64, 8,8,8,    0.1,0.5,0.5,      false,false,true)
  -- test_rand(32,64, 8,8,8,    0.1,0.1,0.5,      false,false,true)
  -- test_rand(32,64, 16,16,16, 0.0,0.0,0.0,      false,false,true)
  -- test_rand(32,64, 16,16,16, 0.5,0.5,0.5,      false,false,true)
  -- test_rand(32,64, 16,16,16, 0.1,0.5,0.5,      false,false,true)
  -- test_rand(32,64, 16,16,16, 0.1,0.1,0.5,      false,false,true)
  -- test_rand(32,32, 24,24,24, 0.0,0.0,0.0,      false,false,true) 
  -- test_rand(32,32, 24,24,24, 0.1,0.1,0.5,      false,false,true) 
  -- test_rand(32,32, 32,32,32, 0.0,0.0,0.0,   true,false,true) 
  -- test_rand(32,32, 32,32,32, 0.1,0.1,0.5,   true,false,true) 
  
  for _, fcs in ipairs{false, 32, 64} do
    test_bin(32, 'test_octrees/plane_32.bin',    false,false,true, fcs)
    test_bin(32, 'test_octrees/table_32.bin',    false,false,true, fcs)

    test_bin(32, 'test_octrees/table_64.bin',    false,false,true, fcs)
    test_bin(32, 'test_octrees/plane_64.bin',    false,false,true, fcs)

    test_bin(32, 'test_octrees/plane_128.bin',   false,false,true, fcs)
    test_bin(32, 'test_octrees/table_128.bin',   false,false,true, fcs)

    -- test_bin(32, 'test_octrees/plane_256.bin',   false,false,true, fcs)
    -- test_bin(32, 'test_octrees/table_256.bin',   false,false,true, fcs)
    
    -- test_bin(32, 'test_octrees/plane_512.bin',   false,false,true, fcs)
    -- test_bin(32, 'test_octrees/table_512.bin',   false,false,true, fcs)
  end

  -- test_bin(32, 'test_octrees/plane_p32.bin',   false,false,true)
  -- test_bin(32, 'test_octrees/plane_p64.bin',   false,false,true)
  -- test_bin(32, 'test_octrees/plane_p128.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/plane_p256.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/plane_p512.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/table_p32.bin',   false,false,true)
  -- test_bin(32, 'test_octrees/table_p64.bin',   false,false,true)
  -- test_bin(32, 'test_octrees/table_p128.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/table_p256.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/table_p512.bin',  false,false,true)

  -- test_bin(32, 'test_octrees/plane_pp32.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/plane_pp64.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/plane_pp128.bin', false,false,true)
  -- test_bin(32, 'test_octrees/plane_pp256.bin', false,false,true)
  -- test_bin(32, 'test_octrees/plane_pp512.bin', false,false,true)
  -- test_bin(32, 'test_octrees/table_pp32.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/table_pp64.bin',  false,false,true)
  -- test_bin(32, 'test_octrees/table_pp128.bin', false,false,true)
  -- test_bin(32, 'test_octrees/table_pp256.bin', false,false,true)
  -- test_bin(32, 'test_octrees/table_pp512.bin', false,false,true)
end


function speed_io()
  local function test_oc(n, grid_depth, grid_height, grid_width, feature_size, n_threads)
    local grids = {}
    local oc_paths = {}
    print('write data')
    for idx = 1, n do
      local grid = test_utils.octree_rand(1,grid_depth,grid_height,grid_width, feature_size, 1,1,0)
      table.insert(grids, grid)
      local oc_path = string.format('test_grid_%02d.oc', idx)
      table.insert(oc_paths, oc_path)
      grid:write_to_bin(oc_path)
    end

    print('load data')
    local grid_b = oc.FloatOctree()
    sys.tic()
    grid_b:read_from_bin_batch(oc_paths, n_threads)
    local t = sys.toc()

    print(string.format('io batch (%d, %d,%d,%d, %d), %d took %f[s]', n, grid_depth,grid_height,grid_width, feature_size, n_threads, t))
  end 

  local function test_dense(n, depth, height, width, feature_size, n_threads)
    local tensors = {}
    local tensor_paths = {}

    print('write data')
    for idx = 1, n do
      local tensor = torch.randn(1, feature_size, depth,height,width):float()
      table.insert(tensors, tensor)
      local tensor_path = string.format('test_tensor_%02d.cdhw', idx)
      table.insert(tensor_paths, tensor_path)
      oc.write_dense_to_bin(tensor_path, tensor)
    end

    print('load data')
    local tensor_b = torch.FloatTensor(n, feature_size, depth,height,width)
    sys.tic()
    oc.read_dense_from_bin_batch(tensor_paths, tensor_b, n_threads)
    local t = sys.toc()
    
    print(string.format('io cdhw (%d, %d,%d,%d, %d), %d took %f[s]', n, depth,height,width, feature_size, n_threads, t))
  end 


  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_oc(8, 1,1,1, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_oc(32, 1,1,1, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_oc(32, 8,8,8, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_oc(32, 16,16,16, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_oc(32, 16,16,16, 8, nt) end

  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_dense( 8, 8,8,8, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_dense(32, 8,8,8, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_dense(32, 64,64,64, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_dense(32, 128,128,128, 1, nt) end
  print(''); for _, nt in ipairs{1, 2, 4, 8} do test_dense(32, 128,128,128, 8, nt) end

end 

-- constat seed, so experiments can be repeated and still yield same results
local seed = 42 
print('seed: '..seed)
math.randomseed(seed)
torch.manualSeed(seed)
torch.setnumthreads(1)

-- speed_conv_fwd()
speed_io()
