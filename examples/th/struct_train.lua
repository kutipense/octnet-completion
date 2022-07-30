#!/usr/bin/env th


function split_target(target)
  local tmp_oc2 = oc.FloatOctree():cuda()
  local tmp_oc3 = oc.FloatOctree():cuda()
  local tmp_oc4 = oc.FloatOctree():cuda()
  oc.gpu.octree_pool2x2x2_max_gpu(target.grid, false, false, true, tmp_oc2.grid)
  oc.gpu.octree_pool2x2x2_max_gpu(tmp_oc2.grid, false, true, false, tmp_oc3.grid)
  oc.gpu.octree_pool2x2x2_max_gpu(tmp_oc3.grid, true, false, false, tmp_oc4.grid)

  tmp_oc2:to_occupancy()
  tmp_oc3:to_occupancy()
  tmp_oc4:to_occupancy()

  ------ l1       l2       l3
  return tmp_oc4, tmp_oc3, tmp_oc2
end

function get_loss(criterion, output, target)
  local tmp_oc = oc.FloatOctree():cuda()
  oc.gpu.octree_split_by_prob_gpu(output[#output].grid, output[#output].grid, 0.5, true, tmp_oc.grid)
  output[#output] = tmp_oc

  local f = {}
  local dfdx = {}

  for i, c in ipairs(criterion) do
    f[i] = c:forward(output[i], target[i])
    dfdx[i] = c:backward(output[i], target[i]):cuda() --:mul(#criterion - i + 1)
  end

  return f, dfdx
end

function train_epoch(opt, data_loader)
  local net = opt.net or error('no net in train_epoch')
  local criterion = opt.criterion or error('no criterion in train_epoch')
  local optimizer = opt.optimizer or error('no optimizer in train_epoch')

  local n_batches = data_loader:n_batches()

  net:training()

  local parameters, grad_parameters = net:getParameters()
  for batch_idx = 1, n_batches do
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      grad_parameters:zero()

      local _input, _target = data_loader:getBatch()
      local input = oc.FloatOctree():octree_create_from_dense_features_batch(_input, opt.tr_dist):cuda()
      local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()
     
      local l1, l2, l3 = split_target(target)
      target:to_occupancy()
      
      local output = net:forward(input)
      local f, dfdx = get_loss(criterion, output, {l1, l2, l3, target})
      
      -- print(f, dfdx)
      net:backward(input, dfdx)
      
      local saved = false
      if(f[#f] < opt.min_loss) then
        opt.min_loss = f[#f]
        -- local net_path = 'struct_models/best.t7' --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
        -- torch.save(net_path, opt.net:clearState())
    
        -- local state_path = 'struct_models/state.t7'
        -- if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
        --   opt.net = opt.net:clearState()
        --   torch.save(state_path, opt)
        -- end
        -- saved = true
      end


      if batch_idx < 129 or batch_idx % math.floor((n_batches / 200)) == 0 then
        print(
          string.format('epoch=%2d | iter=%4d | loss=%9.6f, %9.6f, %9.6f, %9.6f', opt.epoch, batch_idx, table.unpack(f)) ..  ( saved and 'saved' or ''))
      end

      return f, grad_parameters
    end
    optimizer(feval, parameters, opt)
    -- xlua.progress(batch_idx, n_batches)
  end
end

function test_epoch(opt, data_loader)
  local net = opt.net or error('no net in train_epoch')
  local criterion = opt.criterion_test or error('no criterion in train_epoch')

  local n_batches = data_loader:n_batches()

  net:evaluate()

  local avg_f = 0
  local accuracy = 0
  local n_samples = 0
  for batch_idx = 1, n_batches do
    local _input, _target = data_loader:getBatch()
    local input = oc.FloatOctree():octree_create_from_dense_features_batch(_input, opt.tr_dist):cuda()
    local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()
    
    target:to_occupancy()
    
    local output = net:forward(input)
    
    avg_f = avg_f + criterion:forward(output[#output], target)
  end

  avg_f = avg_f / n_batches

  if(avg_f < opt.best_val) then
    opt.best_val = avg_f
    local net_path = 'struct_models/best.t7' --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
    torch.save(net_path, opt.net:clearState())

    local state_path = 'struct_models/state.t7'
    if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
      opt.net = opt.net:clearState()
      torch.save(state_path, opt)
    end
  end

  print(string.format('test_epoch=%d, avg_f=%f, best_val=%f', opt.epoch, avg_f, opt.best_val))
end

function worker(opt, train_data_loader, test_data_loader)
  local start_epoch = 1
  
  opt.min_loss = opt.min_loss or 1/0
  opt.best_val = opt.best_val or 1/0

  print(string.format('[INFO] start_epoch=%d', start_epoch))
  for epoch = start_epoch, opt.n_epochs do
    opt.epoch = epoch
    -- clean up
    opt.net:clearState()
    collectgarbage('collect')
    collectgarbage('collect')

    -- train
    print('[INFO] train epoch ' .. epoch .. ', lr=' .. opt.learningRate)
    train_epoch(opt, train_data_loader)

    -- save network
    if epoch % 20 == 0 then
      print('[INFO] saving progress')
      local net_path = string.format('struct_models/net_epoch%03d.t7', opt.epoch) --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      torch.save(net_path, opt.net:clearState())
  
      -- save state
      local state_path = 'struct_models/state.t7' --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
        opt.net = opt.net:clearState()
        torch.save(state_path, opt)
      end
      print('[INFO] progress saved to: ' .. net_path)
    end

    -- test_epoch(opt, test_data_loader)
    
    -- clean up
    collectgarbage('collect')
    collectgarbage('collect')


    -- adjust learning rate
    if opt.learningRate_steps[epoch] ~= nil then
      opt.learningRate = opt.learningRate * opt.learningRate_steps[epoch]
    end
  end

  -- test network
end

return {
  train_epoch = train_epoch,
  test_epoch = test_epoch,
  split_target = split_target,
  worker = worker
}
