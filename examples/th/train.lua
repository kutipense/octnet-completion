#!/usr/bin/env th

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

      local input, _target = data_loader:getBatch()
      local input = oc.FloatOctree():octree_create_from_dense_features_batch(input, opt.tr_dist):cuda()
      -- local target = oc.FloatOctree():octree_create_from_dense_features_batch(_target, opt.tr_dist):cuda()
      _target = torch.log(torch.abs(_target) + 1)
      _target = _target:cuda()
      -- target:log_scale()
      local output = net:forward(input)
      local f = criterion:forward(output, _target)
      local dfdx = criterion:backward(output, _target):cuda()

      local f = criterion:forward(output[3], target)
      local dfdx = criterion:backward(output[3], target)

      net:backward(input, dfdx)

      -- print(f,f_p)

      local saved = false
      if(f < opt.min_loss) then
        opt.min_loss = f
        local net_path = 'models/best.t7' --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
        torch.save(net_path, opt.net:clearState())
    
        local state_path = 'models/state.t7'
        if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
          opt.net = opt.net:clearState()
          torch.save(state_path, opt)
        end
        saved = true
      end


      if batch_idx < 129 or batch_idx % math.floor((n_batches / 200)) == 0 then
        print(
          string.format('epoch=%2d | iter=%4d | loss=%9.6f ', opt.epoch, batch_idx, f) ..  ( saved and 'saved' or ''))
      end

      return f, grad_parameters
    end
    optimizer(feval, parameters, opt)
    -- xlua.progress(batch_idx, n_batches)
  end
end

function test_epoch(opt, data_loader)
  local net = opt.net or error('no net in test_epoch')
  local criterion = opt.criterion or error('no criterion in test_epoch')
  local data_loader = dataloader.DataLoader(opt.data_paths, opt.batch_size, opt.full_batches, "overfit", opt.tr_dist)
  local n_batches = data_loader:n_batches() --data_loader:n_batches()

  net:evaluate()

  local avg_f = 0
  local accuracy = 0
  local n_samples = 0
  for batch_idx = 1, n_batches do
    print(string.format('[INFO] test batch %d/%d', batch_idx, n_batches))

    local timer = torch.Timer()
    local input, target = data_loader:getBatch()
    local input = oc.FloatOctree():octree_create_from_dense_features_batch(input)
    input = input:cuda()
    input:clamp(opt.tr_dist)
    print(string.format('[INFO] loading data took %f[s] - n_batches %d', timer:time().real, target:size(1)))

    local timer = torch.Timer()
    local output = net:forward(input)
    local target = oc.FloatOctree():octree_create_from_dense_features_batch(target)
    target = target:cuda()
    output = output[{ { 1, target:size(1) }, {} }]
    local f = criterion:forward(output, target)
    print(string.format('[INFO] net/crtrn fwd took %f[s]', timer:time().real))
    avg_f = avg_f + f

    local maxs, indices = torch.max(output, 2)
    for bidx = 1, target:size(1) do
      if indices[bidx][1] == target[bidx] then
        accuracy = accuracy + 1
      end
      n_samples = n_samples + 1
    end
  end
  avg_f = avg_f / n_batches
  accuracy = accuracy / n_samples

  print(string.format('test_epoch=%d, avg_f=%f, accuracy=%f', opt.epoch, avg_f, accuracy))
end

function worker(opt, train_data_loader, test_data_loader)
  local start_epoch = 1
  
  opt.min_loss = opt.min_loss or 1/0

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
    print('[INFO] saving progress')
    if epoch % 10 == 0 then
      local net_path = string.format('models/net_epoch%03d.t7', opt.epoch) --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      torch.save(net_path, opt.net:clearState())
  
      -- save state
      local state_path = 'models/state.t7' --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
        opt.net = opt.net:clearState()
        torch.save(state_path, opt)
      end
      print('[INFO] progress saved to: ' .. net_path)
    end

    -- clean up
    collectgarbage('collect')
    collectgarbage('collect')

    -- adjust learning rate
    if opt.learningRate_steps[epoch] ~= nil then
      opt.learningRate = opt.learningRate * opt.learningRate_steps[epoch]
    end
  end

  -- test network
  -- test_epoch(opt, test_data_loader)
end

return {
  train_epoch = train_epoch,
  test_epoch = test_epoch,
  worker = worker
}
