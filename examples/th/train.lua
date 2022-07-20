#!/usr/bin/env th

function train_epoch(opt, inputs, outputs)
    local net = opt.net or error('no net in train_epoch')
    local criterion = opt.criterion or error('no criterion in train_epoch')
    local optimizer = opt.optimizer or error('no optimizer in train_epoch')
    local n_batches = 1 --data_loader:n_batches()
  
    net:training()
  
    local parameters, grad_parameters = net:getParameters()
    for batch_idx = 1, n_batches do
      local feval = function(x)
        if x ~= parameters then parameters:copy(x) end
        grad_parameters:zero()
  
        -- local input, target = data_loader:getBatch()
        -- print((batch_idx-1)*opt.batch_size+1,batch_idx*opt.batch_size)
        local input = inputs[{{(batch_idx-1)*opt.batch_size+1,batch_idx*opt.batch_size},}]
        local input = oc.FloatOctree():octree_create_from_dense_features_batch(input)
        input = input:cuda()
        local target = outputs[{{(batch_idx-1)*opt.batch_size+1,batch_idx*opt.batch_size},}]
        local target = oc.FloatOctree():octree_create_from_dense_features_batch(target)
        target = target:cuda()

        local output = net:forward(input)
        local f = criterion:forward(output, target)
        local dfdx = criterion:backward(output, target)

        print(f)

        net:backward(input, dfdx)
        
        if batch_idx < 129 or batch_idx % math.floor((n_batches / 200)) == 0 then 
          print(string.format('epoch=%2d | iter=%4d | loss=%9.6f ', opt.epoch, batch_idx, f))
        end
        
        return f, grad_parameters
      end 
      optimizer(feval, parameters, opt)
      xlua.progress(batch_idx, n_batches)
    end 
  end
  
  function test_epoch(opt, data_loader)
    local net = opt.net or error('no net in test_epoch')
    local criterion = opt.criterion or error('no criterion in test_epoch')
    local n_batches = data_loader:n_batches()
  
    net:evaluate()
  
    local avg_f = 0
    local accuracy = 0
    local n_samples = 0
    for batch_idx = 1, n_batches do
      print(string.format('[INFO] test batch %d/%d', batch_idx, n_batches))
  
      local timer = torch.Timer()
      local input, target = data_loader:getBatch()
      print(string.format('[INFO] loading data took %f[s] - n_batches %d', timer:time().real, target:size(1)))
  
      local timer = torch.Timer()
      local output = net:forward(input)
      output = output[{{1,target:size(1)}, {}}]
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

  
function worker(opt, inputs, outputs)
    local start_epoch = 1

    print(string.format('[INFO] start_epoch=%d', start_epoch))
    for epoch = start_epoch, 1 do
      opt.epoch = epoch
      
      -- clean up
      opt.net:clearState()
      collectgarbage('collect')
      collectgarbage('collect')
  
      -- train
      print('[INFO] train epoch '..epoch..', lr='..opt.learningRate)
    --   opt.data_fcn = opt.train_data_fcn
      train_epoch(opt, inputs, outputs)
       
      -- save network
      print('[INFO] save net')
      local net_path = string.format('models/net_epoch%03d.t7', opt.epoch) --paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      torch.save(net_path, opt.net:clearState())
      print('[INFO] saved net to: ' .. net_path)
  
      -- save state
      local state_path = 'models/state.t7'--paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
      if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
        print('[INFO] save state')
        opt.net = opt.net:clearState()
        torch.save(state_path, opt)
        print('[INFO] saved state to: ' .. state_path)
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
    test_epoch(opt, test_data_loader)
  end

return {
    train_epoch = train_epoch,
    test_epoch = test_epoch,
    worker = worker
}