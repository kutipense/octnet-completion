#!/usr/bin/env th

dataloader = {}

local f_ops = require("f_ops")
local torch = require("torch")

function BuildArray(...)
  local arr = {}
  for v in ... do
    table.insert(arr, v)
  end
  return arr
end


local DataLoader = torch.class('dataloader.DataLoader')

function DataLoader:__init(data_paths, batch_size, full_batches, split)
  assert(split == "val" or split == "overfit" or split == "train")
  self.split = split or error('')
  self.data_paths = data_paths or error('')

  self.batch_size = batch_size or error('')
  self.full_batches = full_batches or false
  self.items = BuildArray(io.open(string.format("/root/vol/octnet-completion/benchmark/%s.txt", self.split)):lines())
  self.n_samples = #self.items
  self.idx = 1
  assert(self.idx < self.n_samples, "idx should be smaller than the number of samples")
  assert(self.batch_size < self.n_samples, "Batch size should be smaller than the number of samples")
  
end

function DataLoader:getBatch()
    local bs = math.min(self.batch_size, self.n_samples - self.idx)

    local sdf_batch = torch.Tensor(bs, 32, 32, 32, 2):zero()
    local df_batch = torch.Tensor(bs, 32, 32, 32, 1)

    for batch_idx = 1, bs do
      local sdf_df_ids = {}
      self.idx = self.idx + 1
      local line = self.items[self.idx]
      for str in line:gmatch('[^%s]+') do
        table.insert(sdf_df_ids, str)
      end

      local sdf = f_ops.parse_sdf(f_ops.read_file(self.data_paths[1] .. "/" .. sdf_df_ids[1] .. ".sdf"))
      local df = f_ops.parse_df(f_ops.read_file(self.data_paths[2] .. "/" .. sdf_df_ids[2] .. ".df"))
      sdf_batch[batch_idx] = sdf
      df_batch[batch_idx] = df:view(32, 32, 32, 1)
    end

    collectgarbage(); collectgarbage()
    if self.n_samples - self.idx <= 0 then
      self.idx = 0
    end
    return sdf_batch, df_batch
end


function DataLoader:size()
  return self.n_samples
end

function DataLoader:n_batches()
  if self.full_batches then
    return math.floor(self.n_samples / self.batch_size)
  else
    return math.ceil(self.n_samples / self.batch_size)
  end
end

-- local val = dataloader.DataLoader({sdf_path, df_path}, 5, false, "val", 0.1)
-- print(val:getBatch())
return dataloader
