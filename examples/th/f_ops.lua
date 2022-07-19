#!/usr/bin/env th
local ffi = require("ffi")
local torch = require('torch')

-- Helper function to calculate file size.
local function filesize(fd)
    local current = fd:seek()
    local size = fd:seek("end")
    fd:seek("set", current)
    return size
end

local function read_file(file_name)
    local fd, err = io.open(file_name, "rb")
    if err then error(err) end

    local size = filesize(fd)
    local header_size = 3 -- number
    local data_size = (size - header_size * 8) / 4
    local data_bsize = data_size * ffi.sizeof('float') --bytes
    local header_bsize = header_size * ffi.sizeof('uint64_t')

    local header = ffi.new("uint64_t[?]", header_size)
    local data = ffi.new("float[?]", data_size)

    ffi.copy(header, fd:read(header_bsize), header_bsize)
    ffi.copy(data, fd:read(data_bsize), data_bsize)
    fd:close()

    tensor_file = torch.FloatTensor(data_size)
    ffi.copy(tensor_file:data(), data, data_size * ffi.sizeof('float'))
    tensor_file = tensor_file:view(
        tonumber(header[0]), tonumber(header[1]), tonumber(header[2]), 1)

    return tensor_file
end

local function parse_sdf(input_sdf, tr_dist)
    local sdf = torch.clamp(input_sdf, -tr_dist, tr_dist)
    return torch.cat(torch.abs(sdf), torch.sign(sdf), 4)

end

local function parse_df(input_df, tr_dist)
    return torch.squeeze(torch.exp(torch.add(input_df, 1)),4)
end

return {
    read_file = read_file,
    parse_sdf = parse_sdf,
    parse_df = parse_df,
}

-- local filename = "10155655850468db78d106ce0a280f87__0__.sdf"
-- local sdf = read_file(filename)
-- df = parse_df(sdf, 0.1)
-- sdf = parse_sdf(sdf, 0.1)

-- print(sdf:size())
-- print(df:size())

