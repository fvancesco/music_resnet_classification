--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

-- USAGE
-- SINGLE FILE MODE
--          th extract-features.lua [MODEL] [FILE] ...
--
-- BATCH MODE
--          th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES] 
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'

if #arg < 2 then
   io.stderr:write('Usage (Single file mode): th extract-features.lua [MODEL] [FILE] ... \n')
   io.stderr:write('Usage (Batch mode)      : th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES]  \n')
   os.exit(1)
end

-- get the list of files
local list_of_filenames = {}
local batch_size = 1

if not paths.filep(arg[1]) then
    io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
    os.exit(1)
end
    
local path_list = ""
if tonumber(arg[2]) ~= nil then -- batch mode ; collect file from list ----- TODO
    
    local lfs  = require 'lfs'
    batch_size = tonumber(arg[2])
    path_list = arg[3]

else -- single file mode ; collect images from list
    path_list = arg[2]
end

--fill up table "list_of_files"
for line in io.lines(path_list) do
    --load the image as a RGB float tensor with values 0..1
    --local t = string.gmatch(line, '([^\t]+)')
    --local img_path = "/homedtic/fbarbieri/imagemusic/dataset/images/" .. t(0) .. ".jpg"
    local img_path = line
    if not paths.filep(img_path) then
        io.stderr:write('file not found: -->' .. img_path .. '<--\n')
        os.exit(1)
    else
       table.insert(list_of_filenames, img_path)
    end
end

local number_of_files = #list_of_filenames

if batch_size > number_of_files then batch_size = number_of_files end

-- Load the model
local model = torch.load(arg[1]):cuda()

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local features

for i=1,number_of_files,batch_size do
    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform 

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do 
        img_name = list_of_filenames[i+j-1]

        if img_name ~= nil then
            image_count = image_count + 1
            print(img_name)
            local img = image.load(img_name, 3, 'float')
            img = transform(img)
            img_batch[{j, {}, {}, {} }] = img
        end
    end

    -- if this is last batch it may not be the same size, so check that
    if image_count ~= batch_size then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

   -- Get the output of the layer before the (removed) fully connected layer
   local output = model:forward(img_batch:cuda()):squeeze(1)

   -- this is necesary because the model outputs different dimension based on size of input
   if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 

   if not features then
       features = torch.FloatTensor(number_of_files, output:size(2)):zero()
   end
       features[{ {i, i-1+image_count}, {}  } ]:copy(output)

end

local name = path_list:match( "([^/]+)$" )
local timestamp = os.clock()
local outputname = 'visual-' .. name .. '-' .. timestamp .. '.npy'
npy4th = require 'npy4th'
npy4th.savenpy(outputname, features)
--torch.save(outputname, {features=features, image_list=list_of_filenames})
print('saved features to ' .. outputname)
