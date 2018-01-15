--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require '../datasets/transforms'
local imagenetLabel = require './imagenet'

if #arg < 2 then
   io.stderr:write('Usage: th classify.lua [MODEL] [PATH_FILE]...\n')
   os.exit(1)
end
for _, f in ipairs(arg) do
   if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
   end
end


-- Load the model
local model = torch.load(arg[1]):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

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

local N = 5

local path_list = arg[2]
for line in io.lines(path_list) do
   -- load the image as a RGB float tensor with values 0..1
   local t = string.gmatch(line, '([^\t]+)')
   local img_path = "/homedtic/fbarbieri/imagemusic/dataset/images/" .. t(0) .. ".jpg"
   local img = image.load(img_path, 3, 'float')
   local name = img_path:match( "([^/]+)$" )

   -- Scale, normalize, and crop the image
   img = transform(img)

   -- View as mini-batch of size 1
   local batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()

   -- Get the top 5 class indexes and probabilities
   local probs, indexes = output:topk(N, true, true)
   print('Classes for', img_path)
   for n=1,N do
     print(probs[n], imagenetLabel[indexes[n]])
   end
   print('')

end
