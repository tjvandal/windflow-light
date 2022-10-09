import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import glob

import numpy as np
import pandas as pd
import torch
from torch import nn
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats

from .datasets import goesr, stats
from .networks.models import get_flow_model
from .datasets.preprocess import image_histogram_equalization
from .datasets.utils import cartesian_to_speed

def split_array(arr, tile_size, overlap):
    c, h, w = arr.shape
    patches = []
    indicies = []
    for i in range(0, h, tile_size-overlap):
        for j in range(0, w, tile_size-overlap):
            i = min(i, h - tile_size)
            j = min(j, w - tile_size)
            indicies.append([i, j])
            patches.append(arr[np.newaxis,:,i:i+tile_size, j:j+tile_size])
    indices = np.array(indicies)
    patches = np.concatenate(patches, axis=0)
    return patches, indicies

def reassemble_split_array(arr, upperleft, shape, trim=0):    
    assert len(arr) > 0

    # perform inference on patches
    height, width = arr.shape[2:4]
    counter = np.zeros(shape)
    out_sum = np.zeros(shape)
    for i, x in enumerate(arr):
        ix, iy = upperleft[i]  
        if trim > 0:
            counter[:,ix+trim:ix+height-trim,iy+trim:iy+width-trim] += 1
            out_sum[:,ix+trim:ix+height-trim,iy+trim:iy+width-trim] += x[:,trim:-trim,trim:-trim]    
        else:
            counter[:,ix:ix+height,iy:iy+width] += 1
            out_sum[:,ix:ix+height,iy:iy+width] += x

    out = out_sum / counter
    return out

def reassemble_with_2d_gaussian(arr, upperleft, shape, trim=0):    
    assert len(arr) > 0

    # perform inference on patches
    height, width = arr.shape[2:4]
    counter = np.zeros(shape)
    out_sum = np.zeros(shape)
    
    nsig = 3
    x = np.linspace(-nsig, nsig, height+1-trim*2)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    pdf = kern2d/kern2d.sum()
    
    for i, x in enumerate(arr):
        ix, iy = upperleft[i]  
        if trim > 0:
            counter[:,ix+trim:ix+height-trim,iy+trim:iy+width-trim] += pdf
            out_sum[:,ix+trim:ix+height-trim,iy+trim:iy+width-trim] += x[:,trim:-trim,trim:-trim] * pdf  
        else:
            counter[:,ix:ix+height,iy:iy+width] += pdf
            out_sum[:,ix:ix+height,iy:iy+width] += x * pdf

    out = out_sum / counter
    return out


def write_to_netcdf(filename, ua, va, lat, lon, ts,
                    pressure=None):
    
    ua = xr.DataArray(ua.astype(np.float32), coords=[('pressure', pressure),('lat', lat), ('lon', lon)])
    va = xr.DataArray(va.astype(np.float32), coords=[('pressure', pressure),('lat', lat), ('lon', lon)])
    newds = xr.Dataset(dict(ua=ua, va=va, time=ts, pressure=pressure))
    newds.to_netcdf(filename)
    print("Dataset written to file: {}".format(filename))


def inference(model, X0, X1, 
              tile_size=512, 
              overlap=128, 
              upsample_flow_factor=None,
              batch_size=32): 
    c, h, w = X0.shape
    trim = 0 #overlap // 4
    
    if isinstance(upsample_flow_factor, int):
        upsample_flow = nn.Upsample(scale_factor=upsample_flow_factor, mode='bilinear')
    else:
        upsample_flow = None
        
    #if isinstance(upsample_input, float)
    
    f_sum = np.zeros((1, 2, h, w))
    f_counter = np.zeros((1, 1, h, w))
    
    x0_patches, upperleft = split_array(X0, tile_size, overlap)
    x1_patches, _ = split_array(X1, tile_size, overlap)
    
    x0_patches = torch.from_numpy(x0_patches).float()
    x1_patches = torch.from_numpy(x1_patches).float()
    pred = []
    for batch in range(0, x1_patches.shape[0], batch_size):
        x0_batch = x0_patches[batch:batch+batch_size]
        x1_batch = x1_patches[batch:batch+batch_size]

        if next(model.parameters()).is_cuda:
            x0_batch = x0_batch.cuda()
            x1_batch = x1_batch.cuda()

        # testing 2x upsampling before inference
        #upsample_2x = nn.Upsample(scale_factor=2, mode='bilinear')
        #x0_batch = upsample_2x(x0_batch)
        #x1_batch = upsample_2x(x1_batch)

        #try:
        model_out = model(x0_batch, x1_batch, test_mode=True)[0]
        #except TypeError:
            #model_out = model(x0, x1)[0]
        #    model_out = model(torch.cat([x0_batch, x1_batch], 1))[0] 
            
        if upsample_flow:
            model_out = upsample_flow(model_out) #* upsample_flow_factor
            #model_out = upsample_2x(model_out) / 2
            
        pred.append(model_out.cpu().detach().numpy())

    pred = np.concatenate(pred, 0)   
    #UV = reassemble_split_array(pred, upperleft, (2, h, w), trim=trim)
    UV = reassemble_with_2d_gaussian(pred, upperleft, (2, h, w), trim=trim)

    return UV


class FlowRunner(object):
    '''
    Operator to perform inference on general inputs and flow models
    '''
    def __init__(self, 
                 model_name, 
                 tile_size=512, 
                 overlap=384, 
                 batch_size=8, 
                 upsample_input=None,
                 device='cuda:0'):        
        self.model = get_flow_model(model_name, small=False)
        self.model_name = model_name
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.device = device

        # upsample input 2x can improve feature tracking at inference time
        #if upsample_input_factor is not None:
        #    self.upsample_input = nn.Upsample(scale_factor=upsample_input_factor, mode='bilinear')
        #else:
        #    upsample_input_factor = 1
        
        if self.model_name.lower() in ['flownets', 'pwcnet', 'pwc-net', 'maskflownet']:
            self.upsample_flow_factor = 4 #/ upsample_input_factor
        else:
            self.upsample_flow_factor = None #1 / upsample_input_factor
            
        self.model = self.model.to(device)
        #self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        #self.model = torch.nn.DataParallel(self.model, device_ids=[int(list(device)[-1])])
        #self.model = torch.nn.DataParallel(self.model, device_ids=[0,])

    def load_checkpoint(self, checkpoint_file):

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.global_step = checkpoint['global_step']
        #for key in checkpoint['model']:
        #    if 'module' in key:
        #        new_key = key.replace('module.', '')
        #        checkpoint[new_key] = checkpoint[key]
        #        del checkpoint[key]

        try:
            self.model.module.load_state_dict(checkpoint['model'])
        except:
            self.model.load_state_dict(checkpoint['model'])
                
        print(f"Loaded checkpoint: {checkpoint_file}")

    def preprocess(self, x):
        x[~np.isfinite(x)] = 0.
        x = image_histogram_equalization(x)
        return x    

    def forward(self, img1, img2):
        mask = (img1 == img1)
        mask[~mask] = np.nan

        img1_norm = self.preprocess(img1)
        img2_norm = self.preprocess(img2)

        flows = inference(self.model, img1_norm[np.newaxis], img2_norm[np.newaxis], 
                          tile_size=self.tile_size, overlap=self.overlap, 
                          upsample_flow_factor=self.upsample_flow_factor,
                          batch_size=self.batch_size)
        return img1, flows * mask[np.newaxis]



class GeoFlows(FlowRunner):
    '''
    Object to manage optical flow prediction for geostationary L1b observations
    '''
    def __init__(self, model_name, 
                 data_directory, 
                 product='ABI-L1b-RadF', 
                 upsample_data=None,
                 channels=[10], 
                 timestep=10, 
                 spatial=None, 
                 **kwargs):
        FlowRunner.__init__(self, model_name, **kwargs)
        self.data_directory = data_directory
        self.timestep = timestep
        self.upsample_data = upsample_data
        self.spatial = spatial
        self.goes = goesr.GOESL1b(product=product, channels=channels, 
                                  data_directory=data_directory)
        self.channels = channels

    def flows_by_time(self, t, reproject=False):
        t2 = t + dt.timedelta(minutes=self.timestep)
        file1 = self.goes.snapshot_file(t.year, t.timetuple().tm_yday, t.hour, 
                                        t.minute, spatial=self.spatial).values[0]
        file2 = self.goes.snapshot_file(t2.year, t2.timetuple().tm_yday, t2.hour, 
                                        t2.minute, spatial=self.spatial).values[0]

        obj1 = goesr.L1bBand(file1)
        obj2 = goesr.L1bBand(file2)
        
        if self.upsample_data is not None:
            obj1.interp(self.upsample_data)
            obj2.interp(self.upsample_data)
            
        lats, lons = obj1.latlon()

        if reproject:
            data1 = obj1.reproject_to_latlon()
            data2 = obj2.reproject_to_latlon()
        else:
            data1 = obj1.open_dataset()
            data2 = obj2.open_dataset()
         
        img1 = data1['Rad'].values
        img2 = data2['Rad'].values

        flows = self.forward(img1, img2)[1]
        
        if reproject:
            data1['U'] = xr.DataArray(flows[0], dims=['lat', 'lon'], 
                                      coords=dict(lat=data1.lat.values, lon=data1.lon.values))
            data1['V']= xr.DataArray(flows[1], dims=['lat', 'lon'], 
                                      coords=dict(lat=data1.lat.values, lon=data1.lon.values))
        else:
            data1['lat'] = xr.DataArray(lats, dims=['y', 'x'], 
                                        coords=dict(y=data1.y.values, x=data1.x.values))
            data1['lon'] = xr.DataArray(lons, dims=['y', 'x'],
                                        coords=dict(y=data1.y.values, x=data1.x.values))

            data1['U'] = xr.DataArray(flows[0], dims=['y', 'x'], 
                    coords=dict(y=data1.y.values, x=data1.x.values))
            data1['V']= xr.DataArray(flows[1], dims=['y', 'x'], 
                    coords=dict(y=data1.y.values, x=data1.x.values))

        data1 = cartesian_to_speed(data1)
        data1['U'] *= 2000 / (self.timestep * 60) * 0.8 # m/s on 2km grid -- 0.8 scales to remove bias
        data1['V'] *= 2000 / (self.timestep * 60) * 0.8 # m/s on 2km grid
        return data1
    
    
    '''def flows_iterate(self, start_year, start_dayofyear, start_hour, steps=6*24):
        
        files = self.goes.local_files(start_year, start_dayofyear).reset_index()
        files = files[files['hour'] >= start_hour]
        
        curr_date = dt.datetime(start_year, 1, 1, 1) + dt.timedelta(days=start_dayofyear-1)
        while len(files) < steps:
            curr_date += dt.timedelta(days=1)
            more_files = self.goes.local_files(curr_date.year, curr_date.timetuple().tm_yday).reset_index()
            files = pd.concat([files, more_files])
        
        band1 = goesr.L1bBand(files['file'].values[0,0])
        data1 = band1.open_dataset()
        img1 = data1['Rad'].values
        
        for idx in range(1, steps):
            band2 = goesr.L1bBand(files['file'].values[idx,0])
            data2 = band2.open_dataset()
            img2 = data2['Rad'].values

            flows = self.forward(img1, img2)[1]
            
            yield band1, flows
            
            band1 = band2
            data1 = data2
            img1 = img2
    '''      
         
if __name__ == "__main__":

    data_directory = '/nex/datapool/geonex/public/GOES16/NOAA-L1B/'
    model = FlowNetS.FlowNetS(None, 1)
    checkpoint_path = '/nobackupp10/tvandal/nex-ai-opticalflow/scripts/experiments/gmao_osse/flownets-size_512-lognorm/checkpoint.flownet.pth.tar'
    model.training = False

    inference = inference_flows.GeoFlows(model, data_directory)

    
