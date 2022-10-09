import os, sys
import glob
import xarray as xr
import pandas as pd
import time
import numpy as np
import dask as da
import datetime as dt
import botocore, boto3
import matplotlib.pyplot as plt
import pyproj
import pyresample
#mport s3fs
import pickle


def get_filename_metadata(f):
    '''
    Extract metadata from ABI L1b filename.
    
    Parameters
    ----------
    f: str
        File path (s3 or local)
    
    Returns
    ----------
    dict(band, year, dayofyear, hour, minute, second, spatial)
    '''
    f = os.path.basename(f)
    band = int(f.split("_")[1][-2:])
    spatial = f.split("-")[2]
    t1 = f.split("_")[4]
    year = int(t1[1:5])
    dayofyear = int(t1[5:8])
    hour = int(t1[8:10])
    minute = int(t1[10:12])
    second = int(t1[12:15])
    datetime = dt.datetime(year, 1, 1, hour, minute, int(np.around(second/10) % 60)) + dt.timedelta(days=dayofyear-1)
    return dict(
        band=band,
        year=year,
        dayofyear=dayofyear,
        hour=hour,
        minute=minute,
        second=second,
        spatial=spatial,
        datetime=datetime,
    )


class L1bBand(object):
    '''
    Class to manipulate ABI L1b file. Functionality includes opening dataset,
    reprojection, plotting, and metadata.  
    '''
    def __init__(self, fpath):
        '''
        Parameters
        ----------
        fpath: str
            Filepath to read data, can be on S3 or local.
        '''
        self.fpath = fpath
        meta = get_filename_metadata(fpath)
        self.band = meta["band"]
        self.year = meta["year"]
        self.dayofyear = meta["dayofyear"]
        self.hour = meta["hour"]
        self.minute = meta["minute"]
        self.second = meta["second"]
        self.spatial = meta["spatial"]
        self.datetime = dt.datetime(
            self.year, 1, 1, self.hour, self.minute, self.second // 10
        ) + dt.timedelta(days=self.dayofyear - 1)

    def open_dataset(self, rescale=True, force=False, chunks=None):
        '''
        Open file from local disk or s3
        
        Parameters
        ----------
        rescale: bool (default=True)
            Rescale data based on metadata parameters.
        force: bool (default=False)
            Overwrite data currently in memory.
        chunks: dict (default=None)
        '''
        if (not hasattr(self, "data")) or force:
            if self.fpath[:3] == 's3:':
                s3 = s3fs.S3FileSystem()
                fobj = s3.open(self.fpath)
                ds = xr.open_dataset(fobj, engine='h5netcdf')
            else:
                ds = xr.open_dataset(self.fpath, chunks=chunks)
                
            ds_rad = ds["Rad"].where(ds.DQF.isin([0, 1]))
            band = ds.band_id[0]
            # normalize radiance
            if rescale:
                # radiance = ds['Rad']
                if band <= 6:
                    ds_rad = ds_rad * ds.kappa0
                else:
                    fk1 = ds.planck_fk1.values
                    fk2 = ds.planck_fk2.values
                    bc1 = ds.planck_bc1.values
                    bc2 = ds.planck_bc2.values
                    tmp = fk1 / ds_rad + 1
                    tmp = np.where(tmp > 1, tmp, np.nan)
                    T = (fk2 / (np.log(tmp)) - bc1) / bc2
                    ds_rad.values = T
                ds["Rad"] = ds_rad
            self.data = ds
        return self.data

    def plot(self, ax=None, cmap=None, norm=None):
        '''
        Plotting indvidual band.
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axes (optional)
        cmap: str (optional)
        norm: cmap normalizer (optional)
        
        Returns
        ----------
        matplotlib.pyplot.axes
        '''
        from mpl_toolkits.basemap import Basemap

        if not hasattr(self, "data"):
            self.open_dataset()

        # Satellite height
        sat_h = self.data["goes_imager_projection"].perspective_point_height
        # Satellite longitude
        sat_lon = self.data["goes_imager_projection"].longitude_of_projection_origin

        # The geostationary projection
        x = self.data["x"].values * sat_h
        y = self.data["y"].values * sat_h
        if ax is None:
            fig = plt.figure(figsize=[10, 10])
            ax = fig.add_subplot(111)

        m = Basemap(
            projection="geos",
            lon_0=sat_lon,
            resolution="i",
            rsphere=(6378137.00, 6356752.3142),
            llcrnrx=x.min(),
            llcrnry=y.min(),
            urcrnrx=x.max(),
            urcrnry=y.max(),
            ax=ax,
        )
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        ax.set_title(
            "GOES-16 -- Band {}".format(self.band), fontweight="semibold", loc="left"
        )
        ax.set_title("%s" % self.datetime.strftime("%H:%M UTC %d %B %Y"), loc="right")
        return m.imshow(self.data["Rad"].values[::-1], cmap=cmap, norm=norm)
        # return m

    def plot_infrared(
        self, ax=None, colortable="ir_drgb_r", colorbar=False, cmin=180, cmax=270
    ):
        from metpy.plots import colortables

        ir_norm, ir_cmap = colortables.get_with_range("WVCIMSS", cmin, cmax)
        # Use a colortable/colormap available from MetPy
        # ir_norm, ir_cmap = colortables.get_with_range(colortable, 190, 350)
        im = self.plot(ax, cmap=ir_cmap, norm=ir_norm)
        if colorbar:
            plt.colorbar(
                im,
                pad=0.01,
                aspect=50,
                ax=ax,
                shrink=0.85,
                ticks=range(cmin, cmax, 10),
                label="Temperature (K)",
            )
        return im

    def interp(self, scale):
        if not hasattr(self, "data"):
            self.open_dataset()

        new_x = np.linspace(
            self.data.x.values[0],
            self.data.x.values[-1],
            int(self.data.x.values.shape[0] * scale),
        )
        new_y = np.linspace(
            self.data.y.values[0],
            self.data.y.values[-1],
            int(self.data.y.values.shape[0] * scale),
        )
        self.data = self.data.interp(x=new_x, y=new_y)
        return self.data

    def latlon(self):
        if not hasattr(self, "lats"):
            if not hasattr(self, "data"):
                self.open_dataset()
            # Satellite height
            sat_h = self.data["goes_imager_projection"].perspective_point_height
            # Satellite longitude
            sat_lon = self.data["goes_imager_projection"].longitude_of_projection_origin
            sat_sweep = self.data["goes_imager_projection"].sweep_angle_axis
            
            h, w = self.data["Rad"].shape
            cache_file = os.path.join(
                os.path.dirname(__file__),
                f"latlon_{self.spatial}_{sat_lon:3.3f}_{h}x{w}.pkl",
            )
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as fp:
                    cache = pickle.load(fp)
                self.lats = cache['lats']
                self.lons = cache['lons']
                
            else:
                p = pyproj.Proj(proj="geos", h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
                X = self.data["x"].values * sat_h
                Y = self.data["y"].values * sat_h
                XX, YY = np.meshgrid(X, Y)
                lons, lats = p(XX, YY, inverse=True)
                self.lats = lats
                self.lons = lons
                NANs = np.isnan(self.data["Rad"].values)
                self.lats[NANs] = np.nan
                self.lons[NANs] = np.nan
                self.lats[~np.isfinite(self.lats)] = np.nan
                self.lons[~np.isfinite(self.lons)] = np.nan
                cache = dict(lats=self.lats, lons=self.lons)
                with open(cache_file, "wb") as fp:
                    pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)  
              
        return self.lats, self.lons

    def latlon_lookup(self, lat, lon):
        self.latlon()
        if (
            (lat > self.lats.min())
            and (lat < self.lats.max())
            and (lon > self.lons.min())
            and (lon < self.lons.max())
        ):
            dist = ((self.lats - lat) ** 2 + (self.lons - lon) ** 2) ** 0.5
            ix, iy = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            return ix, iy

    def reproject_to_latlon(self, chunks=None, bounds=None):
        """
        args:
            chunks: dask chunk size by dimension eg. dict(lat=500,lon=500)
            bounds: [lat_ul, lon_ul, lat_lr, lon_lr]
        """
        s = 0.02
        t0 = time.time()
        
        data = self.open_dataset(chunks=chunks)
        if self.band in [1, 3, 5]:
            data = self.interp(0.5)
            orig_res = 1
        elif self.band == 2:
            # s = 0.005
            data = self.interp(0.25)
            orig_res = 0.5
        else:
            orig_res = 2
            
        lats, lons = self.latlon()
        
        lats = lats.astype(np.float32)
        lons = lons.astype(np.float32)
        
        rad = data['Rad'].values
        
        sat_lon = data["goes_imager_projection"].longitude_of_projection_origin
        
        if not bounds:
            lat_min = np.around(max(np.nanmin(lats), -60), 3)
            lat_max = np.around(min(np.nanmax(lats), 60), 3)
            lon_min = np.around(max(np.nanmin(lons), sat_lon - 60), 3)
            lon_max = np.around(min(np.nanmax(lons), sat_lon + 60), 3)
        else:
            lat_min = bounds[2]
            lat_max = bounds[0]
            lon_min = bounds[1]
            lon_max = bounds[3]
            
            # crop unprojected data
           # top = np.where(np.any(lat_max >= lats, axis=1))[0][0]
           # bottom = np.where(np.any(lat_min <= lats, axis=1))[0][-1]

         #   left = np.where(np.any(lon_min <= lons, axis=0))[0][0]
         #   right = np.where(np.any(lon_max >= lons, axis=0))[0][-1]
            
         #   lats = lats[top:bottom,left:right]
          #  lons = lons[top:bottom,left:right]
            
          #  rad = rad[top:bottom,left:right]
            
            #print(lats, lons)


        lats_new = np.arange(lat_min, lat_max, s).astype(np.float32)
        lons_new = np.arange(lon_min, lon_max, s).astype(np.float32)
        lons_new, lats_new = np.meshgrid(lons_new, lats_new)
        h, w = lats_new.shape

        source_def = pyresample.geometry.SwathDefinition(lats=lats, lons=lons)
        target_def = pyresample.geometry.GridDefinition(lons=lons_new, lats=lats_new)
        neighbor_cache_file = os.path.join(
            os.path.dirname(__file__),
            f"neighbors_{self.spatial}_{sat_lon:3.3f}_{lat_min:3.3f}_{lon_min:3.3f}_{orig_res}_{h}x{w}.pkl",
        )
        if os.path.exists(neighbor_cache_file):
            with open(neighbor_cache_file, "rb") as fp:
                neighbor_info = pickle.load(fp)
        else:
            neighbor_info = pyresample.kd_tree.get_neighbour_info(
                source_def, target_def, 50000, epsilon=0.02, neighbours=1
            )

            if not bounds:
                with open(neighbor_cache_file, "wb") as fp:
                    pickle.dump(neighbor_info, fp, protocol=pickle.HIGHEST_PROTOCOL)
        result = pyresample.kd_tree.get_sample_from_neighbour_info(
            "nn",
            target_def.shape,
            rad.astype(np.float32),
            neighbor_info[0],
            neighbor_info[1],
            neighbor_info[2],
            distance_array=neighbor_info[3],
            fill_value=np.nan
        )

        data_new = xr.DataArray(
            result,
            coords=dict(lat=lats_new[:, 0], lon=lons_new[0, :]),
            dims=("lat", "lon"),
        )
        ds = xr.Dataset(dict(Rad=data_new))
        return ds    

class GOESS3(object):
    def __init__(
        self,
        product="ABI-L1b-RadF",
        bands=range(1, 17),
        sensor="goes16",
        spatial=None,
        cache_dir=os.path.dirname(os.path.abspath(__file__)),
    ):

        self.product = product
        self.bands = bands
        self.sensor = sensor
        self.spatial = spatial
        self.cache_dir = cache_dir

        if spatial is None:
            if self.product == "ABI-L1b-RadF":
                self.spatial = "RadF"
                self.timestep = 10
            elif self.product == "ABI-L1b-RadC":
                self.spatial = "RadC"
                self.timestep = 5
            elif self.product == "ABI-L1b-RadM":
                self.spatial = "RadM1"
                self.timestep = 1

        self.bucket_name = f"noaa-{sensor}"
        self._connect_to_s3()

    def _connect_to_s3(self):
        res = boto3.resource("s3")
        self.bucket = res.Bucket(self.bucket_name)

    def get_keys_by_time(self, t):
        "https://noaa-goes16.s3.amazonaws.com/ABI-L1b-RadF/2020/001/00/OR_ABI-L1b-RadF-M6C01_G16_s20200010000216_e20200010009524_c20200010009588.nc"
        prefix = (
            os.path.join(
                self.product,
                "%04i" % t.year,
                "%03i" % t.timetuple().tm_yday,
                "%02i" % t.hour,
            )
            + "/"
        )            
            
        key_list = []
        for obj in self.bucket.objects.filter(Prefix=prefix):
            if obj.key[-3:] == ".nc":
                v = get_filename_metadata(obj.key)
                tstep = self.timestep
                                    
                minute_diff = (v['datetime'] - t).total_seconds() / 60
                
                if np.abs(minute_diff) > (tstep / 2):
                    continue
                if v["band"] not in self.bands:
                    continue
                v["key"] = obj.key
                key_list.append(v)
       
        tstep = self.timestep     
        if (t.minute < tstep) and (len(key_list) == 0):
            prevhour = t - dt.timedelta(hours=1)
            prefix = (
                os.path.join(
                    self.product,
                    "%04i" % prevhour.year,
                    "%03i" % prevhour.timetuple().tm_yday,
                    "%02i" % prevhour.hour,
                )
                + "/"
            )   
            #trying previous hour prefix
            key_list = []
            for obj in self.bucket.objects.filter(Prefix=prefix):
                if obj.key[-3:] == ".nc":
                    v = get_filename_metadata(obj.key)
                    
                    minute_diff = (v['datetime'] - t).total_seconds() / 60
                    if np.abs(minute_diff) > (tstep / 2):
                        continue
                    if v["band"] not in self.bands:
                        continue
                    v["key"] = obj.key
                    key_list.append(v)
                
        keys = pd.DataFrame(key_list)
        return keys

    def download_file(self, key):
        output_file = os.path.join(self.cache_dir, key)
        if os.path.exists(output_file):
            return output_file

        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.bucket.download_file(key, output_file)
        # sys.stdout.write(f"Wrote to {output_file}\n")
        return output_file

    def read_file_from_key(self, key):
        file = os.path.join(self.cache_dir, key)
        if not os.path.exists(file):
            file = self.download_file(key)
        return file