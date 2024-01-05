from tkinter import TOP
from matplotlib.cbook import flatten
from tqdm import tqdm
from osgeo import gdal,ogr,osr
import numpy as np
import cv2
import argparse
import os


def read_tif(tif_path):
    ds = gdal.Open(tif_path)
    row = ds.RasterXSize
    col = ds.RasterYSize
    band = ds.RasterCount

    for i in range(band):
        data = ds.GetRasterBand(i+1).ReadAsArray()

        data = np.expand_dims(data , 2)
        if i == 0:
            allarrays = data
        else:
            allarrays = np.concatenate((allarrays, data), axis=2)
    # Top left point coordinates; GeoTransform[0],GeoTransform[3] Transform[1] is the pixel width, and Transform[5] is the pixel height
    return {'data':allarrays,'transform':ds.GetGeoTransform(),'projection':ds.GetProjection(),'bands':band,'width':row,'height':col}

def write_tif(fn_out, im_data, transform,proj=None):
    # Set the projection, proj is wkt format
    if proj is None:
        proj = 'GEOGCS["WGS 84",\
                     DATUM["WGS_1984",\
                             SPHEROID["WGS 84",6378137,298.257223563, \
                                    AUTHORITY["EPSG","7030"]], \
                             AUTHORITY["EPSG","6326"]], \
                     PRIMEM["Greenwich",0, \
                            AUTHORITY["EPSG","8901"]], \
                     UNIT["degree",0.0174532925199433, \
                            AUTHORITY["EPSG","9122"]],\
                     AUTHORITY["EPSG","4326"]]'
    # Set data type
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # Adjust the sequence (C, H, W) to (H, W, C)
    # print('shape of im data:', im_data.shape)
    im_bands = min(im_data.shape)
    im_shape = list(im_data.shape)
    im_shape.remove(im_bands)
    im_height, im_width = im_shape
    band_idx = im_data.shape.index(im_bands)
    # Find out which band is on

    # Create the file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fn_out, im_width, im_height, im_bands, datatype)

    # if dataset is not None:
    dataset.SetGeoTransform(transform)  # Writes affine transformation parameters
    dataset.SetProjection(proj)  # Write projection

    if im_bands == 1:
        # print(im_data[:, 0,:].shape)
        if band_idx == 0:
            dataset.GetRasterBand(1).WriteArray(im_data[0, :, :])
        elif band_idx == 1:
            dataset.GetRasterBand(1).WriteArray(im_data[:, 0, :])
        elif band_idx == 2:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])

    else:
        for i in range(im_bands):
            if band_idx == 0:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
            elif band_idx == 1:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
            elif band_idx == 2:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

    dataset.FlushCache()
    del dataset
    driver = None

def extract_nDSM(args):
    input_path = args.input_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file path '{input_path}' does not exist.")
    output_path = args.output_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file path '{output_path}' does not exist.")

    tif = read_tif(input_path)
    tif_data = tif["data"][:, :, 0]

    TOB = np.empty((tif['height'], tif['width']))
    BOB = np.empty((tif['height'], tif['width']))

    window_size = 15
    window_len = int((window_size-1)/2)

    use_kernel = 0
    if use_kernel == 1:
        # Gaussian_Kernel = cv2.getGaussianKernel(window_size, sigma = 1) * cv2.getGaussianKernel(window_size, sigma = 1).T
        kernel = (3,3)
        tif_data = cv2.GaussianBlur(tif_data, kernel, 1, 1) 

    start_row = window_len
    end_row = int(tif['height']-window_len)
    start_col = window_len
    end_col = int(tif['width']-window_len)

    # Do not process the edges to reduce the time required for judgment
    for i in tqdm(range(start_row, end_row), desc= "inside TOB&BOB"):
        for j in range(start_col, end_col):
            window_subset = tif_data[i-window_len:i+window_len+1, j-window_len:j+window_len+1]
            window_max = np.max(window_subset.flatten())
            # window_min = np.min(window_subset.flatten())
            window_min_group = np.sort(window_subset.flatten())[0:10]

            # Remove outliers
            group_mean = np.mean(window_min_group)
            group_std = np.std(window_min_group)
            window_min_final = []
            for q in range(len(window_min_group)):
                temp = group_mean - window_min_group[q] # Only the minimum is detected
                if temp <= 3 * group_std:
                    window_min_final.append(window_min_group[q])
            window_min = np.mean(window_min_final)

            # window_min = np.mean(sorted(window_subset.flatten())[0:25]) 
            TOB[i][j] = window_max
            BOB[i][j] = window_min

    # Process edges
    for i in tqdm(range(start_row, end_row)):
        window_subset = tif_data[i-window_len:i+window_len+1, 0:window_len+1]
        window_subset_max = np.max(window_subset.flatten())
        window_subset_min = np.min(window_subset.flatten())
        TOB[i][0] = window_subset_max
        BOB[i][0] = window_subset_min

        window_subset = tif_data[i-window_len:i+window_len+1, tif['width']-1-window_len:tif['width']]
        window_subset_max = np.max(window_subset.flatten())
        window_subset_min = np.min(window_subset.flatten())
        TOB[i][tif['width']-1] = window_subset_max
        BOB[i][tif['width']-1] = window_subset_min

    for j in range(start_col, end_col):
        window_subset = tif_data[0:window_len+1, j-window_len:j+window_len+1]
        window_subset_max = np.max(window_subset.flatten())
        window_subset_min = np.min(window_subset.flatten())
        TOB[0][j] = window_subset_max
        BOB[0][j] = window_subset_min

        window_subset = tif_data[tif['height']-1-window_len:tif['height'], j-window_len:j+window_len+1]
        window_subset_max = np.max(window_subset.flatten())
        window_subset_min = np.min(window_subset.flatten())
        TOB[tif['height']-1][j] = window_subset_max
        BOB[tif['height']-1][j] = window_subset_min

    # Process corners
    TOB[0][0] = np.max(tif_data[0:window_len, 0:window_len].flatten())
    TOB[0][tif['width']-1] = np.max(tif_data[0:window_len, tif['width']-1-window_len:tif['width']-1].flatten())
    TOB[tif['height']-1][0] = np.max(tif_data[tif['height']-1-window_len:tif['height']-1, 0:window_len].flatten())
    TOB[tif['height']-1][tif['width']-1] = np.max(tif_data[tif['height']-1-window_len:tif['height']-1, tif['width']-1-window_len:tif['width']-1].flatten())
    BOB[0][0] = np.min(tif_data[0:window_len, 0:window_len].flatten())
    BOB[0][tif['width']-1] = np.min(tif_data[0:window_len, tif['width']-1-window_len:tif['width']-1].flatten())
    BOB[tif['height']-1][0] = np.min(tif_data[tif['height']-1-window_len:tif['height']-1, 0:window_len].flatten())
    BOB[tif['height']-1][tif['width']-1] = np.min(tif_data[tif['height']-1-window_len:tif['height']-1, tif['width']-1-window_len:tif['width']-1].flatten())

    # TOB_out = TOB[:, :]
    # BOB_out = BOB[:, :]
    nDSM_TB = TOB-BOB
    nDSM1_TB_out = nDSM_TB[:, :, np.newaxis]
    nDSM_DB = tif_data[:, :] - BOB
    nDSM2_DB_out = nDSM_DB[:, :, np.newaxis]

    # write_tif(fn_nDSM1_out, nDSM1_out, tif['transform'], tif['projection'])
    write_tif(output_path, nDSM2_DB_out, tif['transform'], tif['projection'])

    print('ok')

def main():
    parser = argparse.ArgumentParser(description="Input & Output path")
    parser.add_argument(
        "--input_path",
        default="",
        help="path for input aw3d30",
        type=str
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="path for output nDSM",
        type=str
    )
    parser.add_argument(
        "--use_kerner", default=False, type = bool
    )
    args = parser.parse_args()
    extract_nDSM(args)


if __name__ == "__main__":
    main()