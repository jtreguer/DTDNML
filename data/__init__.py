#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import numpy as np
import os
import torch
from .dataset import Dataset
import spectral

def get_spectral_responseOLD(data_name, srf_name):
    xls_path = os.path.join(os.getcwd(), data_name, srf_name + '.xls')
    print(xls_path)
    if not os.path.exists(xls_path):
        raise Exception("Spectral response path does not exist!")
    data = xlrd.open_workbook(xls_path)
    print(data.sheets())
    table = data.sheets()[0]
    num_cols = table.ncols
    num_cols_sta = 1
    print(table.col_values(0))
    print([np.array(table.col_values(i)).reshape(-1,1) for i in range(num_cols_sta,num_cols)])
    cols_list = [np.array(table.col_values(i)).astype(np.float32).reshape(-1,1) for i in range(num_cols_sta,num_cols)]
    sp_data = np.concatenate(cols_list, axis=1)
    # sp_data = sp_data.astype(float)
    sp_data = sp_data / (sp_data.sum(axis=0))
    return sp_data

def create_dataset(arg, sp_matrix, isTRain):
    dataset_instance = Dataset(arg, sp_matrix, isTRain)
    return dataset_instance

def get_sp_range(sp_matrix):
    HSI_bands, MSI_bands = sp_matrix.shape
    assert(HSI_bands>MSI_bands)
    sp_range = np.zeros([MSI_bands,2])
    for i in range(0,MSI_bands):
        index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
        sp_range[i,0] = index_dim_0[0]
        sp_range[i,1] = index_dim_0[-1]
    return sp_range

class DatasetDataLoader():
    def init(self, arg, isTrain=True):
        # JT
        header_file = os.path.join(os.getcwd(), arg.data_path_name, arg.data_img_name +'.hdr')
        header_spectral = spectral.open_image(header_file)
        self.wavelengths = header_spectral.bands.centers
        # JT
        self.sp_matrix = self.get_spectral_response(arg.data_path_name, arg.srf_name)
        self.sp_range = get_sp_range(self.sp_matrix)
        self.dataset = create_dataset(arg, self.sp_matrix, isTrain)
        self.hsi_channels = self.dataset.hsi_channels
        self.msi_channels = self.dataset.msi_channels
        self.lrhsi_height = self.dataset.lrhsi_height
        self.lrhsi_width  = self.dataset.lrhsi_width
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=arg.batchsize if isTrain else 1,
                                                      shuffle=arg.isTrain if isTrain else False,
                                                      num_workers=arg.nThreads if arg.isTrain else 0)
    def __len__(self):
        return len(self.dataset)
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

    def get_spectral_response(self,data_name, srf_name):
        xls_path = os.path.join(os.getcwd(), data_name, srf_name + '.xls')
        print(xls_path)
        if not os.path.exists(xls_path):
            raise Exception("Spectral response path does not exist!")
        data = xlrd.open_workbook(xls_path)
        srf = data.sheets()[0]
        srf_arr = np.array([srf.col_values(i) for i in range(srf.ncols)]).T
        sp_matrix = np.empty((len(self.wavelengths),srf.ncols-1),dtype=np.float32)
        print(srf.ncols)
        for i in range(1,srf.ncols): # start from 1 to exclude 1st column = wavelengths
            sp_matrix[:,i-1] = np.interp(self.wavelengths, srf_arr[:,0], srf_arr[:,i],left=0, right=0)
        print(f"sp_matrix.shape {sp_matrix.shape}")
        print(f"Min and max of sp_matrix {np.min(sp_matrix / sp_matrix.sum(axis=0))}, {np.max(sp_matrix/ sp_matrix.sum(axis=0))}")
        return sp_matrix / sp_matrix.sum(axis=0)


def get_dataloader(arg, isTrain=True):
    instant_dataloader = DatasetDataLoader()
    instant_dataloader.init(arg, isTrain)
    return instant_dataloader
