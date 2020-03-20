# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:37:23 2020

@author: Rully
"""

class Excel:
    
    from openpyxl import load_workbook
    
    workbook = load_workbook(filename = 'C:/Akademik/TA/rbf_keras-master/Hasil.xlsx')
    sheet = workbook.active
    
    def __init__(self, time_total, MAPE, RMSE):
        self.time_total = time_total
        self.MAPE = MAPE
        self.RMSE = RMSE
        
    def sheet_code(self, row):
        self.sheet['B'+str(row)] = self.time_total
        self.sheet['C'+str(row)] = self.MAPE
        self.sheet['D'+str(row)] = self.RMSE
        
        self.workbook.save(filename = 'C:/Akademik/TA/rbf_keras-master/Hasil.xlsx')