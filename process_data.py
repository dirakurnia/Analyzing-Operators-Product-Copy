import re
import numpy as np
import pandas as pd
from datetime import date

def fix_missing_values(column_values):
    column_values = np.array(column_values, dtype='str')
    fixed = [column_values[0]]
    for i in range(1, len(column_values)):
        if column_values[i] == 'nan':
            fixed.append(fixed[i-1])
        else:
            fixed.append(column_values[i])
    return fixed

def fix_columns(data):
    data['Operator'] = fix_missing_values(data['Operator'].values)
    data['Tabel'] = fix_missing_values(data['Tabel'].values)
    data['Kode General'] = data['Tabel'].apply(lambda row: re.findall(r'\((\w+)\)', row)[-1])
    data['Keterangan Variabel'] = data['Keterangan Variabel'].apply(lambda val: [v.split(':') for v in val.replace('=',':').split('\n')])
    data['Keterangan Variabel'] = data['Keterangan Variabel'].apply(lambda vals: [re.findall(r'([a-zA-Z 0-9]*)\(\w+\)' ,val[1])[0].strip().lower() for val in vals])
    data = data.rename(columns={'Tabel':'Produk', 'RegEx':'Regex'})
    data['Regex'] = data['Regex'].apply(lambda row: row[1:].strip('\'').strip('\"'))

    return data

def convert_raw_label(data):
    convert = {
            'Kuota Utama (GB)' : ['kuota utama',
                                  'kuota utama dalam gb atau mb', 
                                  'kuota utama dalam gb',
                                  'kuota utama  dalam gb',
                                  'besar kuota dalam gb atau mb', 
                                'besar kuota internet dalam gb'],
            'Kuota 4G (GB)' : ['kuota 4g',
                               'kuota khusus 4g dalam gb'],
            'Masa Berlaku (Hari)' : ['masa berlaku dalam hari',
                                     'masa berlaku'],
            'Kuota Aplikasi (GB)' : ['kuota aplikasi dalam gb',
                                     'kuota apps'],
            'Fair Usage Policy (GB)' : ['fair usage policy dalam gb',
                                        'fair usage policy dalam gb atau mb per hari'],
            'Internet Siang (GB)' : ['internet siang dalam gb',
                                     'kuota malam dalam gb'],
            'Internet Malam (GB)' : ['internet malam dalam gb'],
            'Kuota Videomax (GB)' : ['besar kuota videomax dalam gb',
                                     'kuota videomax dalam gb'],
            'Kuota Maxstream (GB)' : ['kuota maxstream dalam gb'],
            'Kuota Youtube (GB)' : ['kuota youtube dalam gb'],
            'Kuota Zona (GB)' : ['kuota zona dalam gb',
                                 'kuota lokal dalam gb atau mb',
                                 'kuota lokal dalam gb'],
            'Durasi Gratis Telepon ke Semua Operator (Menit)' : ['durasi telpon ke semua operator']
                                    }

    full = []

    for val in data['Keterangan Variabel'].values:
        temp = []
        for v in val:
            for key, values in convert.items():
                if v in values:
                    temp.append(key)
                    break
        full.append(np.array(temp))
        

    data['Keterangan Variabel'] = np.array(full)
    return data

def scrape_data():
    url = 'https://portalpulsa.com/paket-data-internet-murah/'
    all_data = pd.read_html(url)
    data = pd.DataFrame()
    for d in all_data:
        data = pd.concat([data, d])
    data = data.reset_index(drop=True)

    return data

def create_general_code(data):
    regex = r'(\D+)\d+'
    data['Kode General'] = data['Kode'].apply(lambda row: re.findall(regex, row)[0])
    data = data.rename(columns={'harga':'Harga'})

    return data

def merge_table(informations, scraped_data):
    merged_data = scraped_data.merge(informations, how = 'inner', on = 'Kode General')
    
    return merged_data
    
def extract_regex(merged_data):
    merged_data['Regex Extract'] = merged_data.apply(lambda row: re.findall(row['Regex'], row['Produk_x']), axis=1)
    bools = merged_data['Regex Extract'].apply(lambda row: len(row)) > 0
    merged_data = merged_data.loc[bools, :]
    vals = merged_data['Regex Extract'].values
    temp_vals = []
    for val in vals:
        if type(val[0]) == type(tuple()):
            val = list(val[0])
            while True:
                if 'GB' in val :
                    index = val.index('GB')
                elif 'MB' in val:
                    index = val.index('MB')
                    val[index-1] = int(val[index-1]) / 1000
                elif 'tahun' in val:
                    index = val.index('tahun')
                    val[index-1] = int(val[index-1]) * 365
                elif 'GB/HR' in val:
                    index = val.index('GB/HR')
                elif 'MB/HR' in val:
                    index = val.index('MB/HR')
                    val[index-1] = int(val[index-1]) / 1000
                else:
                    break
                val.pop(index)
        temp_vals.append(val)
    merged_data.loc[:, 'Regex Extract'] = temp_vals

    return merged_data

def create_final_df(extracted_data):
    final = pd.DataFrame()
    for key, val, product in zip(extracted_data['Keterangan Variabel'].values, extracted_data['Regex Extract'].values, extracted_data['Produk_x'].values):
        dict = {k:[float(v)] for k, v in zip(key, val)}
        dict['Produk'] = [product]
        final = pd.concat([final, pd.DataFrame(dict)])
    final = final.fillna(0)
    data_to_merge = extracted_data[['Produk_x', 'Kode', 'Operator', 'Harga']]
    final = final.merge(data_to_merge, left_on='Produk', right_on='Produk_x', how='left')
    final = final[['Operator',
                   'Produk_x',
                   'Kode',
                   'Harga',
                   'Masa Berlaku (Hari)',
                   'Kuota Utama (GB)',
                   'Kuota 4G (GB)',
                   'Kuota Aplikasi (GB)',
                   'Fair Usage Policy (GB)',
                   'Internet Siang (GB)',
                   'Internet Malam (GB)',
                   'Kuota Videomax (GB)',
                   'Kuota Maxstream (GB)',
                   'Kuota Youtube (GB)',
                   'Kuota Zona (GB)',
                   'Durasi Gratis Telepon ke Semua Operator (Menit)']]
    final = final.rename(columns={'Produk_x':'Produk'})

    return final

def combine_columns(final_data):
    columns_1 = ['Internet Siang (GB)',
        'Internet Malam (GB)',
        'Kuota Zona (GB)',
        'Kuota 4G (GB)']
    columns_2 = ['Kuota Videomax (GB)',
        'Kuota Maxstream (GB)',
        'Kuota Youtube (GB)']
    target_1 = 'Kuota Utama (GB)'
    target_2 = 'Kuota Aplikasi (GB)'
    for target, columns in zip([target_1, target_2], [columns_1, columns_2]):
        for col in columns:
            final_data[target] = final_data[target] + final_data[col]
            final_data = final_data.drop(columns=col)
    final_data = final_data.drop(columns='Durasi Gratis Telepon ke Semua Operator (Menit)')

    return final_data

def temporary_fix(data):
    bool = data['Kode'].apply(lambda row: re.findall(r'(\D+)\d+', row)[0] == 'SMU' or re.findall(r'(\D+)\d+', row)[0] == 'VSMU').values
    fup = data['Fair Usage Policy (GB)'].values
    mb = data['Masa Berlaku (Hari)'].values
    for i, b in enumerate(bool):
        if b == True:
            fup[i] = fup[i] * mb[i]
    data['Fair Usage Policy (GB)'] = fup
    data['Masa Berlaku (Hari)'] = data['Masa Berlaku (Hari)'].apply(lambda row: 365 if row == 0 else row)
    
    return data

raw_data = pd.read_csv('Paket Data Competitors Raw.csv')
clean_data = fix_columns(raw_data)
clean_data = convert_raw_label(clean_data)
scraped_data = scrape_data()
scraped_data = create_general_code(scraped_data)
scraped_data = scraped_data.loc[scraped_data['Status'] == 'normal', :]
merged_data = merge_table(clean_data, scraped_data)
extracted_data = extract_regex(merged_data)
final_data = create_final_df(extracted_data)
data = combine_columns(final_data)
data = temporary_fix(data)
data.to_csv(f'Product Information - {date.today()}.csv', index=False)