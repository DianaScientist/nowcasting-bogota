import sys
import argparse
import GOES
from netCDF4 import Dataset
from pyproj import Proj
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def get_Rad(date_ini,date_fin):
    """
    Download radiance data from GOES satellite in the bands 08-09-10-13-14, cut the images around Colombia
    territory and compress the NETCDF files.

    Parameters
    ----------
    dateini : str ['YY-MM-DD HH:MM']
        Initial download date 
    datefin : str ['YY-MM-DD HH:MM']
        Final download date  
    """
    #crea un dataframe con fechas entre dateini y datefin con un delta de tiempo de 9 minutos. Los objetos
    #almacenados en el dataframe son de tipo timestep
    df = pd.DataFrame()
    df['Tiempo'] = pd.to_datetime(np.arange(datetime(int(date_ini[:4]),
                                                    int(date_ini[5:7]),int(date_ini[8:10]),
                                                    int(date_ini[11:13]),int(date_ini[14:])), 
                                            datetime(int(date_fin[:4]),
                                                    int(date_fin[5:7]),int(date_fin[8:10]),
                                                    int(date_fin[11:13]),int(date_fin[14:])), 
                                            timedelta(minutes=9)).astype(datetime))

    path_out = 'radiances/'  #carpeta donde se almacenan las imagenes completas descargadas
    path_out_c = 'radiances_c/'  #carpeta donde se almacenan las imagenes después de recortarlas y comprimirlas
    canales = ['08', '09', '10', '13', '14']  #lista con los canales que se desean descargar y almacenar en las imagenes
    
    #revisar que ya estén creados los archivos. y los folders, 
    #para reactivar el codigo para que el usuario no. TOmar la fecha del archivo final y que sea el comienzo 
    #de la fecha de la descarga, la final la que el usuario digita por consola o en el arcvhivo de
    #configuracion
    # Parent Directory path 
    parent_dir = os.getcwd()  #obtiene el directorio de trabajo actual
    
    # Path 
    original_path = os.path.join(parent_dir, path_out)  #genera la ruta para crear la carpeta donde se almacenan las imagenes completas 
    path_c = os.path.join(parent_dir, path_out_c)  #genera la ruta para crear la carpeta donde se almacenan las imagenes recortadas y comprimidas
    
    if not os.path.exists(original_path):  #antes de crear el directorio, valida que no exista
        os.mkdir(original_path)  #crea el directorio con la carpeta donde se almacenan las imagenes completas     
    
    if not os.path.exists(path_c):  #antes de crear el directorio, valida que no exista
        os.mkdir(path_c)  #crea el directorio con la carpeta donde se almacenan las imagenes recortadas y comprimidas
            
    print("Directory '% s' created" % original_path) 
    print("Directory '% s' created" % path_c) 

    for dd in (df['Tiempo']):
        try: 
            print('Datetime:',dd)
            print("\n")

            #transforma las fechas del dataframe de timestep a formato '%Y%m%d%H%M%S'
            T_ini = str(dd)
            T_ini = T_ini.replace('-', '')
            T_ini = T_ini.replace(':', '')
            T_ini = T_ini.replace(' ', '-')

            T_fin = str(dd+pd.to_timedelta(9,'min'))
            T_fin = T_fin.replace('-', '')
            T_fin = T_fin.replace(':', '')
            T_fin = T_fin.replace(' ', '-')
            
            #descarga las imágenes
            try: 
                download = GOES.download('goes16', 'ABI-L1b-RadF', DateTimeIni = T_ini, DateTimeFin = T_fin,
                    channel = canales, rename_fmt = '%Y%m%d%H%M%S', path_out = path_out)
                print("All radiance images were downloaded")
            except Exception as e: 
                print(f"The {dd} Radiance band 8,9,10,13,14 not found") 
                continue  
            print("\n")
            
            missing_rad = []
            lista_paths = os.listdir(path_out)  #lista los elemntos de la carpeta, lista los nombres de las imagenes descargadas
            if len(lista_paths) != 0:
                if lista_paths[0][0] == 'O':
                    name = lista_paths[0][:19]
                    date = lista_paths[0][27:]
                else:
                    print(f"NO hay radianzas descargadas en la carpeta: {path_out}")
                    print("\n")
                
                #se abren las imagenes de cada radianza 
                if name + '08_G16_s' + date in lista_paths:  #se valida que la imagen de la radianza exista en la carpeta
                    ds8 = Dataset(path_out + name + '08_G16_s'+ date)  #se lee la imagen
                else: 
                    missing_rad.append(8)  #si no existe, se agrega a la lista de radianzas faltantes

                if name + '09_G16_s' + date in lista_paths:
                    ds9 = Dataset(path_out + name + '09_G16_s' + date)
                else: 
                    missing_rad.append(9)

                if name + '10_G16_s' + date in lista_paths:
                    ds10 = Dataset(path_out + name + '10_G16_s' + date)
                else: 
                    missing_rad.append(10)

                if name + '13_G16_s' + date in lista_paths:
                    ds13 = Dataset(path_out + name + '13_G16_s' + date)
                else: 
                    missing_rad.append(13)
                
                if name + '14_G16_s' + date in lista_paths:
                    ds14 = Dataset(path_out + name + '14_G16_s' + date)
                else: 
                    missing_rad.append(14)

                #se abre un dataset de referencia para obtener los valores de la altura del satelite y los angulos de inclinación
                ds = Dataset(path_out + lista_paths[0])
                sat_h= ds.variables['goes_imager_projection'].perspective_point_height
                sat_lon = ds.variables['goes_imager_projection'].longitude_of_projection_origin
                sat_sweep = ds.variables['goes_imager_projection'].sweep_angle_axis

                #variables x,y de la imagen
                x = ds.variables['x'][:]
                y = ds.variables['y'][:]

                #proyección para trasformar de coordenadas de la imagen a coordenadas geograficas
                p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

                XX, YY = np.meshgrid(x, y)
                lons, lats = p(XX, YY, inverse=True)

                #coordenadas geograficas para recortar la imagen
                bound = {'lon': [-81.03,-64], 
                        'lat': [-4.1,12.78]}

                #coordenadas de la imagen para recortar 
                xmin,ymin = p(bound['lon'][0],bound['lat'][0])/sat_h
                xmax,ymax = p(bound['lon'][1],bound['lat'][1])/sat_h

                #selección de coordenadas dentro de los límites definidos en bound
                sel_x = np.where((x>=xmin) & (x<=xmax))
                sel_y = np.where((y>=ymin) & (y<=ymax))

                x_col = x[sel_x]
                y_col = y[sel_y]

                print("Croping radiance images...")
                print("\n")

                #selección de datos de radianza dentro de la región recortada
                try:
                    Rad8 = ds8.variables['Rad'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]
                except NameError:
                    pass

                try: 
                    Rad9 = ds9.variables['Rad'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]
                except NameError:
                    pass
                
                try:
                    Rad10 = ds10.variables['Rad'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]
                except NameError: 
                    pass

                try:
                    Rad13 = ds13.variables['Rad'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]
                except NameError:
                    pass
                
                try:
                    Rad14 = ds14.variables['Rad'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]
                except NameError: 
                    pass
                
                #generando arreglos de latitud y longitud de la imagen recordata
                x_, y_ = np.meshgrid(x_col*sat_h, y_col*sat_h)
                lon_colombia, lat_colombia = p(x_, y_, inverse=True)

                lon_colombia_=lon_colombia[0]
                lat_colombia_=lat_colombia[:,0]

                print("Creating new NETCDF file with all bands radiances...")
                print("\n")
                #se crea un nuevo netcdf, una nueva imagen
                file_name = 'RadF_' + date + '.nc'
                ds_out = Dataset(file_name,'w',format='NETCDF4')

                #se generan las dimensiones de la imagen recortada
                ds_out.createDimension('y',len(y_col))
                ds_out.createDimension('x',len(x_col))
                    
                #Agregar variables            
                fill = ds.variables['Rad']._FillValue

                if fill == 65535:
                    datatype = 'u2'
                else: 
                    datatype = 'i2'         

                #se crean variables con cada una de las bandas de las radianzas y se insertan los datos
                try: 
                    dsRad8 = ds_out.createVariable('Rad8',datatype,('y','x',),fill_value=fill)
                    dsRad8.setncatts({k: ds8.variables['Rad'].getncattr(k) for k in ds8.variables['Rad'].ncattrs()})
                    dsRad8[:] = Rad8
                except NameError:
                    pass
                
                try: 
                    dsRad9 = ds_out.createVariable('Rad9',datatype,('y','x',),fill_value=fill)
                    dsRad9.setncatts({k: ds9.variables['Rad'].getncattr(k) for k in ds9.variables['Rad'].ncattrs()})
                    dsRad9[:] = Rad9
                except NameError:
                    pass

                try: 
                    dsRad10 = ds_out.createVariable('Rad10',datatype,('y','x',),fill_value=fill)
                    dsRad10.setncatts({k: ds10.variables['Rad'].getncattr(k) for k in ds10.variables['Rad'].ncattrs()})
                    dsRad10[:] = Rad10
                except NameError:
                    pass

                try: 
                    dsRad13 = ds_out.createVariable('Rad13',datatype,('y','x',),fill_value=fill)
                    dsRad13.setncatts({k: ds13.variables['Rad'].getncattr(k) for k in ds13.variables['Rad'].ncattrs()})
                    dsRad13[:] = Rad13
                except NameError:
                    pass
                
                try: 
                    dsRad14 = ds_out.createVariable('Rad14',datatype,('y','x',),fill_value=fill)
                    dsRad14.setncatts({k: ds14.variables['Rad'].getncattr(k) for k in ds14.variables['Rad'].ncattrs()})
                    dsRad14[:] = Rad14
                except NameError:
                    pass
                
                print("Saving data in the new NETCDF file...")
                print("\n")

                #se crea la variable de las coordenadas geograficas
                lat = ds_out.createVariable('lat','i2',('y',))
                lon = ds_out.createVariable('lon','i2',('x',))

                n = 16

                max_lon = lon_colombia_.max()
                max_lat = lat_colombia_.max()

                min_lon = lon_colombia_.min()
                min_lat = lat_colombia_.min()

                #se calcula y se agrega el factor de escala
                scale_factor_lon = (max_lon - min_lon) / (2 ** n - 1)
                add_offset_lon = min_lon + 2 ** (n - 1) * scale_factor_lon

                scale_factor_lat = (max_lat - min_lat) / (2 ** n - 1)
                add_offset_lat = min_lat + 2 ** (n - 1) * scale_factor_lon

                lon.add_offset = add_offset_lon
                lon.scale_factor = scale_factor_lon

                lat.add_offset = add_offset_lat
                lat.scale_factor = scale_factor_lat

                lat[:] = lat_colombia_
                lon[:] = lon_colombia_

                #se agrega como parametro la lista de radianzas faltantes
                ds_out.missing_rad = missing_rad

                #se cierran todos los datasets abiertos
                ds_out.close()
                ds.close()
                
                if name + '08_G16_s' + date in lista_paths:
                    ds8.close()

                if name + '09_G16_s' + date in lista_paths:
                    ds9.close()

                if name + '10_G16_s' + date in lista_paths:
                    ds10.close()

                if name + '13_G16_s' + date in lista_paths:
                    ds13.close()
                
                if name + '14_G16_s' + date in lista_paths:
                    ds14.close()

                #se ejecuta un comando de terminal invocado desde python que realiza la compresión de las imagenes
                print("Compressig file...")
                comman_line_compression = os.system(f"nccopy -d9 {file_name} {path_out_c + 'RadFC_' + date}")
                if comman_line_compression == 0:
                    print(f"The file {file_name} was compressed")
                    print("\n")
                else: 
                    print(f"ERROR TRYING TO COMPRESS THE FILES. TRY RUN THE FOLLOWING COMMAND IN THE COMMAND LINE: nccopy -d9 {file_name} {path_out_c + 'LSTFC_' + date}")
                    break
                
                if os.path.isfile(f"{file_name}"):
                    print("Removing crop image incompressing...")
                    print("\n")
                    os.remove(f"{file_name}")
                else: 
                    print(f"The file {file_name} does exist")
                    print("\n") 
                    break  

                #se eliminan las imagenes completas, se dejan las recortadas
                print("Removing originals incompressing files...")
                print("\n")
                if os.path.isfile(f"{path_out + name + '08_G16_s'+ date}"):
                    os.remove(f"{path_out + name + '08_G16_s'+ date}")
                
                if os.path.isfile(f"{path_out + name + '09_G16_s'+ date}"):   
                    os.remove(f"{path_out + name + '09_G16_s'+ date}")
                
                if os.path.isfile(f"{path_out + name + '10_G16_s'+ date}"):
                    os.remove(f"{path_out + name + '10_G16_s'+ date}")
                
                if os.path.isfile(f"{path_out + name + '13_G16_s'+ date}"):
                    os.remove(f"{path_out + name + '13_G16_s'+ date}")
                
                if os.path.isfile(f"{path_out + name + '14_G16_s'+ date}"):    
                    os.remove(f"{path_out + name + '14_G16_s'+ date}")

            else: 
                pass

        except Exception as e:
            print(f"An exception occurred downloading the {dd} RADIANCE: {e}")
            print("\n")
            break     
        else:
            print(f"Ejecution was succesfully, the radiance image was created")
            print("\n")

#------------------------------------------------------------------------------------------------------------------

def get_LST(date_ini,date_fin):
    """
    Download Land Surface Temperature data from GOES satellite, cut the images around Colombia
    territory and compress the NETCDF files.

    Parameters
    ----------
    dateini : str ['YY-MM-DD HH:MM']
        Initial download date 
    datefin : str ['YY-MM-DD HH:MM']
        Final download date  
    """

    df = pd.DataFrame()
    df['Tiempo'] = pd.to_datetime(np.arange(datetime(int(date_ini[:4]),
                                                    int(date_ini[5:7]),int(date_ini[8:10]),
                                                    int(date_ini[11:13]),int(date_ini[14:])), 
                                            datetime(int(date_fin[:4]),
                                                    int(date_fin[5:7]),int(date_fin[8:10]),
                                                    int(date_fin[11:13]),int(date_fin[14:])), 
                                            timedelta(minutes=60)).astype(datetime))

    path_out = 'LST/'
    path_out_c = 'LST_c/'
    
    # Parent Directory path 
    parent_dir = os.getcwd()
    
    # Path 
    original_path = os.path.join(parent_dir, path_out)
    path_c = os.path.join(parent_dir, path_out_c) 
    
    if not os.path.exists(original_path):
        os.mkdir(original_path)
    
    if not os.path.exists(path_c):
        os.mkdir(path_c)

    print("Directory '% s' created" % original_path) 
    print("Directory '% s' created" % path_c) 

    for dd in (df['Tiempo']):
        try: 
            print('Datetime:',dd)
            print("\n")

            T_ini = str(dd)
            T_ini = T_ini.replace('-', '')
            T_ini = T_ini.replace(':', '')
            T_ini = T_ini.replace(' ', '-')

            T_fin = str(dd+pd.to_timedelta(60,'min'))
            T_fin = T_fin.replace('-', '')
            T_fin = T_fin.replace(':', '')
            T_fin = T_fin.replace(' ', '-')
            #fls = gen_Rad(T_ini,T_fin,path_out,canales,ltmin=4.6563-0.1,ltmax=4.6563+0.1,lnmin=-74.100-0.1, lnmax=-74.100+0.1,temp = 'LST_temp/')
            try:
                download = GOES.download('goes16','ABI-L2-LSTF',DateTimeIni=T_ini,DateTimeFin=T_fin, 
                                    rename_fmt = '%Y%m%d%H%M%S', path_out=path_out)
                print("The LST image were downloaded")
            except Exception as e:
                print(f"The {dd} LST not found") 
                continue
            print("\n")

            lista_paths = os.listdir(path_out)
            if len(lista_paths) != 0:
                if lista_paths[0][0] == 'O':
                    name = lista_paths[0][:23]
                    date = lista_paths[0][23:]
                else:
                    print(f"NO hay LST descargadas en la carpeta: {path_out}")
                    print("\n")

                LST = Dataset(path_out + name + date)

                sat_h = LST.variables['goes_imager_projection'].perspective_point_height
                sat_lon = LST.variables['goes_imager_projection'].longitude_of_projection_origin
                sat_sweep = LST.variables['goes_imager_projection'].sweep_angle_axis

                gip = LST.variables['goes_imager_projection']

                x = LST.variables['x'][:]
                y = LST.variables['y'][:]

                p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

                XX, YY = np.meshgrid(x, y)
                lons, lats = p(XX, YY, inverse=True)

                bound = {'lon': [-81.03,-64], 
                        'lat': [-4.1,12.78]}

                xmin,ymin = p(bound['lon'][0],bound['lat'][0])/sat_h
                xmax,ymax = p(bound['lon'][1],bound['lat'][1])/sat_h

                sel_x = np.where((x>=xmin) & (x<=xmax))
                sel_y = np.where((y>=ymin) & (y<=ymax))

                x_col = x[sel_x]
                y_col = y[sel_y]

                print("Croping LST images...")
                print("\n")
                LST_ = LST.variables['LST'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]

                x_, y_ = np.meshgrid(x_col*sat_h, y_col*sat_h)
                lon_colombia, lat_colombia = p(x_, y_, inverse=True)

                lon_colombia_=lon_colombia[0]
                lat_colombia_=lat_colombia[:,0]

                x_par = {'scale_factor': LST.variables['x'].scale_factor,
                    'add_offset': LST.variables['x'].add_offset,
                    'units': 'rad',
                    'axis': 'X'}
            
                y_par = {'scale_factor': LST.variables['y'].scale_factor,
                            'add_offset': LST.variables['y'].add_offset,
                            'units': 'rad',
                            'axis': 'Y'}

                print("Creating new NETCDF file with LST data...")
                print("\n")
                file_name = 'LSTF_' + date
                ds = Dataset(file_name,'w',format='NETCDF4')
                ds.createDimension('y',len(y_col))
                ds.createDimension('x',len(x_col))
                    
                #Agregar variables            
                fill = LST.variables['LST']._FillValue 

                if fill == 65535:
                    datatype = 'u2'
                else: 
                    datatype = 'i2'     

                dsLST = ds.createVariable('LST',datatype,('y','x',),fill_value=fill)

                dsLST.setncatts({k: LST.variables['LST'].getncattr(k) for k in LST.variables['LST'].ncattrs()})

                lat = ds.createVariable('lat','i2',('y',))
                lon = ds.createVariable('lon','i2',('x',))

                n = 16

                max_lon = lon_colombia_.max()
                max_lat = lat_colombia_.max()

                min_lon = lon_colombia_.min()
                min_lat = lat_colombia_.min()

                scale_factor_lon = (max_lon - min_lon) / (2 ** n - 1)
                add_offset_lon = min_lon + 2 ** (n - 1) * scale_factor_lon

                scale_factor_lat = (max_lat - min_lat) / (2 ** n - 1)
                add_offset_lat = min_lat + 2 ** (n - 1) * scale_factor_lon

                lon.add_offset = add_offset_lon
                lon.scale_factor = scale_factor_lon

                lat.add_offset = add_offset_lat
                lat.scale_factor = scale_factor_lat

                print("Saving data in the new NETCDF file...")
                print("\n")
                lat[:] = lat_colombia_
                lon[:] = lon_colombia_

                dsLST[:] = LST_

                ds.close()

                print("Compressig file...")
                comman_line_compression = os.system(f"nccopy -d9 {file_name} {path_out_c + 'LSTFC_' + date}")
                if comman_line_compression == 0:
                    print(f"The file {file_name} was compressed")
                    print("\n")
                else: 
                    print(f"ERROR TRYING TO COMPRESS THE FILES. TRY RUN THE FOLLOWING COMMAND IN THE COMMAND LINE: nccopy -d9 {file_name} {path_out_c + 'LSTFC_' + date}")
                    break

                if os.path.isfile(f"{file_name}"):
                    print("Removing crop image incompressing...")
                    print("\n")
                    os.remove(f"{file_name}")
                else: 
                    print(f"The file {file_name} does exist")
                    print("\n")
                    break   

                if os.path.isfile(f"{path_out + name + date}"):
                    print("Removing originals incompressing files...")
                    print("\n")
                    os.remove(f"{path_out + name + date}")

                else: 
                    print(f"The file {path_out + name + date} does exist")
                    print("\n")
                    break   
            else: 
                pass

        except Exception as e:
            print(f"An exception occurred downloading the {dd} LST: {e}")
            print("\n")
            break     
        else:
            print(f"Ejecution was succesfully, the LST image was created")
            print("\n")

#---------------------------------------------------------------------------------------------------------

def get_TPW(date_ini,date_fin):
    """
    Download Total Precipition Water data from GOES satellite, cut the images around Colombia
    territory and compress the NETCDF files.

    Parameters
    ----------
    dateini : str ['YY-MM-DD HH:MM']
        Initial download date 
    datefin : str ['YY-MM-DD HH:MM']
        Final download date  
    """

    df = pd.DataFrame()
    df['Tiempo'] = pd.to_datetime(np.arange(datetime(int(date_ini[:4]),
                                                    int(date_ini[5:7]),int(date_ini[8:10]),
                                                    int(date_ini[11:13]),int(date_ini[14:])), 
                                            datetime(int(date_fin[:4]),
                                                    int(date_fin[5:7]),int(date_fin[8:10]),
                                                    int(date_fin[11:13]),int(date_fin[14:])), 
                                            timedelta(minutes=9)).astype(datetime))

    path_out = 'TPW/'
    path_out_c = 'TPW_c/'
    
    # Parent Directory path 
    parent_dir = os.getcwd()
    
    # Path 
    original_path = os.path.join(parent_dir, path_out)
    path_c = os.path.join(parent_dir, path_out_c) 
    
    if not os.path.exists(original_path):
        os.mkdir(original_path)
    
    if not os.path.exists(path_c):
        os.mkdir(path_c)

    print("Directory '% s' created" % original_path) 
    print("Directory '% s' created" % path_c) 

    for dd in (df['Tiempo']):
        try: 
            print('Datetime:',dd)
            print("\n")

            T_ini = str(dd)
            T_ini = T_ini.replace('-', '')
            T_ini = T_ini.replace(':', '')
            T_ini = T_ini.replace(' ', '-')

            T_fin = str(dd+pd.to_timedelta(9,'min'))
            T_fin = T_fin.replace('-', '')
            T_fin = T_fin.replace(':', '')
            T_fin = T_fin.replace(' ', '-')
            #fls = gen_Rad(T_ini,T_fin,path_out,canales,ltmin=4.6563-0.1,ltmax=4.6563+0.1,lnmin=-74.100-0.1, lnmax=-74.100+0.1,temp = 'LST_temp/')
            try: 
                download = GOES.download('goes16','ABI-L2-TPWF',DateTimeIni=T_ini,DateTimeFin=T_fin, 
                                   path_out=path_out)
                print("The TPW image were downloaded")
            except Exception as e:
                print(f"The {dd} TPW not found")   
                continue
            print("\n")

            lista_paths = os.listdir(path_out)
            if len(lista_paths) != 0:
                if lista_paths[0][0] == 'O':
                    name = lista_paths[0][:23]
                    date = lista_paths[0][23:]
                else:
                    print(f"NO hay TPW descargadas en la carpeta: {path_out}")
                    print("\n")
                
                TPW = Dataset(path_out + name + date)

                sat_h = TPW.variables['goes_imager_projection'].perspective_point_height
                sat_lon = TPW.variables['goes_imager_projection'].longitude_of_projection_origin
                sat_sweep = TPW.variables['goes_imager_projection'].sweep_angle_axis

                gip = TPW.variables['goes_imager_projection']

                x = TPW.variables['x'][:]
                y = TPW.variables['y'][:]

                p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

                XX, YY = np.meshgrid(x, y)
                lons, lats = p(XX, YY, inverse=True)

                bound = {'lon': [-81.03,-64], 
                        'lat': [-4.1,12.78]}

                xmin,ymin = p(bound['lon'][0],bound['lat'][0])/sat_h
                xmax,ymax = p(bound['lon'][1],bound['lat'][1])/sat_h

                sel_x = np.where((x>=xmin) & (x<=xmax))
                sel_y = np.where((y>=ymin) & (y<=ymax))

                x_col = x[sel_x]
                y_col = y[sel_y]

                print("Croping TPW images...")
                print("\n")
                TPW_ = TPW.variables['TPW'][sel_y[0].min():sel_y[0].max()+1,sel_x[0].min():sel_x[0].max()+1]

                x_, y_ = np.meshgrid(x_col*sat_h, y_col*sat_h)
                lon_colombia, lat_colombia = p(x_, y_, inverse=True)

                lon_colombia_=lon_colombia[0]
                lat_colombia_=lat_colombia[:,0]

                x_par = {'scale_factor': TPW.variables['x'].scale_factor,
                    'add_offset': TPW.variables['x'].add_offset,
                    'units': 'rad',
                    'axis': 'X'}
            
                y_par = {'scale_factor': TPW.variables['y'].scale_factor,
                            'add_offset': TPW.variables['y'].add_offset,
                            'units': 'rad',
                            'axis': 'Y'}

                print("Creating new NETCDF file with TPW data...")
                print("\n")
                file_name = 'TPWF_' + date
                ds = Dataset(file_name,'w',format='NETCDF4')
                ds.createDimension('y',len(y_col))
                ds.createDimension('x',len(x_col))
                    
                #Agregar variables            
                fill = TPW.variables['TPW']._FillValue

                if fill == 65535:
                    datatype = 'u2'
                else: 
                    datatype = 'i2'                       

                dsTPW = ds.createVariable('TPW',datatype,('y','x',),fill_value=fill)

                dsTPW.setncatts({k: TPW.variables['TPW'].getncattr(k) for k in TPW.variables['TPW'].ncattrs()})

                lat = ds.createVariable('lat','i2',('y',))
                lon = ds.createVariable('lon','i2',('x',))

                n = 16

                max_lon = lon_colombia_.max()
                max_lat = lat_colombia_.max()

                min_lon = lon_colombia_.min()
                min_lat = lat_colombia_.min()

                scale_factor_lon = (max_lon - min_lon) / (2 ** n - 1)
                add_offset_lon = min_lon + 2 ** (n - 1) * scale_factor_lon

                scale_factor_lat = (max_lat - min_lat) / (2 ** n - 1)
                add_offset_lat = min_lat + 2 ** (n - 1) * scale_factor_lon

                lon.add_offset = add_offset_lon
                lon.scale_factor = scale_factor_lon

                lat.add_offset = add_offset_lat
                lat.scale_factor = scale_factor_lat

                print("Saving data in the new NETCDF file...")
                print("\n")
                lat[:] = lat_colombia_
                lon[:] = lon_colombia_

                dsTPW[:] = TPW_

                ds.close()

                print("Compressig file...")
                comman_line_compression = os.system(f"nccopy -d9 {file_name} {path_out_c + 'TPWFC_' + date}")
                if comman_line_compression == 0:
                    print(f"The file {file_name} was compressed")
                    print("\n")
                else: 
                    print(f"ERROR TRYING TO COMPRESS THE FILES. TRY RUN THE FOLLOWING COMMAND IN THE COMMAND LINE: nccopy -d9 {file_name} {path_out_c + 'LSTFC_' + date}")
                    break

                if os.path.isfile(f"{file_name}"):
                    print("Removing crop image incompressing...")
                    print("\n")
                    os.remove(f"{file_name}")
                else: 
                    print(f"The file {file_name} does exist")
                    print("\n")
                    break   

                if os.path.isfile(f"{path_out + name + date}"):
                    print("Removing originals incompressing files...")
                    print("\n")
                    os.remove(f"{path_out + name + date}")

                else: 
                    print(f"The file {path_out + name + date} does exist")
                    print("\n")
                    break   
            else: 
                pass
                    
        except Exception as e:
            print(f"An exception was occurd downloaded the {dd} TPW: {e}")
            print("\n")
            break     
        else:
            print(f"Ejecution was succesfully, the TPW image was created")
            print("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that download, crop and compress images from GOES satellite for the products: LST, TPW and radiances"
    )
    parser.add_argument("--function", required=True, type=str)
    parser.add_argument("--date_ini", required=False, type=str)
    parser.add_argument("--date_fin", required=True, type=str)
    args = parser.parse_args()
    #archivo configuracion por si no se ejecutan por terminal 
    #fecha se vuelven opcionales 
    
    func = args.function
    date_ini = args.date_ini
    date_fin = args.date_fin
    #invocate all functions
    if func == "get_Rad":
        get_Rad(date_ini,date_fin)

    elif func == "get_LST":
        get_LST(date_ini,date_fin)

    elif func == "get_TPW":
        get_TPW(date_ini,date_fin)
    else: 
        print("Unknow parameters, please digit 'get_Rad' if you want generate radiances images, or 'get_LST' to get the Land Sourface Temperature images, or 'get_TPW' to get the Total Precipition Water images'")        
