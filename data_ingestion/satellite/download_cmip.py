import GOES
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
from pyproj import Proj

def get_satellite(dt):
    """Retorna el satélite correcto según la fecha."""
    transition = datetime(2025, 4, 4, 15, 10)  # 15:10 UTC
    if dt < transition:
        return 'goes16'
    else:
        return 'goes19'


def download_cmip(date_ini, date_fin):
    """
    Download ABI-L2-CMIPF (Cloud and Moisture Imagery) from GOES-16
    for bands 08, 09, 10, 13. Crops to ~400x400 km domain centered
    on Bogotá (4.596°N, 74.077°O) and saves compressed NetCDF files.

    Parameters
    ----------
    date_ini : str ['YYYY-MM-DD HH:MM']
    date_fin : str ['YYYY-MM-DD HH:MM']
    """

    # Generar secuencia de timesteps cada 10 minutos
    df = pd.DataFrame()
    df['Tiempo'] = pd.to_datetime(
        np.arange(
            datetime(int(date_ini[:4]), int(date_ini[5:7]), int(date_ini[8:10]),
                     int(date_ini[11:13]), int(date_ini[14:])),
            datetime(int(date_fin[:4]), int(date_fin[5:7]), int(date_fin[8:10]),
                     int(date_fin[11:13]), int(date_fin[14:])),
            timedelta(minutes=10)
        ).astype(datetime)
    )

    canales = ['08', '09', '10', '13', '14']

    path_out = 'cmip_raw/'       # archivos full-disk descargados
    path_out_c = 'cmip_cropped/' # archivos recortados y comprimidos
    log_file = 'cmip_log.txt'    # log de timesteps completados

    parent_dir = os.getcwd()
    for p in [path_out, path_out_c]:
        full = os.path.join(parent_dir, p)
        if not os.path.exists(full):
            os.mkdir(full)
            print(f"Directorio creado: {full}")

    # Cargar log de timesteps ya completados
    completados = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            completados = set(line.strip() for line in f.readlines())
    print(f"Timesteps ya completados: {len(completados)}")

    # Dominio ~400x400 km centrado en Bogotá
    bound = {'lon': [-75.877, -72.277],
             'lat': [2.796, 6.396]}

    for dd in df['Tiempo']:
        ts_key = str(dd)

        # Si ya fue procesado, saltar
        if ts_key in completados:
            print(f"Ya procesado, saltando: {dd}")
            continue

        try:
            print(f'\nDatetime: {dd}')
            satellite = get_satellite(dd)
            print(f"Satélite: {satellite}")

            T_ini = str(dd).replace('-', '').replace(':', '').replace(' ', '-')
            T_fin = str(dd + pd.to_timedelta(10, 'min')).replace('-', '').replace(':', '').replace(' ', '-')

            # Descargar
            try:
                satellite = get_satellite(dd)
                GOES.download(satellite, 'ABI-L2-CMIPF',
                              DateTimeIni=T_ini, DateTimeFin=T_fin,
                              channel=canales,
                              rename_fmt='%Y%m%d%H%M%S',
                              path_out=path_out)
                print("Bandas descargadas")
            except Exception as e:
                print(f"No encontrado en AWS: {dd} — {e}")
                continue

            lista_paths = os.listdir(path_out)
            if len(lista_paths) == 0:
                print(f"Carpeta vacía después de descarga: {dd}")
                continue

            # Identificar archivos por banda
            archivos = {canal: None for canal in canales}
            for f in lista_paths:
                for canal in canales:
                    if f'C{canal}' in f:
                        archivos[canal] = f

            # Necesitamos al menos una banda para obtener proyección
            ref_file = next((v for v in archivos.values() if v is not None), None)
            if ref_file is None:
                print(f"Sin archivos de referencia: {dd}")
                continue

            ds_ref = Dataset(path_out + ref_file)
            sat_h = ds_ref.variables['goes_imager_projection'].perspective_point_height
            sat_lon = ds_ref.variables['goes_imager_projection'].longitude_of_projection_origin
            sat_sweep = ds_ref.variables['goes_imager_projection'].sweep_angle_axis

            x = ds_ref.variables['x'][:]
            y = ds_ref.variables['y'][:]

            p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

            xmin, ymin = p(bound['lon'][0], bound['lat'][0]) / sat_h
            xmax, ymax = p(bound['lon'][1], bound['lat'][1]) / sat_h

            sel_x = np.where((x >= xmin) & (x <= xmax))[0]
            sel_y = np.where((y >= ymin) & (y <= ymax))[0]

            x_crop = x[sel_x]
            y_crop = y[sel_y]

            x_, y_ = np.meshgrid(x_crop * sat_h, y_crop * sat_h)
            lon_grid, lat_grid = p(x_, y_, inverse=True)

            # Crear NetCDF de salida
            file_name = f'CMIP_{T_ini}.nc'
            ds_out = Dataset(file_name, 'w', format='NETCDF4')
            ds_out.createDimension('y', len(y_crop))
            ds_out.createDimension('x', len(x_crop))

            # Guardar coordenadas
            lat_var = ds_out.createVariable('lat', 'f4', ('y',))
            lon_var = ds_out.createVariable('lon', 'f4', ('x',))
            lat_var[:] = lat_grid[:, 0]
            lon_var[:] = lon_grid[0, :]
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'

            # Guardar cada banda
            for canal in canales:
                if archivos[canal] is None:
                    print(f"Banda {canal} faltante en {dd}")
                    continue
                ds_banda = Dataset(path_out + archivos[canal])
                cmi = ds_banda.variables['CMI'][
                    sel_y.min():sel_y.max() + 1,
                    sel_x.min():sel_x.max() + 1
                ]
                fill = ds_banda.variables['CMI']._FillValue
                dtype_cmi = ds_banda.variables['CMI'].dtype
                var = ds_out.createVariable(
                    f'CMI_C{canal}', dtype_cmi, ('y', 'x'),
                    fill_value=fill
                )
                var.setncatts({k: ds_banda.variables['CMI'].getncattr(k)
                               for k in ds_banda.variables['CMI'].ncattrs()})
                var[:] = cmi
                ds_banda.close()

            ds_ref.close()
            ds_out.close()

            # Comprimir
            cmd = f"nccopy -d9 {file_name} {path_out_c}CMIPC_{T_ini}.nc"
            ret = os.system(cmd)
            if ret == 0:
                os.remove(file_name)
                print(f"Comprimido y guardado: CMIPC_{T_ini}.nc")
            else:
                print(f"Error comprimiendo {file_name} — se deja sin comprimir")
                continue

            # Limpiar archivos full-disk
            for f in lista_paths:
                fp = path_out + f
                if os.path.isfile(fp):
                    os.remove(fp)

            # Registrar en log
            with open(log_file, 'a') as f:
                f.write(ts_key + '\n')

        except Exception as e:
            print(f"Error en {dd}: {e}")
            continue

    print("\nDescarga completada.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date_ini", required=True, type=str)
    parser.add_argument("--date_fin", required=True, type=str)
    args = parser.parse_args()
    download_cmip(args.date_ini, args.date_fin)