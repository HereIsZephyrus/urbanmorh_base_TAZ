from osgeo import gdal, ogr, osr
from dotenv import load_dotenv

import os
import shutil
import sys

load_dotenv()
gdal.UseExceptions()
gdal.AllRegister()
ogr.UseExceptions()
ogr.RegisterAll()

from qgis.core import *
from qgis.PyQt.QtCore import QMetaType

QGIS_PREFIX_PATH = os.environ.get("QGIS_PREFIX_PATH")
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
app = QgsApplication([], False)
app.initQgis()
#from qgis import processing
import processing
from processing.core.Processing import Processing
Processing.initialize()

lcz_dir = os.getenv("LCZ_DIR")

split_level_mapper = {
    "natural": [0,10],
    "water": [16.1]
}

def copy_base_road():
    base_road_filepath = os.getenv("OSM_ROAD")
    current_filepath = os.path.join(os.getcwd(), "result1.shp")
    if os.path.exists(current_filepath):
        os.remove(current_filepath)
    # copy this shapefile to current directory
    shutil.copy(base_road_filepath, os.getcwd())
    current_filepath = os.path.join(os.getcwd(), "result1.shp")
    return current_filepath

def split_tiff(location_dir, file, split_type):
    file_path = os.path.join(location_dir, file)
    LCZtiff = gdal.Open(file_path)
    LCZdata = LCZtiff.GetRasterBand(1)
    spatialRef = LCZtiff.GetSpatialRef()

    try:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds_path = os.path.join(location_dir, f"{split_type}_contour.shp")
        print(ds_path)
        if os.path.exists(ds_path):
            driver.DeleteDataSource(ds_path)
        ds = driver.CreateDataSource(ds_path)
        src = osr.SpatialReference()
        src.ImportFromWkt(spatialRef.ExportToWkt())
    except Exception as e:
        print(f"error creating output shapefile: {e}")

    try:
        boundaries_layer = ds.CreateLayer(f"{split_type}_boundaries", src, ogr.wkbLineString)
        if boundaries_layer is None:
            print("error creating natural boundaries layer")
            return
        field_id = ogr.FieldDefn("ID", ogr.OFTInteger)
        boundaries_layer.CreateField(field_id)
        field_elev = ogr.FieldDefn("elev", ogr.OFTReal)
        boundaries_layer.CreateField(field_elev)

        levels = split_level_mapper[split_type]
        boundary = gdal.ContourGenerate(
            LCZdata,0,0,
            levels,0,0,
            boundaries_layer,0,1
        )
        if (boundary != 0):
            raise Exception(f"error creating {split_type} boundaries layer")
            
    except Exception as e:
        print(f"error creating natural boundaries layer: {e}")
    
    return ds_path

def delete_small_features(line_path):
    '''
    Use QGIS API to convert line into polygon, calc the area, and then record the id list to return
    '''
    line_dir = os.path.dirname(line_path)
    line_name = os.path.basename(line_path).split('.')[0]
    poly_path = os.path.join(line_dir, f'{line_name}_poly.shp')
    proj_poly_path = os.path.join(line_dir, f'{line_name}_proj_poly.shp')
    if os.path.exists(poly_path):
        os.remove(poly_path)
    if os.path.exists(proj_poly_path):
        os.remove(proj_poly_path)

    convert_params = {
        "INPUT": line_path,
        "OUTPUT": poly_path
    }
    try:
        poly = processing.run("qgis:linestopolygons", convert_params)
    except Exception as e:
        print(f"error converting line to polygon: {e}")
        return

    crs_params = {
        'CONVERT_CURVED_GEOMETRIES' : False,
        'INPUT' : poly_path,
        'OPERATION' : '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=50 +ellps=WGS84',
        'OUTPUT' : proj_poly_path,
        'TARGET_CRS' : QgsCoordinateReferenceSystem('EPSG:32650')
    }
    try:
        poly = processing.run("native:reprojectlayer", crs_params)
    except Exception as e:
        print(f"error projecting polygon: {e}")
        return
    
    # iterate through the polygon layer and add the area attribute
    poly_layer = QgsVectorLayer(proj_poly_path, 'Multipolygon Layer', 'ogr')
    if not poly_layer.isValid():
        print("Layer failed to load!")
    line_layer = QgsVectorLayer(line_path, 'Line Layer', 'ogr')
    if not line_layer.isValid():
        print("Line layer failed to load!")
    
    poly_layer.dataProvider().addAttributes([QgsField("area", QMetaType.Type.Double)])
    poly_layer.updateFields()
    poly_layer.startEditing()
    line_layer.dataProvider().addAttributes([QgsField("remain", QMetaType.Type.Int)])
    line_layer.updateFields()
    line_layer.startEditing()
    remain_id_list = []
    for feature, line_feature in zip(poly_layer.getFeatures(), line_layer.getFeatures()):
        area = feature.geometry().area()
        #print(area)
        feature['area'] = area
        poly_layer.updateFeature(feature)
        if area > 200000: # 0.1 km2
            line_feature['remain'] = 1
        else:
            line_feature['remain'] = 0
        line_layer.updateFeature(line_feature)

    poly_layer.commitChanges()
    line_layer.commitChanges()

def create_mask(location_dir, file_name, split_type):
    contour_path = split_tiff(location_dir, file_name, split_type)
    delete_small_features(contour_path)
    return contour_path

def merge_shapefile(base_road_filepath, water_path):
    base_road_ds = ogr.Open(base_road_filepath)
    base_road_layer = base_road_ds.GetLayer(0)
    water_ds = ogr.Open(water_path)
    water_layer = water_ds.GetLayer(0)
    # merge the two layers
    for feature in water_layer:
        base_road_layer.CreateFeature(feature)

def __main__():
    app.initQgis()
    base_road_filepath = copy_base_road()
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        for file in os.listdir(location_dir):
            if file.endswith(".tif"):
                print(file)
                natural_path = create_mask(location_dir, file, "natural")
                water_path = create_mask(location_dir, file, "water")
                break
        #merge_shapefile(base_road_filepath, water_path)
    app.exitQgis()

if __name__ == "__main__":
    __main__()