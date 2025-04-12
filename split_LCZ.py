from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
from qgis.core import *
import os
import logging

logging.basicConfig(filename='split_LCZ.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
gdal.UseExceptions()
gdal.AllRegister()
ogr.UseExceptions()
ogr.RegisterAll()

lcz_dir = os.getenv("LCZ_DIR")
QGIS_PREFIX_PATH = os.environ.get("QGIS_PREFIX_PATH")
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
app = QgsApplication([], False)
app.initQgis()
from construct_qgis_functions import *

logger.info("QGIS initialized")
split_level_mapper = {
    "natural": [0,10],
    "water": [16.1]
}

def copy_base_road():
    base_road_filepath = os.getenv("OSM_ROAD")
    current_filepath = os.path.join(os.getcwd(), "result1.shp")
    reproject_shapefile(base_road_filepath, current_filepath)
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

def delete_shapefile(shp_path):
    shp_dir = os.path.dirname(shp_path)
    shp_name = os.path.basename(shp_path).split('.')[0]
    for file in os.listdir(shp_dir):
        if file.startswith(shp_name):
            os.remove(os.path.join(shp_dir, file))

def delete_small_features(line_path):
    '''
    Use QGIS API to convert line into polygon, calc the area, and then record the id list to return
    '''
    poly_path = convert_line_to_polygon(line_path)
    proj_poly_path = reproject_shapefile(poly_path)
    filter_path = filter_remain_field(proj_poly_path, line_path)
    
    # delete polygon layer
    delete_shapefile(poly_path)
    delete_shapefile(proj_poly_path)

    return filter_path

def create_mask(location_dir, file_name, split_type):
    logger.debug(f"creating mask for {file_name}")
    contour_path = split_tiff(location_dir, file_name, split_type)
    logger.debug(f"contour_path: {contour_path}")
    filter_path = delete_small_features(contour_path)
    logger.debug(f"filter_path: {filter_path}")
    return filter_path

def merge_shapefile(base_road_filepath, feature_path):
    proj_feature_path = reproject_shapefile(feature_path)
    logger.debug(f"proj_feature_path: {proj_feature_path}")
    splited_road_path = split_lines(base_road_filepath, proj_feature_path)
    logger.debug(f"splited_road_path: {splited_road_path}")
    splited_road_centroid_path = calc_line_centroid(splited_road_path)
    logger.debug(f"splited_road_centroid_path: {splited_road_centroid_path}")
    distance_extracted_centroid_path = extract_whithindistance(splited_road_centroid_path, proj_feature_path)
    logger.debug(f"distance_extracted_centroid_path: {distance_extracted_centroid_path}")
    joined_splited_road_path = join_by_attribute(splited_road_path, distance_extracted_centroid_path)
    logger.debug(f"joined_splited_road_path: {joined_splited_road_path}")
    filtered_splited_road_path = extract_nonull_attribute(joined_splited_road_path, "FID_2")
    logger.debug(f"filtered_splited_road_path: {filtered_splited_road_path}")
    return filtered_splited_road_path

def __main__():
    app.initQgis()
    base_road_filepath = copy_base_road()
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        logger.info(f"processing {location}")
        for file in os.listdir(location_dir):
            if file.endswith(".tif"):
                natural_path = create_mask(location_dir, file, "natural")
                logger.debug(f"natural_path: {natural_path}")
                water_path = create_mask(location_dir, file, "water")
                logger.debug(f"water_path: {water_path}")
                break
        logger.info(f"{location} extracted")
        merge_shapefile(base_road_filepath, natural_path)
        logger.info(f"{location} merged")
    app.exitQgis()

if __name__ == "__main__":
    __main__()