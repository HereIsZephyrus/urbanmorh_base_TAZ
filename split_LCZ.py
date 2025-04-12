from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
from qgis.core import *
import os
import logging

logging.basicConfig(filename='split_LCZ.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
        logger.debug(f"ds_path: {ds_path}")
        if os.path.exists(ds_path):
            driver.DeleteDataSource(ds_path)
        ds = driver.CreateDataSource(ds_path)
        src = osr.SpatialReference()
        src.ImportFromWkt(spatialRef.ExportToWkt())
    except Exception as e:
        logger.error(f"error creating output shapefile: {e}")

    try:
        boundaries_layer = ds.CreateLayer(f"{split_type}_boundaries", src, ogr.wkbLineString)
        if boundaries_layer is None:
            logger.error("error creating natural boundaries layer")
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
        logger.error(f"error creating natural boundaries layer: {e}")
    
    return ds_path

def delete_shapefile(shp_path):
    shp_dir = os.path.dirname(shp_path)
    shp_name = os.path.basename(shp_path).split('.')[0] + '.'
    for file in os.listdir(shp_dir):
        if file.startswith(shp_name):
            os.remove(os.path.join(shp_dir, file))

def rename_shapefile(shp_path, new_name):
    shp_dir = os.path.dirname(shp_path)
    shp_name = os.path.basename(shp_path).split('.')[0] + '.'
    new_name = new_name + '.'
    for file in os.listdir(shp_dir):
        if file.startswith(shp_name):
            os.rename(os.path.join(shp_dir, file), os.path.join(shp_dir, new_name))

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
    proj_filter_path = reproject_shapefile(filter_path)
    logger.debug(f"proj_filter_path: {proj_filter_path}")
    return proj_filter_path

def filter_lcz_vectors(splited_road_path, feature_path):
    splited_road_centroid_path = calc_line_centroid(splited_road_path)
    logger.debug(f"splited_road_centroid_path: {splited_road_centroid_path}")
    distance_extracted_centroid_path = extract_whithindistance(splited_road_centroid_path, feature_path, 20)
    logger.debug(f"distance_extracted_centroid_path: {distance_extracted_centroid_path}")
    joined_splited_road_path = join_by_attribute(splited_road_path, distance_extracted_centroid_path)
    logger.debug(f"joined_splited_road_path: {joined_splited_road_path}")
    filtered_splited_road_path = extract_nonull_attribute(joined_splited_road_path, "FID_2")
    logger.debug(f"filtered_splited_road_path: {filtered_splited_road_path}")
    delete_shapefile(splited_road_path)
    delete_shapefile(splited_road_centroid_path)
    delete_shapefile(distance_extracted_centroid_path)
    delete_shapefile(joined_splited_road_path)
    return filtered_splited_road_path

def mess_up_splited_feature(splited_path):
    # delete all attributes except FID
    layer = QgsVectorLayer(splited_path, "splited_path", "ogr")
    data_provider = layer.dataProvider()
    field_indices = list(range(len(data_provider.fields())))
    data_provider.deleteAttributes(field_indices)
    layer.updateFields()
    data_provider.addAttributes([construct_index_field("FID")])
    layer.updateFields()
    layer.commitChanges()
    reindex_feature(splited_path,"FID")

def exclude_bivertices(point_intersection_path, field_name):
    layer = QgsVectorLayer(point_intersection_path, "point_intersection", "ogr")
    stack = []
    fid_index = layer.fields().indexOf(field_name)
    if fid_index == -1:
        logger.error(f"field {field_name} not found")
        return []
    last_fid = 0
    for feature in layer.getFeatures():
        fid = feature.attributes()[fid_index]
        if (fid == last_fid):
            stack.append(fid)
        last_fid = fid
    return stack

def calc_remained_road(splited_with_feature_path, exclude_list):
    layer = QgsVectorLayer(splited_with_feature_path, "splited_with_feature", "ogr")
    if (len(exclude_list) == 0):
        logger.warning(f"exclude_list is empty")
        return splited_with_feature_path
    
    # select not in exclude_list
    remained_list = []
    scan_index = 0
    fid_field_index = layer.fields().indexOf("FID")
    for feature in layer.getFeatures():
        current_fid = feature.attributes()[fid_field_index]
        if (current_fid < exclude_list[scan_index]):
            remained_list.append(current_fid)
        elif (current_fid == exclude_list[scan_index]):
            scan_index += 1
            if (scan_index == len(exclude_list)):
                scan_index -= 1
        else:
            remained_list.append(current_fid)

    request = QgsFeatureRequest().setFilterFids(remained_list)
    output_layer = layer.materialize(request)
    output_path = generate_save_path(splited_with_feature_path, "selected")
    if os.path.exists(output_path):
        delete_shapefile(output_path)

    transform_context = QgsProject.instance().transformContext()
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "utf-8"

    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        output_layer,
        output_path,
        transform_context,
        options,
    )
    if (error[0] != 0):
        logger.error(f"error writing output layer: {error}")
        return ""
    return output_path

def post_process_road(road_filepath, feature_path, intersection_with_feature_path):
    dissolved_splited_road_path = dissolve_shapefile(road_filepath)
    logger.debug(f"dissolved_splited_road_path: {dissolved_splited_road_path}")
    splited_dissove_path = split_line_with_line(dissolved_splited_road_path, dissolved_splited_road_path)
    logger.debug(f"splited_dissove_path: {splited_dissove_path}")
    splited_with_feature_path = split_line_with_line(splited_dissove_path, feature_path)
    logger.debug(f"splited_with_feature_path: {splited_with_feature_path}")
    mess_up_splited_feature(splited_with_feature_path)
    logger.debug(f"mess_up_splited_feature_path")
    endpoint_path = specific_vertices(splited_with_feature_path)
    logger.debug(f"endpoint_path: {endpoint_path}")
    create_spatial_index(endpoint_path)
    create_spatial_index(intersection_with_feature_path)
    point_intersection_path = extract_whithindistance(endpoint_path, intersection_with_feature_path, 2)
    logger.debug(f"point_intersection_path: {point_intersection_path}")
    exclude_list = exclude_bivertices(point_intersection_path, "FID")
    remained_road_path = calc_remained_road(splited_with_feature_path, exclude_list)
    logger.debug(f"remained_road_path: {remained_road_path}")
    delete_shapefile(dissolved_splited_road_path)
    delete_shapefile(splited_dissove_path)
    delete_shapefile(endpoint_path)
    return remained_road_path

def merge_vector(road_filepath, feature_path):
    layer_list = [road_filepath, feature_path]
    merged_path = merge_layers(layer_list, 32650)
    dissolved_path = dissolve_shapefile(merged_path)
    rename_shapefile(dissolved_path, "boundary")
    return merged_path

def merge_shapefile(base_road_filepath, feature_path):
    splited_road_path = split_lines(base_road_filepath, feature_path)
    logger.debug(f"splited_road_path: {splited_road_path}")
    intersection_with_feature_path = calc_line_intersection(splited_road_path, feature_path)
    logger.debug(f"intersection_with_feature_path: {intersection_with_feature_path}")
    filtered_splited_road_path = filter_lcz_vectors(splited_road_path, feature_path)
    processed_road_path = post_process_road(filtered_splited_road_path, feature_path, intersection_with_feature_path)
    merged_vector_path = merge_vector(processed_road_path, feature_path)
    return merged_vector_path

def search_tif(location_dir):
    for file in os.listdir(location_dir):
        if file.endswith(".tif"):
            return file
    return None

def __main__():
    app.initQgis()
    base_road_filepath = copy_base_road()
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        logger.info(f"processing {location}")

        tif_path = search_tif(location_dir)
        natural_path = create_mask(location_dir, tif_path, "natural")
        logger.debug(f"natural_path: {natural_path}")
        water_path = create_mask(location_dir, tif_path, "water")
        logger.debug(f"water_path: {water_path}")

        merge_shapefile(base_road_filepath, natural_path)
        logger.info(f"{location} merged")
    app.exitQgis()

if __name__ == "__main__":
    __main__()