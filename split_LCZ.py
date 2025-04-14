from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
from qgis.core import *
from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator
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
os.environ["QT_QPA_PLATFORM"] = "offscreen"
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
app = QgsApplication([], False)
app.initQgis()
from construct_qgis_functions import *

logger.info("QGIS initialized")
split_level_mapper = {
    "natural": [10],
    "water": [16.1]
}
split_operand_mapper = {
    "natural": "> 10",
    "water": "= 17"
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

def create_contour_mask(location_dir, file_name, split_type):
    contour_path = split_tiff(location_dir, file_name, split_type)
    logger.debug(f"contour_path: {contour_path}")
    filter_path = delete_small_features(contour_path)
    logger.debug(f"filter_path: {filter_path}")
    proj_filter_path = reproject_shapefile(filter_path)
    logger.debug(f"proj_filter_path: {proj_filter_path}")
    return proj_filter_path

def create_image_mask(location_dir, file_name, split_type):
    mask_operand = split_operand_mapper[split_type]
    tif_location = os.path.join(location_dir, file_name)
    rawtiff_layer = QgsRasterLayer(tif_location, "rawtiff_layer")
    if not rawtiff_layer.isValid():
        logger.error(f"rawtiff_layer is not valid")
        return ""
    
    entry = QgsRasterCalculatorEntry()
    entry.ref = 'rawtiff_layer@1'  # 引用名称
    entry.raster = rawtiff_layer
    entry.bandNumber = 1
    entries = [entry]
    output_path = os.path.join(location_dir, f"{split_type}_mask.tif")
    expression = f'"rawtiff_layer@1" {mask_operand}'
    calc = QgsRasterCalculator(expression, output_path, 'GTiff',
                            rawtiff_layer.extent(), 
                            rawtiff_layer.crs(),
                            rawtiff_layer.width(), 
                            rawtiff_layer.height(),
                            entries,
                            QgsProject.instance().transformContext()
                            )
    result = calc.processCalculation() 
    if result != 0:
        logger.error(f"error processing calculation: {result}")
        return ""
    logger.debug(f"tiff_path: {output_path}")
    polygonize_path = raster_to_vector(output_path)
    logger.debug(f"polygonize_path: {polygonize_path}")
    selected_poly_path = extract_by_value(polygonize_path, "DN", "=", 1)
    logger.debug(f"selected_poly_path: {selected_poly_path}")
    proj_selected_poly_path = reproject_shapefile(selected_poly_path)
    logger.debug(f"proj_selected_poly_path: {proj_selected_poly_path}")
    return proj_selected_poly_path

def create_boundary_mask(image_mask_path, buffer_size = 0):
    if (buffer_size > 0):
        buffer_path = create_buffer(image_mask_path, buffer_size)
        logger.debug(f"buffer_path: {buffer_path}")
        proj_line_path = polygon_to_line(buffer_path)
        logger.debug(f"proj_line_path: {proj_line_path}")
    else:
        proj_line_path = polygon_to_line(image_mask_path)
        logger.debug(f"proj_line_path: {proj_line_path}")
    singlepart_proj_line_path = multipart_to_singleparts(proj_line_path)
    logger.debug(f"singlepart_proj_line_path: {singlepart_proj_line_path}")
    filter_path = delete_small_features(singlepart_proj_line_path)
    logger.debug(f"filter_path: {filter_path}")
    return filter_path

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

def exclude_edges(point_intersection_path, edge_type, field_name):
    if (edge_type == "bial"):
        compare_operator = False
    elif (edge_type == "single"):
        compare_operator = True
    else:
        logger.error("invalid operator signal")
        return None
    layer = QgsVectorLayer(point_intersection_path, "point_intersection", "ogr")
    stack = []
    fid_index = layer.fields().indexOf(field_name)
    if fid_index == -1:
        logger.error(f"field {field_name} not found")
        return []
    last_fid = 0
    for feature in layer.getFeatures():
        fid = feature.attributes()[fid_index]
        if (compare_operator == False):
            if (last_fid == fid):
                stack.append(fid)
        else:
            if (last_fid == 0):
                stack.append(fid)
            elif (last_fid == fid):
                stack.pop()
            else:
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
    exclude_list = sorted(list(set(exclude_edges(point_intersection_path, "bial","FID")) | set(exclude_edges(endpoint_path, "single","FID"))))
    logger.debug(f"calculated exclude_list")
    remained_road_path = calc_remained_road(splited_with_feature_path, exclude_list)
    logger.debug(f"remained_road_path: {remained_road_path}")
    delete_shapefile(dissolved_splited_road_path)
    delete_shapefile(splited_dissove_path)
    #delete_shapefile(endpoint_path)
    return remained_road_path

def merge_vector(layer_list):
    merged_path = merge_layers(layer_list, 32650)
    dissolved_path = dissolve_shapefile(merged_path)
    rename_shapefile(dissolved_path, "boundary")
    return merged_path

def merge_road(base_road_filepath, natural_mask_path, natural_path, water_path):
    splited_road_path = split_lines(base_road_filepath, natural_path)
    logger.debug(f"splited_road_path: {splited_road_path}")
    intersection_with_feature_path = calc_line_intersection(splited_road_path, natural_path)
    logger.debug(f"intersection_with_feature_path: {intersection_with_feature_path}")
    filtered_splited_road_path = filter_lcz_vectors(splited_road_path, natural_path)
    processed_road_path = post_process_road(filtered_splited_road_path, natural_path, intersection_with_feature_path)
    masked_road_path = exclude_by_mask(processed_road_path, natural_mask_path)
    logger.debug(f"masked_road_path: {masked_road_path}")
    merged_vector_path = merge_vector([masked_road_path, natural_path, water_path])
    return merged_vector_path

def simplify_road(input_feature_path):
    smoothed_path = smooth_shapefile(input_feature_path, 0.4)
    logger.debug(f"smoothed_path: {smoothed_path}")
    single_simplified_path = simplify_shapefile(smoothed_path, 30)
    logger.debug(f"simplified_path: {single_simplified_path}")
    dissolved_single_simplified_path = dissolve_shapefile(single_simplified_path)
    logger.debug(f"dissolved_simplified_path: {dissolved_single_simplified_path}")
    exploded_single_simplified_path = explode_line(dissolved_single_simplified_path)
    logger.debug(f"exploded_single_simplified_path: {exploded_single_simplified_path}")
    vertices_path = all_vertices(exploded_single_simplified_path)
    logger.debug(f"vertices_path: {vertices_path}")
    create_spatial_index(vertices_path)
    create_spatial_index(exploded_single_simplified_path)
    shortest_line_path = shortest_line(vertices_path, exploded_single_simplified_path, 3, 80.0)
    logger.debug(f"shortest_line_path: {shortest_line_path}")
    merged_vector_path = merge_vector([shortest_line_path, exploded_single_simplified_path])
    logger.debug(f"merged_vector_path: {merged_vector_path}")
    dissolved_merged_vector_path = dissolve_shapefile(merged_vector_path)
    return dissolved_merged_vector_path

def adjust_road(road_feature_path):
    polygonized_path = convert_line_to_polygon(road_feature_path)
    reindex_feature(polygonized_path, "FID")
    calc_area(polygonized_path)
    logger.debug(f"calculated area")
    create_spatial_index(polygonized_path)
    polygon_layer = QgsVectorLayer(polygonized_path, "polygon_layer", "ogr")
    polygon_index = QgsSpatialIndex(polygon_layer)
    logger.debug(f"created layer's spatial index")
    if not polygon_layer.isValid():
        logger.error(f"polygon_layer is not valid")
        return ""
    request = QgsFeatureRequest().setFilterExpression("area < 100000")
    merge_count = 0
    for feature in polygon_layer.getFeatures(request):
        bbox = feature.geometry().boundingBox()
        bbox.grow(5)
        candidates = polygon_index.intersects(bbox)
        feature_geom = feature.geometry()
        shared_edge_dict = {}
        for candidate_fid in candidates:
            if candidate_fid == feature.id():
                continue
            candidate = polygon_layer.getFeature(candidate_fid)
            shared_edge = feature_geom.intersection(candidate.geometry())
            if shared_edge.type() != QgsWkbTypes.LineGeometry:
                logger.warning(f"{feature.id()}-{candidate_fid}'s shared_edge is not a line")
                continue
            if shared_edge.length() > 0:
                shared_edge_dict[candidate_fid] = shared_edge.length()
        # sort the shared_edge_dict by value to get the largest shared edge
        sorted_shared_edge_dict = sorted(shared_edge_dict.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_shared_edge_dict) > 0:
            with edit(polygon_layer):
                largest_shared_edge_fid = sorted_shared_edge_dict[0][0]
                largest_shared_edge_geom = polygon_layer.getFeature(largest_shared_edge_fid).geometry()
                # merge the feature into the largest candidate
                merged_geom = feature_geom.combine(largest_shared_edge_geom)
                new_feature = QgsFeature()
                new_feature.setGeometry(merged_geom)
                new_feature['area'] = merged_geom.area()
                polygon_layer.addFeature(new_feature)
                polygon_layer.deleteFeature(feature.id())
                merge_count += 1

    reindex_feature(polygonized_path, "FID")
    logger.info(f"merge_feature_count: {merge_count}")
    logger.debug(f"polygonized_path: {polygonized_path}")
    adjust_road_path = polygon_to_line(polygonized_path)
    return adjust_road_path

def __main__():
    app.initQgis()
    base_road_filepath = copy_base_road()
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        logger.info(f"processing {location}")

        tif_path = os.path.join(location_dir, location + ".tif")
        #natural_mask_path = create_image_mask(location_dir, tif_path, "natural")
        #buffered_natural_mask_path = create_boundary_mask(natural_mask_path, buffer_size = 50)
        #logger.info(f"buffered_natural_mask_path: {buffered_natural_mask_path}")
        #natural_path = create_contour_mask(location_dir, tif_path, "natural")
        #logger.info(f"natural_path: {natural_path}")
        #water_mask_path = create_image_mask(location_dir, tif_path, "water")
        #logger.info(f"water_mask_path: {water_mask_path}")
        #water_path = create_boundary_mask(water_mask_path, buffer_size = 10)
        #logger.info(f"water_path: {water_path}")

        #merged_vector_path = merge_road(base_road_filepath, buffered_natural_mask_path, natural_path, water_path)
        #logger.info(f"merged_vector_path: {merged_vector_path}")
        merged_vector_path = "/mnt/repo/YZB/TAZ/precise/LCZ/wuhan/natural_contour_f_p_roads_j_uni_d_s_s_selected_mask_m.shp"
        simplified_merged_vector_path = simplify_road(merged_vector_path)
        logger.info(f"simplified_merged_vector_path: {simplified_merged_vector_path}")
        #adjust_road_path = adjust_road(simplified_merged_vector_path)
    app.exitQgis()

if __name__ == "__main__":
    __main__()