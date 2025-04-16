from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
from qgis.core import *
from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator
from rich.progress import Progress as ProgressBar
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

QGIS_PREFIX_PATH = os.environ.get("QGIS_PREFIX_PATH")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
app = QgsApplication([], False)
app.initQgis()
from construct_qgis_functions import *

logger.info("QGIS initialized")

def copy_base_road():
    base_road_filepath = os.getenv("OSM_ROAD")
    current_filepath = os.path.join(os.getcwd(), "road.shp")
    reproject_shapefile(base_road_filepath, current_filepath)
    return current_filepath

def generate_expression(layer_name, split_type):
    split_operand_mapper = {
        "natural": ["!=4"],
        "water": ["=3","=7"]
    }
    operands = split_operand_mapper[split_type]
    for operand, index in enumerate(operands):
        if index == 0:
            expression = f'"{layer_name}@1"{operand}'
        else:
            expression += f' or "{layer_name}@1"{operand}'

    return expression

def split_tiff(location_dir, file, split_type):
    file_path = os.path.join(location_dir, file)
    LCZtiff = gdal.Open(file_path)
    LCZdata = LCZtiff.GetRasterBand(1)
    spatialRef = LCZtiff.GetSpatialRef()
    split_level_mapper = {
        "natural": [10],
        "water": [16.1]
    }

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
    tif_location = os.path.join(location_dir, file_name)
    layer_name = "rawtiff_layer"
    rawtiff_layer = QgsRasterLayer(tif_location, layer_name)
    if not rawtiff_layer.isValid():
        logger.error(f"rawtiff_layer is not valid")
        return ""
    
    entry = QgsRasterCalculatorEntry()
    entry.ref = 'rawtiff_layer@1'  # 引用名称
    entry.raster = rawtiff_layer
    entry.bandNumber = 1
    entries = [entry]
    output_path = os.path.join(location_dir, f"{split_type}_mask.tif")
    expression = generate_expression(layer_name, split_type)
    print(expression)
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

def filter_lcz_vectors(splited_road_path, feature_path):
    splited_road_centroid_path = calc_line_centroid(splited_road_path)
    logger.debug(f"splited_road_centroid_path: {splited_road_centroid_path}")
    distance_extracted_centroid_path = extract_whithindistance(splited_road_centroid_path, feature_path, 20)
    logger.debug(f"distance_extracted_centroid_path: {distance_extracted_centroid_path}")
    joined_splited_road_path = join_by_attribute(splited_road_path, distance_extracted_centroid_path, "monoid")
    proj_joined_splited_road_path = reproject_shapefile(joined_splited_road_path)
    mess_up_splited_feature(proj_joined_splited_road_path)
    return proj_joined_splited_road_path

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
    output_path = generate_save_path(splited_with_feature_path, "" ,"selected")
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
    fixed_merged_path = fix_geometry(merged_path)
    dissolved_path = dissolve_shapefile(fixed_merged_path)
    return dissolved_path

def merge_road(base_road_filepath, natural_mask_path, natural_path, water_path):
    splited_road_path = split_lines(base_road_filepath, natural_path)
    logger.debug(f"splited_road_path: {splited_road_path}")
    intersection_with_feature_path = calc_line_intersection(splited_road_path, natural_path)
    logger.debug(f"intersection_with_feature_path: {intersection_with_feature_path}")
    filtered_splited_road_path = filter_lcz_vectors(splited_road_path, natural_path)
    logger.debug(f"filtered_splited_road_path: {filtered_splited_road_path}")
    processed_road_path = post_process_road(filtered_splited_road_path, natural_path, intersection_with_feature_path)
    logger.debug(f"processed_road_path: {processed_road_path}")
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
    return merged_vector_path

def parallel_mingle_road(polygon_layer, fields, request, area_threshold, minor_area_threshold, TAZ_raster_path):
    def merged_into_closest_neighbor(feature, polygon_index, area_threshold, minor_area_threshold):
        fid = feature.id()
        feature_geom = feature.geometry()
        feature_area = feature_geom.area()
        sample_field_index = fields.indexOf("SAMPLE")
        sample_value = feature.attributes()[sample_field_index]
        if sample_value == 0:
            logger.warning(f"feature {fid} has no sample value")
            return None
        if feature_area > area_threshold:
            return None
        bbox = feature_geom.boundingBox()
        bbox.grow(2)
        candidates = polygon_index.intersects(bbox)
        if len(candidates) == 0:
            return None
        closest_neighbor_fid = 0
        min_distance = 0
        for candidate_fid in candidates:
            if candidate_fid == feature.id():
                continue
            candidate = polygon_layer.getFeature(candidate_fid)
            shared_edge = feature_geom.intersection(candidate.geometry())
            if shared_edge.type() != QgsWkbTypes.LineGeometry or shared_edge.type() != QgsWkbTypes.MultiLineGeometry:
                continue
            candidate_sample_value = candidate.attributes()[sample_field_index]
            if candidate_sample_value == 0:
                continue
            sample_distance = abs(sample_value - candidate_sample_value)
            if sample_distance < min_distance:
                min_distance = sample_distance
                closest_neighbor_fid = candidate_fid
                if feature_area < minor_area_threshold:
                    break
        return tuple(fid, closest_neighbor_fid)
    
    def update_feature(id_pair, polygon_layer, fields, monitor):
        if id_pair is None:
            return False
        minor_feature_id, major_feature_id = id_pair
        minor_feature = polygon_layer.getFeature(minor_feature_id)
        major_feature = polygon_layer.getFeature(major_feature_id)
        minor_geom = minor_feature.geometry()
        major_geom = major_feature.geometry()
        merged_geom = minor_geom.combine(major_geom)
        new_feature = QgsFeature(fields)
        new_feature.setGeometry(merged_geom)
        new_feature['FID'] = major_feature_id
        new_feature['area'] = minor_geom.area()
        polygon_layer.addFeature(new_feature)
        #polygon_layer.deleteFeature(minor_feature_id) lazy delete
        monitor.update(task, advance=1, description=f"Adjusting roads ...")
        return True

    #calculate the sample number on the polygon layer
    polygon_index = QgsSpatialIndex(polygon_layer)
    sample_in_polygon_layer = sample_in_polygon(polygon_layer, point_inside_number=10, min_distance_inside=200, seed=None)
    logger.debug(f"sampled in polygon layer")
    sample_on_LCZ_layer = sample_raster_values(TAZ_raster_path, sample_in_polygon_layer, None)
    logger.debug(f"sampled on LCZ layer")
    result = processing.run("native:aggregate", {'INPUT':sample_on_LCZ_layer,
        'GROUP_BY':'"FID"',
        'AGGREGATES':[{'aggregate': 'first_value','delimiter': ',','input': '"FID"','length': 11,'name': 'FID','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},
        {'aggregate': 'first_value','delimiter': ',','input': '"area"','length': 20,'name': 'area','precision': 10,'sub_type': 0,'type': 6,'type_name': 'double precision'},
        {'aggregate': 'mean','delimiter': ',','input': '"SAMPLE_1"','length': 5,'name': 'SAMPLE','precision': 2,'sub_type': 0,'type': 6,'type_name': 'double precision'}],
        'OUTPUT':'TEMPORARY_OUTPUT'
    })
    if 'error' in result:
        logger.error(f"error in aggregate: {result['error']}")
        return 0
    aggregateed_sample_layer = result['OUTPUT']
    logger.debug(f"aggregated sample layer")
    result = processing.run("native:joinattributestable", {
        'INPUT':aggregateed_sample_layer,'FIELD':'FID','INPUT_2':aggregateed_sample_layer,
        'FIELD_2':'FID','FIELDS_TO_COPY':['SAMPLE'],'METHOD':1,'DISCARD_NONMATCHING':False,'PREFIX':'',
        'OUTPUT':'TEMPORARY_OUTPUT'
    })
    if 'error' in result:
        logger.error(f"error in join attribute table: {result['error']}")
        return 0
    joined_polygon_layer = result['OUTPUT']
    logger.debug(f"joined sample value into polygon layer")

    adjusted_num = 0
    with ProgressBar() as monitor:
        merge_func = partial(merged_into_closest_neighbor, polygon_index = polygon_index, area_threshold = area_threshold, minor_area_threshold = minor_area_threshold)
        update_func = partial(update_feature, polygon_layer = joined_polygon_layer, fields = fields, monitor = monitor)
        task = monitor.add_task("Adjusting road", total=None)
        polygon_index = QgsSpatialIndex(joined_polygon_layer)
        with edit(joined_polygon_layer):
            with ThreadPoolExecutor() as executor:
                pairs = executor.map(merge_func, joined_polygon_layer.getFeatures(request))
                logger.debug(f"finish pair search")
                results = executor.map(update_func, pairs)
                logger.debug(f"finish feature add")
                for result in results:
                    if result:
                        adjusted_num += 1
    polygon_layer_path = joined_polygon_layer.source()
    aggregate_layer = aggregate_features(polygon_layer_path, 'FID', None)
    delete_shapefile(polygon_layer_path)

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "utf-8"
    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        aggregate_layer,
        polygon_layer_path,
        QgsProject.instance().transformContext(),
        options,
    )
    if (error[0] != 0):
        logger.error(f"error writing output layer: {error}")
        return ""
    polygon_layer.reload()
    logger.debug("reloaded polygon layer")
    return adjusted_num

def sequence_mingle_road(polygon_layer, fields, request, area_threshold, minor_area_threshold):
    polygon_index = QgsSpatialIndex(polygon_layer)    
    adjusted_num = 0
    with ProgressBar() as monitor:
        task = monitor.add_task("Adjusting road", total=100)
        for feature in polygon_layer.getFeatures(request):
            feature_geom = feature.geometry()
            feature_area = feature_geom.area()
            if feature_area > area_threshold:
                continue
            monitor.update(task, completed=feature_area / area_threshold * 100, description=f"Adjusting road {feature_area}")
            bbox = feature_geom.boundingBox()
            bbox.grow(2)
            candidates = polygon_index.intersects(bbox)
            if len(candidates) == 0:
                continue
            largest_shared_edge_fid = 0
            max_shared_edge_length = 0
            for candidate_fid in candidates:
                if candidate_fid == feature.id():
                    continue
                candidate = polygon_layer.getFeature(candidate_fid)
                shared_edge = feature_geom.intersection(candidate.geometry())
                if (shared_edge.type() != QgsWkbTypes.LineGeometry or shared_edge.type() != QgsWkbTypes.MultiLineGeometry):
                    logger.warning(f"{feature.id()}-{candidate_fid}'s shared_edge is not a line")
                    continue
                if shared_edge.length() > max_shared_edge_length:
                    max_shared_edge_length = shared_edge.length()
                    largest_shared_edge_fid = candidate_fid
                    if feature_area < minor_area_threshold:
                        break
            if (largest_shared_edge_fid == 0):
                continue
            with edit(polygon_layer):
                task.update(completed=feature_area / area_threshold * 100, description=f"Adjusting road {feature.id()} with {largest_shared_edge_fid}")
                largest_shared_edge = polygon_layer.getFeature(largest_shared_edge_fid)
                largest_shared_edge_geom = largest_shared_edge.geometry()
                # merge the feature into the largest candidate
                merged_geom = feature_geom.combine(largest_shared_edge_geom)
                new_feature = QgsFeature(fields)
                new_feature.setGeometry(merged_geom)
                new_feature['FID'] = largest_shared_edge.id()
                new_feature['area'] = merged_geom.area()
                polygon_layer.addFeature(new_feature)
                polygon_layer.deleteFeature(feature.id())
                adjusted_num += 1
    return adjusted_num
        
def adjust_road(road_feature_path, TAZ_raster_path):    
    polygonized_path = direct_polygonize(road_feature_path)
    logger.debug(f"polygonized_path: {polygonized_path}")
    reindex_feature(polygonized_path, "FID")
    calc_area(polygonized_path)
    logger.debug(f"calculated area")
    create_spatial_index(polygonized_path)
    logger.debug(f"created layer's spatial index")
    sorted_polygonized_path = sort_features(polygonized_path, "area", True)
    logger.debug(f"sorted_polygonized_path: {sorted_polygonized_path}")
    polygon_layer = QgsVectorLayer(sorted_polygonized_path, "polygon_layer", "ogr")
    fields = polygon_layer.fields()
    request = QgsFeatureRequest().setFilterExpression(f"area < {area_threshold}")
    if not polygon_layer.isValid():
        logger.error(f"polygon_layer is not valid")
        return ""
    area_threshold = 50000
    minor_area_threshold = 200
    iteration = 0
    while True:
        iteration += 1
        logger.debug(f"iteration {iteration}")
        #adjusted_num = sequence_mingle_road(polygon_layer = polygon_layer, 
        #                                    fields = fields, 
        #                                    request = request, 
        #                                    area_threshold = area_threshold, 
        #                                    minor_area_threshold = minor_area_threshold)
        adjusted_num = parallel_mingle_road(polygon_layer = polygon_layer, 
                                            fields = fields, 
                                            request = request, 
                                            area_threshold = area_threshold, 
                                            minor_area_threshold = minor_area_threshold,
                                            TAZ_raster_path = TAZ_raster_path)
        if adjusted_num == 0:
            logger.debug(f"no more adjustment")
            break
        logger.debug(f"adjusted_num: {adjusted_num}")
        reindex_feature(polygonized_path, "FID")

    logger.debug(f"polygonized_path: {polygonized_path}")
    adjust_road_path = polygon_to_line(polygonized_path)
    return adjust_road_path

def split_road_by_purity(adjust_road_path, LCZ_raster_path, boundary_city_path):
    polygon_layer = convert_line_to_polygon(adjust_road_path,None)
    logger.debug(f"created polygon layer")
    grid_point_layer = create_grid_point(polygon_layer, spacing = 100, output_feature_path = None)
    logger.debug(f"grid_point_layer: {grid_point_layer}")
    params = {'INPUT':grid_point_layer,
              'OVERLAY':adjust_road_path,
              'INPUT_FIELDS':['SAMPLE_1'],
              'OVERLAY_FIELDS':['FID'],
              'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT',
              'GRID_SIZE':None}
    result = processing.run("native:intersection", params)
    if 'error' in result:
        logger.error(f"error in intersection: {result['error']}")
        return ""
    sample_layer = result['OUTPUT']
    logger.debug(f"sample_layer: {sample_layer}")
    with_purity_layer = calc_purity(polygon_layer,sample_layer,None)
    logger.debug(f"calculated purity")
    
    return result['OUTPUT']

def split_road_by_geometry(adjust_road_path):
    return adjust_road_path

def process_location(location_dir, base_road_filepath, lcz_path, landcover_path, boundary_city_path):
    natural_mask_path = create_image_mask(location_dir, landcover_path, "natural")
    buffered_natural_mask_path = create_boundary_mask(natural_mask_path, buffer_size = 20)
    logger.info(f"buffered_natural_mask_path: {buffered_natural_mask_path}")
    #natural_path = create_contour_mask(location_dir, lcz_path, "natural")
    natural_path = create_boundary_mask(natural_mask_path, buffer_size = 10)
    logger.info(f"natural_path: {natural_path}")
    water_mask_path = create_image_mask(location_dir, landcover_path, "water")
    logger.info(f"water_mask_path: {water_mask_path}")
    water_path = create_boundary_mask(water_mask_path, buffer_size = 10)
    logger.info(f"water_path: {water_path}")

    merged_vector_path = merge_road(base_road_filepath, buffered_natural_mask_path, natural_path, water_path)
    logger.info(f"merged_vector_path: {merged_vector_path}")
    simplified_merged_vector_path = simplify_road(merged_vector_path)
    logger.info(f"simplified_merged_vector_path: {simplified_merged_vector_path}")
    adjust_road_path = adjust_road(simplified_merged_vector_path, lcz_path)
    logger.info(f"adjust_road_path: {adjust_road_path}")
    pure_road_path = split_road_by_purity(adjust_road_path, lcz_path, boundary_city_path)
    logger.info(f"pure_road_path: {pure_road_path}")
    adjust_pure_road_path = adjust_road(pure_road_path, lcz_path)
    logger.info(f"adjust_pure_road_path: {adjust_pure_road_path}")
    geometry_road_path = split_road_by_geometry(adjust_pure_road_path)
    logger.info(f"geometry_road_path: {geometry_road_path}")
    adjust_geometry_road_path = adjust_road(geometry_road_path, lcz_path)
    return adjust_geometry_road_path

def __main__():
    app.initQgis()
    total_base_road_filepath = copy_base_road()
    lcz_dir = os.getenv("LCZ_DIR")
    total_landcover_path = os.path.join(os.getenv("LAND_COVER_DIR"), os.getenv("COVER_YEAR") + ".tif")
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        logger.info(f"processing {location}")

        tif_path = os.path.join(location_dir, location + ".tif")
        boundary_city_path = os.path.join(location_dir, location + ".shp")
        logger.info(f"find tif file: {tif_path} and boundary city file: {boundary_city_path}")
        
        boundary_line_path = polygon_to_line(boundary_city_path)
        base_road_filepath = merge_layers([boundary_line_path, clip_vector(total_base_road_filepath, boundary_city_path,None)],4326)
        logger.info(f"base_road_filepath: {base_road_filepath}")

        cliped_tif_path = clip_raster(tif_path, boundary_city_path)
        logger.info(f"cliped_LCZ_tif_path: {cliped_tif_path}")

        cliped_landcover_path = clip_raster(total_landcover_path, boundary_city_path)
        logger.info(f"cliped_landcover_path: {cliped_landcover_path}")

        result = process_location(location_dir, base_road_filepath, cliped_tif_path, cliped_landcover_path, boundary_city_path)
        logger.info(f"result: {result}")

    app.exitQgis()

if __name__ == "__main__":
    __main__()