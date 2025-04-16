from qgis.core import *
from qgis.PyQt.QtCore import QMetaType
import processing
from processing.core.Processing import Processing
Processing.initialize()
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial


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

def delete_shapefile(shp_path):
    shp_dir = os.path.dirname(shp_path)
    shp_name = os.path.basename(shp_path).split('.')[0] + '.'
    for file in os.listdir(shp_dir):
        if file.startswith(shp_name):
            os.remove(os.path.join(shp_dir, file))

def generate_save_path(origin_path, output_path, prefix = "", type = "vector"):
    if output_path is None:
        return 'TEMPORARY_OUTPUT'
    if type == "vector":
        extension = ".shp"
    elif type == "raster":
        extension = ".tif"
    else:
        extension = ""
    if output_path == "":
        if not isinstance(origin_path, str):
            origin_path = origin_path.source()
        dir = os.path.dirname(origin_path)
        name = os.path.basename(origin_path).split('.')[0]
        new_path = os.path.join(dir, f'{name}_{prefix}{extension}')
        return new_path
    else:
        return output_path

def run_processing_algorithm(algorithm, params):
    result = processing.run(algorithm, params)
    if 'error' in result:
        logger.error(f"error running the processing {algorithm}: {result['error']}")
        return ""
    return result['OUTPUT']

def construct_index_field(field_name):
    return QgsField(field_name, QMetaType.Type.Int)

def reindex_feature(feature_path, field_name):
    '''
    Use QGIS API to reindex the feature layer
    feature_path: the path of the feature shapefile
    field_name: the name of the field to reindex
    '''
    # add the field to the feature layer
    feature_layer = QgsVectorLayer(feature_path, 'Feature Layer', 'ogr')
    if not feature_layer.isValid():
        logger.error(f"feature_layer is not valid")
        return ""
    if field_name not in [f.name() for f in feature_layer.fields()]:
        feature_layer.dataProvider().addAttributes([construct_index_field(field_name)])
        feature_layer.updateFields()
    monoid_field_index = feature_layer.fields().indexOf(field_name)
    monoid_map = {
        f.id(): {monoid_field_index: f.id()}
        for f in feature_layer.getFeatures()
    }
    feature_layer.dataProvider().changeAttributeValues(monoid_map)
    return feature_path

def reproject_shapefile(geo_shp_path,proj_shp_path = ""):
    '''
    convert WGS84 to WGS84 UTM50N
    geo_shp_path: the path of the original shapefile
    proj_shp_path: the path of the projected shapefile
    if proj_shp_path is not provided, the projected shapefile will be saved in the same directory as the original shapefile
    output: the path of the projected shapefile
    '''
    proj_shp_path = generate_save_path(geo_shp_path, proj_shp_path, "p")
    if os.path.exists(proj_shp_path):
        delete_shapefile(proj_shp_path)
    crs_params = {
        'CONVERT_CURVED_GEOMETRIES' : False,
        'INPUT' : geo_shp_path,
        'OPERATION' : '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=50 +ellps=WGS84',
        'OUTPUT' : proj_shp_path,
        'TARGET_CRS' : QgsCoordinateReferenceSystem('EPSG:32650')
    }
    if not run_processing_algorithm("native:reprojectlayer", crs_params):
        return ""
    return proj_shp_path

def filter_remain_field(proj_poly_path, line_path, filter_path = ""):
    '''
    Use QGIS API to filter the line layer by the remain field
    proj_poly_path: the path of the projected polygon shapefile
    line_path: the path of the line shapefile
    filter_path: the path of the filtered line shapefile
    if filter_path is not provided, the filtered line shapefile will be saved in the same directory as the line shapefile
    output: the path of the filtered line shapefile
    '''
    def determine_remain(area):
        if area > 100000: # 0.1 km2
            return 1
        else:
            return 0

    filter_path = generate_save_path(line_path, filter_path, "f")
    if os.path.exists(filter_path):
        delete_shapefile(filter_path)
    # iterate through the polygon layer and add the area attribute
    poly_layer = QgsVectorLayer(proj_poly_path, 'Multipolygon Layer', 'ogr')
    if not poly_layer.isValid():
        logger.error("Layer failed to load!")
        return ""
    line_layer = QgsVectorLayer(line_path, 'Line Layer', 'ogr')
    if not line_layer.isValid():
        logger.error("Line layer failed to load!")
        return ""
    line_layer.dataProvider().addAttributes([QgsField("remain", QMetaType.Type.Int)])
    line_layer.updateFields()
    remain_field = line_layer.fields().indexOf("remain")
    with edit(line_layer):
        for feature, line_feature in zip(poly_layer.getFeatures(), line_layer.getFeatures()):
            area = feature.geometry().area()
            line_layer.changeAttributeValue(line_feature.id(), remain_field, determine_remain(area))

    filter_params = {
        'FIELD' : 'remain',
        'INPUT' : line_path,
        'OPERATOR' : 0,
        'OUTPUT' : filter_path,
        'VALUE' : '1'
    }
    if not run_processing_algorithm("qgis:extractbyattribute", filter_params):
        return ""
    return filter_path

def convert_line_to_polygon(line_path, poly_path = ""):
    '''
    Use QGIS API to convert line to polygon
    line_path: the path of the line shapefile
    poly_path: the path of the polygon shapefile
    if poly_path is not provided, the polygon shapefile will be saved in the same directory as the line shapefile
    output: the path of the polygon shapefile
    '''
    poly_path = generate_save_path(line_path, poly_path, "poly")
    if os.path.exists(poly_path):
        delete_shapefile(poly_path)
    convert_params = {
        "INPUT": line_path,
        "OUTPUT": poly_path
    }
    if not run_processing_algorithm("qgis:linestopolygons", convert_params):
        return ""
    return poly_path

def split_lines(base_road_filepath, feature_path, splited_road_path = ""):
    '''
    Use QGIS API to split the line layer by the feature layer
    base_road_filepath: the path of the base road shapefile
    feature_path: the path of the feature shapefile
    splited_road_path: the path of the splited road shapefile
    if splited_road_path is not provided, the splited road shapefile will be saved in the same directory as the feature shapefile
    output: the path of the splited road shapefile
    '''
    splited_road_path = generate_save_path(feature_path, splited_road_path, "roads")
    if os.path.exists(splited_road_path):
        delete_shapefile(splited_road_path)
    merge_params = {
        'INPUT' : base_road_filepath,
        'LINES' : feature_path,
        'OUTPUT' : splited_road_path
    }
    if not run_processing_algorithm("native:splitwithlines", merge_params):
        return ""

    reindex_feature(splited_road_path,"monoid")
    return splited_road_path

def calc_line_centroid(line_path, centroid_path = ""):
    '''
    Use QGIS API to calculate the centroid of the line layer
    line_path: the path of the line shapefile
    centroid_path: the path of the centroid shapefile
    if centroid_path is not provided, the centroid shapefile will be saved in the same directory as the line shapefile
    output: the path of the centroid shapefile
    '''
    centroid_path = generate_save_path(line_path, centroid_path, "c")
    if os.path.exists(centroid_path):
        delete_shapefile(centroid_path)
    calc_params = {
        'ALL_PARTS' : True,
        'INPUT' : line_path,
        'OUTPUT' : centroid_path
    }
    if not run_processing_algorithm("native:centroids", calc_params):
        return ""
    return centroid_path

def create_spatial_index(feature_path):
    '''
    Use QGIS API to create a spatial index for the feature layer
    feature_path: the path of the feature shapefile
    '''
    if not run_processing_algorithm("native:createspatialindex", {'INPUT' : feature_path}):
        return False
    return True

def join_by_attribute(input_feature_path, add_feature_path, field_name, join_path = ""):
    '''
    Use QGIS API to join the input feature layer by the add feature layer
    input_feature_path: the path of the input feature shapefile
    add_feature_path: the path of the add feature shapefile
    field_name: the name of the field to join
    if join_path is not provided, the joined shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the joined shapefile
    '''
    join_path = generate_save_path(input_feature_path, join_path, "j")
    if os.path.exists(join_path):
        delete_shapefile(join_path)
    join_params = {
        'DISCARD_NONMATCHING' : False,
        'FIELD' : field_name,
        'FIELDS_TO_COPY' : [],
        'FIELD_2' : field_name,
        'INPUT' : input_feature_path,
        'INPUT_2' : add_feature_path,
        'METHOD' : 1,
        'OUTPUT' : join_path,
        'PREFIX' : '' 
    }
    return run_processing_algorithm("native:joinattributestable", join_params)


def extract_nonull_attribute(input_feature_path, field_name, extracted_path = ""):
    '''
    Use QGIS API to extract the non-null attribute of the input feature layer
    input_feature_path: the path of the input feature shapefile
    field_name: the name of the field to extract
    extracted_path: the path of the extracted shapefile
    if extracted_path is not provided, the extracted shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the extracted shapefile
    '''
    extracted_path = generate_save_path(input_feature_path, extracted_path, "uni")
    if os.path.exists(extracted_path):
        delete_shapefile(extracted_path)
    extract_params = {
        'FIELD' : field_name,
        'INPUT' : input_feature_path,
        'OPERATOR' : 8,
        'OUTPUT' : extracted_path,
        'VALUE' : ''
    }
    return run_processing_algorithm("native:extractbyattribute", extract_params)

def extract_by_value(input_feature_path, field_name, operator, value, extracted_path = ""):
    '''
    Use QGIS API to extract the input feature layer by the attribute
    input_feature_path: the path of the input feature shapefile
    field_name: the name of the field to extract
    operator: the operator to use
    value: the value to use
    extracted_path: the path of the extracted shapefile
    if extracted_path is not provided, the extracted shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the extracted shapefile
    '''
    extracted_path = generate_save_path(input_feature_path, extracted_path, "flt")
    if operator == "=":
        operator = 0
    elif operator == "!=":
        operator = 1
    elif operator == ">":
        operator = 2
    elif operator == ">=":
        operator = 3
    elif operator == "<":
        operator = 4
    elif operator == "<=":
        operator = 5
    if os.path.exists(extracted_path):
        delete_shapefile(extracted_path)
    extract_params = {
        'FIELD' : field_name,
        'INPUT' : input_feature_path,
        'OPERATOR' : 0,
        'OUTPUT' : extracted_path,
        'VALUE' : value
    }
    return run_processing_algorithm("native:extractbyattribute", extract_params)

def extract_whithindistance(input_feature_path, compare_feature_path, distance, distance_extracted_path = ""):
    '''
    Use QGIS API to extract the input feature layer within the distance of the compare feature layer
    input_feature_path: the path of the input feature shapefile
    compare_feature_path: the path of the compare feature shapefile
    distance_extracted_path: the path of the distance extracted shapefile
    if distance_extracted_path is not provided, the distance extracted shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the distance extracted shapefile
    '''
    distance_extracted_path = generate_save_path(input_feature_path, distance_extracted_path, "distance")
    if os.path.exists(distance_extracted_path):
        delete_shapefile(distance_extracted_path)
    extract_params = {
        'DISTANCE' : distance,
        'INPUT' : input_feature_path,
        'REFERENCE' : compare_feature_path,
        'OUTPUT' : distance_extracted_path
    }
    create_spatial_index(input_feature_path)
    create_spatial_index(compare_feature_path)
    return run_processing_algorithm("native:extractwithindistance", extract_params)

def dissolve_shapefile(input_feature_path, dissolved_path = ""):
    '''
    Use QGIS API to dissolve the input feature layer
    input_feature_path: the path of the input feature shapefile
    dissolved_path: the path of the dissolved shapefile
    if dissolved_path is not provided, the dissolved shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the dissolved shapefile
    '''
    dissolved_path = generate_save_path(input_feature_path, dissolved_path, "d")
    if os.path.exists(dissolved_path):
        delete_shapefile(dissolved_path)
    dissolve_params = {
        'FIELD' : [],
        'INPUT' : input_feature_path,
        'OUTPUT' : dissolved_path,
        'SEPARATE_DISJOINT' : False
    }
    return run_processing_algorithm("native:dissolve", dissolve_params)

def split_line_with_line(line_path, overlap_line_path, splited_line_path = ""):
    '''
    Use QGIS API to split the line layer by the overlap line layer
    line_path: the path of the line shapefile
    overlap_line_path: the path of the overlap line shapefile
    if splited_line_path is not provided, the splited line shapefile will be saved in the same directory as the line shapefile
    output: the path of the splited line shapefile
    '''
    splited_line_path = generate_save_path(line_path, splited_line_path, "s")
    if os.path.exists(splited_line_path):
        delete_shapefile(splited_line_path)
    split_params = { 
        'INPUT' : line_path,
        'LINES' : overlap_line_path, 
        'OUTPUT' : splited_line_path
    }
    return run_processing_algorithm("native:splitwithlines", split_params)

def specific_vertices(input_feature_path, specific_vertices_path = ""):
    '''
    Use QGIS API to extract the specific vertices of the input feature layer
    input_feature_path: the path of the input feature shapefile
    if specific_vertices_path is not provided, the specific vertices shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the specific vertices shapefile
    '''
    specific_vertices_path = generate_save_path(input_feature_path, specific_vertices_path, "v")
    if os.path.exists(specific_vertices_path):
        delete_shapefile(specific_vertices_path)
    vertice_calc_params = {
        'INPUT' : input_feature_path,
        'VERTICES' : '0, -1',
        'OUTPUT' : specific_vertices_path
    }
    return run_processing_algorithm("native:extractspecificvertices", vertice_calc_params)

def calc_intersection(input_feature_path, compare_feature_path, grid_size = 0.01, intersection_path = ""):
    '''
    Use QGIS API to calculate the intersection of the input feature layer and the compare feature layer
    input_feature_path: the path of the input feature shapefile
    compare_feature_path: the path of the compare feature shapefile
    grid_size: the size of the grid
    if intersection_path is not provided, the intersection shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the intersection shapefile
    '''
    intersection_path = generate_save_path(input_feature_path, intersection_path, "intersection")
    if os.path.exists(intersection_path):
        delete_shapefile(intersection_path)
    intersection_params = {
        'INPUT' : input_feature_path,
        'OVERLAY' : compare_feature_path,
        'INPUT_FIELDS' : [],
        'OVERLAY_FIELDS' : [],
        'OVERLAY_FIELDS_PREFIX' : '',
        'OUTPUT' : intersection_path,
        'GRID_SIZE' : grid_size
    }
    return run_processing_algorithm("native:intersection", intersection_params)

def calc_line_intersection(input_feature_path, compare_feature_path, intersection_path = ""):
    '''
    Use QGIS API to calculate the intersection of the input feature layer and the compare feature layer
    input_feature_path: the path of the input feature shapefile
    compare_feature_path: the path of the compare feature shapefile
    if intersection_path is not provided, the intersection shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the intersection shapefile
    '''
    intersection_path = generate_save_path(input_feature_path, intersection_path, "int")
    if os.path.exists(intersection_path):
        delete_shapefile(intersection_path)
    intersection_params = {
        'INPUT' : input_feature_path,
        'INTERSECT' : compare_feature_path,
        'INPUT_FIELDS' : [],
        'INTERSECT_FIELDS' : [],
        'INTERSECT_FIELDS_PREFIX' : '',
        'OUTPUT' : intersection_path
    }
    return run_processing_algorithm("native:lineintersections", intersection_params)

def merge_layers(layer_list, EPSG_code, merged_path = ""):
    '''
    Use QGIS API to merge the input feature layer and the compare feature layer
    layer_list: the list of the feature shapefile
    EPSG_code: the EPSG code of the merged shapefile
    merged_path: the path of the merged shapefile
    '''
    merged_path = generate_save_path(layer_list[0], merged_path, "m")
    if os.path.exists(merged_path):
        delete_shapefile(merged_path)
    merge_params = {
        'LAYERS' : layer_list,
        'CRS' : QgsCoordinateReferenceSystem(f'EPSG:{EPSG_code}'),
        'OUTPUT' : merged_path
    }
    return run_processing_algorithm("native:mergevectorlayers", merge_params)

def raster_to_vector(raster_path, vector_path = ""):
    '''
    Use QGIS API to convert the raster layer to the vector layer
    raster_path: the path of the raster shapefile
    vector_path: the path of the vector shapefile
    if vector_path is not provided, the vector shapefile will be saved in the same directory as the raster shapefile
    output: the path of the vector shapefile
    '''
    vector_path = generate_save_path(raster_path, vector_path, "r2v")
    if os.path.exists(vector_path):
        delete_shapefile(vector_path)
    convert_params = {
        'INPUT':raster_path,
        'BAND':1,
        'FIELD':'DN',
        'EIGHT_CONNECTEDNESS':False,
        'EXTRA':'',
        'OUTPUT':vector_path
    }
    return run_processing_algorithm("gdal:polygonize", convert_params)

def polygon_to_line(input_feature_path, output_feature_path = ""):
    '''
    Use QGIS API to convert the polygon layer to the line layer
    input_feature_path: the path of the input feature shapefile
    output_feature_path: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(input_feature_path, output_feature_path, "p2l")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    params = {
        'INPUT':input_feature_path,
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:polygonstolines", params)

def create_buffer(input_feature_path, buffer_distance, buffer_path = ""):
    '''
    Use QGIS API to create the buffer of the input feature layer
    input_feature_path: the path of the input feature shapefile
    buffer_distance: the distance of the buffer
    if buffer_path is not provided, the buffer shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the buffer shapefile
    '''
    buffer_path = generate_save_path(input_feature_path, buffer_path, "b")
    if os.path.exists(buffer_path):
        delete_shapefile(buffer_path)
    buffer_params = {
        'INPUT':input_feature_path,
        'DISTANCE':buffer_distance,
        'SEGMENTS':5,
        'END_CAP_STYLE':0,
        'JOIN_STYLE':0,
        'MITER_LIMIT':2,
        'DISSOLVE':True,
        'SEPARATE_DISJOINT':False,
        'OUTPUT':buffer_path
    }
    return run_processing_algorithm("native:buffer", buffer_params)

def multipart_to_singleparts(input_feature_path, output_feature_path = ""):
    '''
    Use QGIS API to convert the multipart feature layer to the singlepart feature layer
    input_feature_path: the path of the input feature shapefile
    output_feature_path: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(input_feature_path, output_feature_path, "m2s")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    params = {
        'INPUT':input_feature_path,
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:multiparttosingleparts", params)

def exclude_by_mask(input_feature_path, mask_path, output_feature_path = ""):
    '''
    Use QGIS API to exclude the input feature layer by the mask layer
    input_feature_path: the path of the input feature shapefile
    mask_path: the path of the mask shapefile
    '''
    output_feature_path = generate_save_path(input_feature_path, output_feature_path, "mask")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    
    create_spatial_index(input_feature_path)
    create_spatial_index(mask_path)

    line_layer = QgsVectorLayer(input_feature_path, "line_layer", "ogr")
    polygon_layer = QgsVectorLayer(mask_path, "polygon_layer", "ogr")
    polygon_index = QgsSpatialIndex(polygon_layer.getFeatures())

    result_layer = QgsVectorLayer("LineString?crs=" + line_layer.crs().authid(), "external_line_features", "memory")
    result_layer.dataProvider().addAttributes(line_layer.fields().toList())
    result_layer.updateFields()

    with edit(result_layer):
        # parallel the search
        def process_line_feature(line_feature, polygon_layer, polygon_index):
            line_geom = line_feature.geometry()
            intersecting_poly_ids = polygon_index.intersects(line_geom.boundingBox())
            outside = True
            for poly_id in intersecting_poly_ids:
                poly_feature = polygon_layer.getFeature(poly_id)
                if line_geom.intersects(poly_feature.geometry()):
                    outside = False
                    break
            #print(f"{line_feature.id()}: calc {len(intersecting_poly_ids)} that {outside}")
            return (outside, line_feature) if outside else None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            process_func = partial(process_line_feature, polygon_layer=polygon_layer, polygon_index=polygon_index)
            results = executor.map(process_func, line_layer.getFeatures())
            
            # Add features that are outside
            for result in results:
                if result:
                    outside, line_feature = result
                    result_layer.dataProvider().addFeature(line_feature)

    # export the result layer to the output feature path
    transform_context = QgsProject.instance().transformContext()
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "utf-8"
    error = QgsVectorFileWriter.writeAsVectorFormatV3(result_layer, output_feature_path, transform_context, options)
    if error[0] != 0:
        logger.error(f"error writing output layer: {error}")
        return ""
    return output_feature_path

def simplify_shapefile(input_feature_path, tolerance, simplified_path = ""):
    '''
    Use QGIS API to simplify the input feature layer
    input_feature_path: the path of the input feature shapefile
    tolerance: the tolerance of the simplification
    simplified_path: the path of the simplified shapefile
    if simplified_path is not provided, the simplified shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the simplified shapefile
    '''
    simplified_path = generate_save_path(input_feature_path, simplified_path, "sp")
    if os.path.exists(simplified_path):
        delete_shapefile(simplified_path)
    simplify_params = {
        'INPUT':input_feature_path,
        'METHOD':0,
        'TOLERANCE':tolerance,
        'OUTPUT':simplified_path
    }
    return run_processing_algorithm("native:simplifygeometries", simplify_params)

def smooth_shapefile(input_feature_path, offset, smoothed_path = ""):
    '''
    Use QGIS API to smooth the input feature layer
    input_feature_path: the path of the input feature shapefile
    offset: the offset of the smoothing
    smoothed_path: the path of the smoothed shapefile
    if smoothed_path is not provided, the smoothed shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the smoothed shapefile
    '''
    smoothed_path = generate_save_path(input_feature_path, smoothed_path, "sm")
    if os.path.exists(smoothed_path):
        delete_shapefile(smoothed_path)
    smoothed_params = {
        'INPUT':input_feature_path,
        'ITERATIONS':3,
        'OFFSET':offset,
        'MAX_ANGLE':180,
        'OUTPUT':smoothed_path
    }
    return run_processing_algorithm("native:smoothgeometry", smoothed_params)

def all_vertices(input_feature_path, vertices_path = ""):
    '''
    Use QGIS API to extract all vertices of the input feature layer
    input_feature_path: the path of the input feature shapefile
    vertices_path: the path of the vertices shapefile
    if vertices_path is not provided, the vertices shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the vertices shapefile
    '''
    vertices_path = generate_save_path(input_feature_path, vertices_path, "v")
    if os.path.exists(vertices_path):
        delete_shapefile(vertices_path)
    params = {
        'INPUT':input_feature_path,
        'OUTPUT':vertices_path
    }
    return run_processing_algorithm("native:extractvertices", params)

def shortest_line(source_path, target_path, max_neighbor, max_distance, shortest_line_path = ""):
    '''
    Use QGIS API to calculate the shortest line between the source feature layer and the target feature layer
    source_path: the path of the source feature shapefile
    target_path: the path of the target feature shapefile
    max_neighbor: the maximum number of neighbors
    max_distance: the maximum distance of the shortest line
    shortest_line_path: the path of the shortest line shapefile
    if shortest_line_path is not provided, the shortest line shapefile will be saved in the same directory as the source feature shapefile
    output: the path of the shortest line shapefile
    '''
    shortest_line_path = generate_save_path(source_path, shortest_line_path, "sl")
    if os.path.exists(shortest_line_path):
        delete_shapefile(shortest_line_path)
    QgsProject.instance().setDistanceUnits(QgsUnitTypes.DistanceMeters)
    shortest_line_params = {
        'DISTANCE':max_distance,
        'DESTINATION':target_path,
        'METHOD':0,
        'NEIGHBORS':max_neighbor,
        'OUTPUT':shortest_line_path,
        'SOURCE':source_path,
    }
    return run_processing_algorithm("native:shortestline", shortest_line_params)

def fix_geometry(input_feature_path, output_feature_path = ""):
    '''
    Use QGIS API to fix the geometry of the input feature layer
    input_feature_path: the path of the input feature shapefile
    output_feature_path: the path of the output feature shapefile
    if output_feature_path is not provided, the output feature shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(input_feature_path, output_feature_path, "f")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    fix_geometry_params = {
        'INPUT':input_feature_path,
        'METHOD':0,
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:fixgeometries", fix_geometry_params)

def calc_area(feature_path):
    feature_layer = QgsVectorLayer(feature_path, "feature_layer", "ogr")
    if not feature_layer.isValid():
        logger.error(f"feature_layer is not valid")
        return ""
    # promise "area" field exists
    if not "area" in [f.name() for f in feature_layer.fields()]:
        area_field = QgsField(
            name="area",
            type=QMetaType.Type.Double,
            len=20,
            prec=10,
        )
        feature_layer.dataProvider().addAttributes([area_field])
        feature_layer.updateFields()
    area_field_index = feature_layer.fields().indexOf("area")
    area_map = {
        f.id(): {area_field_index: f.geometry().area()}
        for f in feature_layer.getFeatures()
    }
    feature_layer.dataProvider().changeAttributeValues(area_map)
    return feature_path

def explode_line(input_feature_path, exploded_path = ""):
    '''
    Use QGIS API to explode the input feature layer
    input_feature_path: the path of the input feature shapefile
    exploded_path: the path of the exploded shapefile
    if exploded_path is not provided, the exploded shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the exploded shapefile
    '''
    exploded_path = generate_save_path(input_feature_path, exploded_path, "e")
    if os.path.exists(exploded_path):
        delete_shapefile(exploded_path)
    params = {
        'INPUT':input_feature_path,
        'OUTPUT':exploded_path
    }
    return run_processing_algorithm("native:explodelines", params)

def direct_polygonize(raster_path, vector_path = ""):
    '''
    Use QGIS API to polygonize the input raster layer
    raster_path: the path of the input raster shapefile
    vector_path: the path of the output vector shapefile
    '''
    vector_path = generate_save_path(raster_path, vector_path, "dp")
    if os.path.exists(vector_path):
        delete_shapefile(vector_path)
    params = {
        'INPUT':raster_path,
        'KEEP_FIELDS':False,
        'OUTPUT':vector_path
    }
    return run_processing_algorithm("native:polygonize", params)

def sort_features(feature_path, field_name, ascending = True, output_feature_path = ""):
    '''
    Use QGIS API to sort the input feature layer by the field
    feature_path: the path of the input feature shapefile
    field_name: the name of the field to sort
    ascending: the order of the sorting
    '''
    output_feature_path = generate_save_path(feature_path, output_feature_path, "sorted")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    expression = f'"{field_name}"'
    params = {
        'INPUT':feature_path,
        'EXPRESSION':expression,
        'ASCENDING':ascending,
        'NULLS_FIRST':True,
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:orderbyexpression", params)

def aggregate_features(feature_path, field_name, output_feature_path = ""):
    '''
    Use QGIS API to aggregate the input feature layer by the field
    feature_path: the path of the input feature shapefile
    field_name: the name of the field to aggregate
    output_feature_path: the path of the output feature shapefile
    if output_feature_path is not provided, the output feature shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(feature_path, output_feature_path, "a")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    params = {
        'INPUT':feature_path,
        'GROUP_BY':f'{field_name}',
        'AGGREGATES':[
            {'aggregate': 'first_value','delimiter': ',','input': '"FID"','length': 11,'name': 'FID','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},
            {'aggregate': 'sum','delimiter': ',','input': '"area"','length': 20,'name': 'area','precision': 10,'sub_type': 0,'type': 6,'type_name': 'double precision'}],
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:aggregate", params)

def sample_in_polygon(input_feature_path, point_inside_number, min_distance_inside, output_feature_path = "" , seed = None):
    '''
    Use QGIS API to sample the input feature layer in the polygon
    input_feature_path: the path of the input feature shapefile
    point_inside_number: the number of the points inside the polygon
    output_feature_path: the path of the output feature shapefile
    if output_feature_path is not provided, the output feature shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(input_feature_path, output_feature_path, "sample")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    params = {
        'INPUT':input_feature_path,
        'POINTS_NUMBER':point_inside_number,
        'MIN_DISTANCE':min_distance_inside,
        'MIN_DISTANCE_GLOBAL':0,
        'MAX_TRIES_PER_POINT':3,
        'SEED':seed,
        'INCLUDE_POLYGON_ATTRIBUTES':True,
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:randompointsinpolygons", params)

def sample_raster_values(raster_path, sample_feature_path, output_feature_path = ""):
    '''
    Use QGIS API to sample the values of the input raster layer
    raster_path: the path of the input raster shapefile
    sample_feature_path: the path of the sample feature shapefile
    output_feature_path: the path of the output feature shapefile
    if output_feature_path is not provided, the output feature shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(sample_feature_path, output_feature_path, "sample")
    if os.path.exists(output_feature_path):
        delete_shapefile(output_feature_path)
    params = {
        'INPUT':sample_feature_path,
        'RASTERCOPY':raster_path,
        'COLUMN_PREFIX':'SAMPLE_',
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:rastersampling", params)

def clip_raster(raster_path, polygon_path, output_raster_path = ""):
    '''
    Use QGIS API to clip the input raster layer by the input polygon layer
    raster_path: the path of the input raster shapefile
    polygon_path: the path of the input polygon shapefile
    '''
    output_raster_path = generate_save_path(raster_path, output_raster_path, "c", "raster")
    params = {
        'INPUT':raster_path,
        'MASK':polygon_path,
        'OUTPUT':output_raster_path
    }
    return run_processing_algorithm("gdal:cliprasterbymasklayer", params)

def clip_vector(vector_path, polygon_path, output_vector_path = ""):
    '''
    Use QGIS API to clip the input vector layer by the input polygon layer
    vector_path: the path of the input vector shapefile
    polygon_path: the path of the input polygon shapefile
    '''
    output_vector_path = generate_save_path(vector_path, output_vector_path, "c")
    if os.path.exists(output_vector_path):
        delete_shapefile(output_vector_path)
    params = {
        'INPUT':vector_path,
        'OVERLAY':polygon_path,
        'OUTPUT':output_vector_path
    }
    return run_processing_algorithm("native:clip", params)

def create_grid_point(reference_layer, spacing, output_feature_path = ""):
    '''
    Use QGIS API to create a grid of points from the input polygon layer
    reference_layer: the path of the input polygon shapefile
    spacing: the spacing of the grid
    output_feature_path: the path of the output feature shapefile
    if output_feature_path is not provided, the output feature shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the output feature shapefile
    '''
    output_feature_path = generate_save_path(reference_layer, output_feature_path, "grid")
    extent = reference_layer.extent().toString()
    params = {
        'TYPE':0,
        'EXTENT':f'{extent} [EPSG:4326]',
        'HSPACING':spacing,
        'VSPACING':spacing,
        'HOVERLAY':0,
        'VOVERLAY':0,
        'CRS':QgsCoordinateReferenceSystem('EPSG:32650'),
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:creategrid", params)

def calc_purity(polygon_layer, sample_layer, output_feature_path = ""):
    '''
    Use QGIS API to calculate the purity of the input polygon layer
    polygon_layer: the path of the input polygon shapefile
    sample_layer: the path of the sample feature shapefile
    '''
    output_feature_path = generate_save_path(polygon_layer, output_feature_path, "purity")
    aggregate_params = {
        'INPUT':sample_layer,
        'GROUP_BY':'"FID"',
        'AGGREGATES':[
            {'aggregate': 'stdev','delimiter': ',','input': '"SAMPLE_1"','length': 10,'name': 'stdev','precision': 5,'sub_type': 0,'type': 6,'type_name': 'double precision'},
            {'aggregate': 'first_value','delimiter': ',','input': '"FID"','length': 11,'name': 'FID','precision': 0,'sub_type': 0,'type': 4,'type_name': 'int8'},
            {'aggregate': 'mean','delimiter': ',','input': '"SAMPLE_1"','length': 10,'name': 'mean','precision': 5,'sub_type': 0,'type': 6,'type_name': 'double precision'}],
        'OUTPUT':output_feature_path
    }
    result = processing.run("native:aggregate", aggregate_params)
    if 'error' in result:
        logger.error(f"error in aggregate: {result['error']}")
        return ""
    return join_by_attribute(polygon_layer, result['OUTPUT'], 'FID')

def filter_by_line_length(input_feature_path, distance):
    '''
    Use QGIS API to calculate the length of the input feature layer
    input_feature_path: the path of the input feature shapefile
    output_feature_path: the path of the output feature shapefile
    '''
    layer = QgsVectorLayer(input_feature_path, "input_feature_layer", "ogr")
    if not layer.isValid():
        logger.error(f"input_feature_layer is not valid")
        return ""
    # calculate the length of the input feature layer
    length_field = QgsField(
        name="length",
        type=QMetaType.Type.Double,
        len=10,
        prec=3,
    )
    layer.dataProvider().addAttributes([length_field])
    layer.updateFields()
    length_field_index = layer.fields().indexOf("length")
    length_map = {
        f.id(): {length_field_index: f.geometry().length()}
        for f in layer.getFeatures()
    }
    layer.dataProvider().changeAttributeValues(length_map)
    
    # filter the features by the length
    output_feature_path = generate_save_path(input_feature_path, "", "l")
    params = {
        'INPUT':input_feature_path,
        'EXPRESSION':f'length < {distance} and length > 0',
        'OUTPUT':output_feature_path
    }
    return run_processing_algorithm("native:extractbyexpression", params)