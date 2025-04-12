from qgis.core import *
from qgis.PyQt.QtCore import QMetaType
import processing
from processing.core.Processing import Processing
Processing.initialize()
import os
import logging

def reindex_feature(feature_path, field_name):
    '''
    Use QGIS API to reindex the feature layer
    feature_path: the path of the feature shapefile
    field_name: the name of the field to reindex
    '''
    # create a new field
    field = QgsField(field_name, QMetaType.Type.Int)
    # add the field to the feature layer
    feature_layer = QgsVectorLayer(feature_path, 'Feature Layer', 'ogr')
    feature_layer.dataProvider().addAttributes([field])
    feature_layer.updateFields()
    monoid_field_index = feature_layer.fields().indexOf("monoid")
    monoid_map = {
        f.id(): {monoid_field_index: f.id()}
        for f in feature_layer.getFeatures()
    }
    feature_layer.dataProvider().changeAttributeValues(monoid_map)
    #reindex_params = {
    #    'FIELD' : field_name,
    #    'INPUT' : feature_path
    #}
    #with edit(feature_layer):
    #    result = processing.run("native:createattributeindex", reindex_params)
    #
    #print(result)
    #if 'error' in result:
    #    print(f"error reindexing the feature layer: {result['error']}")
    #    return ""
    return feature_path

def reproject_shapefile(geo_shp_path,proj_shp_path = ""):
    '''
    convert WGS84 to WGS84 UTM50N
    geo_shp_path: the path of the original shapefile
    proj_shp_path: the path of the projected shapefile
    if proj_shp_path is not provided, the projected shapefile will be saved in the same directory as the original shapefile
    output: the path of the projected shapefile
    '''
    if proj_shp_path == "":
        geo_dir = os.path.dirname(geo_shp_path) 
        geo_name = os.path.basename(geo_shp_path).split(".")[0]
        proj_shp_path = os.path.join(geo_dir, f'{geo_name}_proj.shp')
    if os.path.exists(proj_shp_path):
        os.remove(proj_shp_path)
    crs_params = {
        'CONVERT_CURVED_GEOMETRIES' : False,
        'INPUT' : geo_shp_path,
        'OPERATION' : '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone=50 +ellps=WGS84',
        'OUTPUT' : proj_shp_path,
        'TARGET_CRS' : QgsCoordinateReferenceSystem('EPSG:32650')
    }
    result = processing.run("native:reprojectlayer", crs_params)
    if 'error' in result:
        print(f"error reprojecting the shapefile: {result['error']}")
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

    if filter_path == "":
        line_dir = os.path.dirname(line_path)
        line_name = os.path.basename(line_path).split('.')[0]
        filter_path = os.path.join(line_dir, f'{line_name}_filter.shp')
    if os.path.exists(filter_path):
        os.remove(filter_path)
    # iterate through the polygon layer and add the area attribute
    poly_layer = QgsVectorLayer(proj_poly_path, 'Multipolygon Layer', 'ogr')
    if not poly_layer.isValid():
        print("Layer failed to load!")
    line_layer = QgsVectorLayer(line_path, 'Line Layer', 'ogr')
    if not line_layer.isValid():
        print("Line layer failed to load!")
    
    line_layer.dataProvider().addAttributes([QgsField("remain", QMetaType.Type.Int)])
    line_layer.updateFields()
    remain_field = line_layer.fields().indexOf("remain")
    with edit(line_layer):
        for feature, line_feature in zip(poly_layer.getFeatures(), line_layer.getFeatures()):
            area = feature.geometry().area()
            feature_id = feature.id()
            line_layer.changeAttributeValue(line_feature.id(), remain_field, determine_remain(area))
            #line_feature['remain'] = determine_remain(area)
            #line_layer.updateFeature(line_feature)

    filter_params = {
        'FIELD' : 'remain',
        'INPUT' : line_path,
        'OPERATOR' : 0,
        'OUTPUT' : filter_path,
        'VALUE' : '1'
    }
    result = processing.run("qgis:extractbyattribute", filter_params)
    if 'error' in result:
        print(f"error filtering line: {result['error']}")
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
    if poly_path == "":
        line_dir = os.path.dirname(line_path)
        line_name = os.path.basename(line_path).split('.')[0]
        poly_path = os.path.join(line_dir, f'{line_name}_poly.shp')
    if os.path.exists(poly_path):
        os.remove(poly_path)
    convert_params = {
        "INPUT": line_path,
        "OUTPUT": poly_path
    }
    try:
        poly = processing.run("qgis:linestopolygons", convert_params)
    except Exception as e:
        print(f"error converting line to polygon: {e}")
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
    if splited_road_path == "":
        feature_dir = os.path.dirname(feature_path)
        feature_name = os.path.basename(feature_path).split('.')[0]
        splited_road_path = os.path.join(feature_dir, f'{feature_name}_roadsplit.shp')
    if os.path.exists(splited_road_path):
        os.remove(splited_road_path)
    merge_params = {
        'INPUT' : base_road_filepath,
        'LINES' : feature_path,
        'OUTPUT' : splited_road_path
    }
    result = processing.run("native:splitwithlines", merge_params)
    if 'error' in result:
        print(f"error merging shapefiles: {result['error']}")
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
    if centroid_path == "":
        line_dir = os.path.dirname(line_path)
        line_name = os.path.basename(line_path).split('.')[0]
        centroid_path = os.path.join(line_dir, f'{line_name}_centroid.shp')
    if os.path.exists(centroid_path):
        os.remove(centroid_path)
    calc_params = {
        'ALL_PARTS' : True,
        'INPUT' : line_path,
        'OUTPUT' : centroid_path
    }
    result = processing.run("native:centroids", calc_params)
    if 'error' in result:
        print(f"error calculating the centroid of the line layer: {result['error']}")
        return ""
    return centroid_path

def create_spatial_index(feature_path):
    '''
    Use QGIS API to create a spatial index for the feature layer
    feature_path: the path of the feature shapefile
    output: the path of the spatial index shapefile
    '''
    processing.run("native:createspatialindex", {'INPUT' : feature_path})

def join_by_attribute(input_feature_path, add_feature_path, join_path = ""):
    '''
    Use QGIS API to join the input feature layer by the add feature layer
    input_feature_path: the path of the input feature shapefile
    add_feature_path: the path of the add feature shapefile
    if join_path is not provided, the joined shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the joined shapefile
    '''
    if join_path == "":
        feature_dir = os.path.dirname(input_feature_path)
        feature_name = os.path.basename(input_feature_path).split('.')[0]
        join_path = os.path.join(feature_dir, f'{feature_name}_join.shp')
    if os.path.exists(join_path):
        os.remove(join_path)
    join_params = {
        'DISCARD_NONMATCHING' : False,
        'FIELD' : 'monoid',
        'FIELDS_TO_COPY' : [],
        'FIELD_2' : 'monoid',
        'INPUT' : input_feature_path,
        'INPUT_2' : add_feature_path,
        'METHOD' : 1,
        'OUTPUT' : join_path,
        'PREFIX' : '' 
    }
    result = processing.run("native:joinattributestable", join_params)
    if 'error' in result:
        print(f"error joining the input feature layer by the add feature layer: {result['error']}")
        return ""
    return join_path


def extract_nonull_attribute(input_feature_path, field_name, extracted_path = ""):
    '''
    Use QGIS API to extract the non-null attribute of the input feature layer
    input_feature_path: the path of the input feature shapefile
    field_name: the name of the field to extract
    extracted_path: the path of the extracted shapefile
    if extracted_path is not provided, the extracted shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the extracted shapefile
    '''
    if extracted_path == "":
        feature_dir = os.path.dirname(input_feature_path)
        feature_name = os.path.basename(input_feature_path).split('.')[0]
        extracted_path = os.path.join(feature_dir, f'{feature_name}_nonull.shp')
    if os.path.exists(extracted_path):
        os.remove(extracted_path)
    extract_params = {
        'FIELD' : field_name,
        'INPUT' : input_feature_path,
        'OPERATOR' : 8,
        'OUTPUT' : extracted_path,
        'VALUE' : ''
    }
    result = processing.run("native:extractbyattribute", extract_params)
    if 'error' in result:
        print(f"error extracting the non-null attribute of the input feature layer: {result['error']}")
        return ""
    return extracted_path

def extract_whithindistance(input_feature_path, compare_feature_path, distance_extracted_path = ""):
    '''
    Use QGIS API to extract the input feature layer within the distance of the compare feature layer
    input_feature_path: the path of the input feature shapefile
    compare_feature_path: the path of the compare feature shapefile
    distance_extracted_path: the path of the distance extracted shapefile
    if distance_extracted_path is not provided, the distance extracted shapefile will be saved in the same directory as the input feature shapefile
    output: the path of the distance extracted shapefile
    '''
    if distance_extracted_path == "":
        input_dir = os.path.dirname(input_feature_path)
        input_name = os.path.basename(input_feature_path).split('.')[0]
        distance_extracted_path = os.path.join(input_dir, f'{input_name}_distance.shp')
    if os.path.exists(distance_extracted_path):
        os.remove(distance_extracted_path)
    extract_params = {
        'DISTANCE' : 20,
        'INPUT' : input_feature_path,
        'REFERENCE' : compare_feature_path,
        'OUTPUT' : distance_extracted_path
    }
    create_spatial_index(input_feature_path)
    create_spatial_index(compare_feature_path)
    result = processing.run("native:extractwithindistance", extract_params)
    if 'error' in result:
        print(f"error extracting the input feature layer within the distance of the compare feature layer: {result['error']}")
        return ""
    return distance_extracted_path
