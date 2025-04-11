from osgeo import gdal, ogr, osr
from dotenv import load_dotenv
import os

load_dotenv()
gdal.UseExceptions()
gdal.AllRegister()
ogr.UseExceptions()
ogr.RegisterAll()

lcz_dir = os.getenv("LCZ_DIR")

split_level_mapper = {
    "natural": [0,10],
    "water": [16.1]
}

def create_mask(location_dir, file, split_type):
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
            LCZdata,
            0,
            0,
            levels,
            0,
            0,
            boundaries_layer,
            0,
            1
        )
    except Exception as e:
        print(f"error creating natural boundaries layer: {e}")
    
    print("success")

def __main__():
    for location in os.listdir(lcz_dir):
        location_dir = os.path.join(lcz_dir, location)
        for file in os.listdir(location_dir):
            if file.endswith(".tif"):
                print(file)
                create_mask(location_dir, file, "natural")
                create_mask(location_dir, file, "water")
                break

if __name__ == "__main__":
    __main__()