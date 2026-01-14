from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import uvicorn
import uuid
import os
import geopandas as gpd
import tempfile
from  field_calculator import FieldCalculator,GeoserverCalcField
app = FastAPI()
class CalcFieldsInputs(BaseModel):
    workspace: str
    layername: str
    expression: str
    target_field: str
@app.post("/field_calculator")
def calculate_fields(inputs:CalcFieldsInputs):
    try:    
        with tempfile.TemporaryDirectory() as tmp:
            geoserver =GeoserverCalcField()
            layer_url=geoserver.get_vector_layer(workspace=inputs.workspace,
                                                 layername=inputs.layername)
            gdf =gpd.read_file(layer_url)
            calculator=FieldCalculator(gdf=gdf,expression=inputs.expression,
                                       target_field=inputs.target_field)
            result_gdf=calculator.calculate()
            layer_name=f'{inputs.layername}_calc_{str(uuid.uuid4())[:4]}'
            result_gdf.to_file(f'{tmp}/{layer_name}.shp')
            zip_file_path=geoserver.create_zipfilepath(tmpdir=tmp)
            geoserver.zip_files(zip_file_path=zip_file_path,tempdir=tmp)
            response_status=geoserver.upload_shapefile(workspace=inputs.workspace,
                                                       layername=layer_name,zip_path=zip_file_path)
            return {"message":"Field calculation successful",
                    "new_layer_name":layer_name,
                    "response_status":response_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9091)
    