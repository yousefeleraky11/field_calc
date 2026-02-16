from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import uvicorn
import geopandas as gpd
# from  field_calculator import FieldCalculator,GeoserverCalcField
from field_calculator_test import ExpressionValidator,GeoserverCalcField
app = FastAPI()
class CalcFieldsInputs(BaseModel):
    """Input model for field calculation."""
    workspace: str
    layername: str
    payload: dict
    target_field: str
# @app.post("/field_calculator")
# async def calculate_fields(inputs:CalcFieldsInputs):
#     try:
#         geoserver =GeoserverCalcField()
#         layer_url=geoserver.get_vector_layer(workspace=inputs.workspace,
#                                              layername=inputs.layername)
#         gdf =gpd.read_file(layer_url)
#         calculator=FieldCalculator(gdf=gdf,expression=inputs.expression,
#                                    target_field=inputs.target_field)
#         calculator.calculate(layer_name=inputs.layername)
#         geoserver.update_datastore(workspace=inputs.workspace,
#                                                       layername=inputs.layername)
#         new_layer={"layer_name": inputs.layername,
#             "layer_type": "vector",
#             "geometry_type": gdf.geom_type.unique().tolist(),
#             "bbox": gdf.total_bounds.tolist(),
#             "area": gdf.geometry.area.sum(),
#             }
#         return new_layer
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post('/expression-validator')
def validate_expression(inputs:CalcFieldsInputs):
    """Endpoint to validate the expression for field calculation."""
    try:
        geoserver=GeoserverCalcField()
        geoserver_url=geoserver.get_vector_layer(workspace=inputs.workspace,
                                                 layername=inputs.layername)
        validator=ExpressionValidator(url=geoserver_url,payload=inputs.payload)
        return validator.validate()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9091)
    