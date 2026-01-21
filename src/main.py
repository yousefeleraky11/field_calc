from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import uvicorn
import geopandas as gpd
from  field_calculator import FieldCalculator,GeoserverCalcField
app = FastAPI()
class CalcFieldsInputs(BaseModel):
    """Input model for the calculate_fields endpoint."""
    workspace: str
    layername: str
    store_name: str
    expression: str
    target_field: str
@app.post("/field_calculator")
async def calculate_fields(inputs:CalcFieldsInputs):
    """Calculates and updates fields in a GeoServer layer based on an expression.
    Args:
        inputs (CalcFieldsInputs): Input parameters including workspace, layername,
                                    store_name, expression, and target_field.
    Returns:
        dict: Metadata about the updated layer including name, type, geometry type,
              bounding box, and area.
    """
    try:    
        
        geoserver =GeoserverCalcField()
        layer_url=geoserver.get_vector_layer(workspace=inputs.workspace,
                                             layername=inputs.layername)
        gdf =gpd.read_file(layer_url)
        calculator=FieldCalculator(gdf=gdf,expression=inputs.expression,
                                   target_field=inputs.target_field)
        calculator.calculate(layer_name=inputs.layername)
        geoserver.update_datastore(workspace=inputs.workspace,
                                                      datastore=inputs.store_name,
                                                      layername=inputs.layername)
        new_layer={"layer_name": inputs.layername,
            "layer_type": "vector",
            "geometry_type": gdf.geom_type.unique().tolist(),
            "bbox": gdf.total_bounds.tolist(),
            "area": gdf.geometry.area.sum(),
            }
        return new_layer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9091)
    