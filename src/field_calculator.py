import re
import geopandas as gpd
from abc import ABC, abstractmethod
from simpleeval import simple_eval
from pandas.api.types import is_numeric_dtype
import os
import zipfile
import requests
from dotenv import load_dotenv

load_dotenv()
class CalcStrategy(ABC):
    """
    Abstract base class for calculation strategies.
    Every engine must implement the execute interface to process expressions.
    """
    @abstractmethod
    def execute(self, expression: str,gdf: gpd.GeoDataFrame,target_field: str):
        """
        Executes the calculation logic.

        Args:
            expression (str): The formula or logic to evaluate.
            gdf (gpd.GeoDataFrame): The input spatial data.
            target_field (str): The column name where results will be stored.

        Returns:
            gpd.GeoDataFrame: The updated GeoDataFrame.
        """
        pass
class VectorStrategy(CalcStrategy):
    """
    JOB: High-speed numerical math using C-optimized pandas/numexpr.
    Best for arithmetic operations on numeric columns.
    """
    def execute(self, expression: str,gdf: gpd.GeoDataFrame,target_field: str):
        try:
            fields = re.findall(r'!(.*?)!', expression)
            for field in fields:
                if not is_numeric_dtype(gdf[field]):
                    raise ValueError(f"Field {field} is not numeric.")
            clean_expr = expression.replace('!', '').replace('[', '').replace(']', '')
            gdf[target_field]=gdf.eval(clean_expr,engine='numexpr')
            return gdf
        except Exception as e:
            raise ValueError(f"VectorStrategy failed to execute expression '{expression}': {e}")
class SpatialStrategy(CalcStrategy):
    """
    JOB: Handle geometry transformations (vectorized).
    Supports chained operations like !shape!.buffer(10).centroid
    """ 
    def execute(self, expression: str,gdf: gpd.GeoDataFrame,target_field: str):
        try:
            clean_expr = expression.replace('!shape!.', '').replace('!shape!', '')
            parts = clean_expr.split('.')
            result = gdf.geometry
            for part in parts:
                if not part: continue
                if '(' in part:
                    method_name = part.split('(')[0]
                    arg_str = part[part.find("(")+1:part.find(")")]
                    args = [float(x.strip()) for x in arg_str.split(',')] if arg_str else []
                    result = getattr(result, method_name)(*args) 
                else:
                    result = getattr(result, part)
            
            gdf[target_field] = result
            return gdf
        except Exception as e:
            raise ValueError(f"SpatialStrategy failed to execute expression '{expression}': {e}")
class LogicStrategy(CalcStrategy):
    """
    JOB: Handle complex Python logic and string manipulation using row-wise evaluation.
    Utilizes simple_eval for safe evaluation of expressions.
    """
    def __init__(self):
        self.functions={"UPPER":lambda x: str(x).upper(),
                               "LOWER":lambda x: str(x).lower(),
                               }
    def execute(self, expression: str,gdf: gpd.GeoDataFrame,target_field: str):
        fields = re.findall(r'!(.*?)!', expression)
        def row_eval(row):
            try:        
                context={f:row[f] for f in fields}
                local_exp=expression
                for f  in fields:
                    local_exp=local_exp.replace(f"!{f}!",f)
                return simple_eval(local_exp,names=context,functions=self.functions)
            except Exception:
                return None
        try:    
            gdf[target_field] = gdf.apply(row_eval, axis=1)
            return gdf
        except Exception as e:
            raise ValueError(f"LogicStrategy failed to execute expression '{expression}': {e}")
 
class ExpressionParser:
    """
    Utility class to route expressions to the appropriate calculation strategy.
    """
    @staticmethod
    def get_strategy(expression:str):
        """
        Analyzes the expression string to determine the best execution engine.
        """
        try:
            expression=expression.lower()
            spatial_expression=['shape','.buffer','.centroid','.area','.length']
            if any (key in expression for key in spatial_expression):
                print("RoutingEngine: Using Spatial Strategy")
                return SpatialStrategy()
            logic_expression=['upper','lower','and','or', "if", "else", "not", "is None", "is not None"]
            if any (key in expression for key in logic_expression):
                   print("RoutingEngine: Using Logic Strategy")
                   return LogicStrategy()
            print("RoutingEngine: Using Vector Strategy.")
            return VectorStrategy()
        except Exception as e:
            raise ValueError(f"ExpressionParser failed to determine strategy for expression '{expression}': {e}")

class FieldCalculator:
    """
    Facade class that provides a high-level API for calculating field values.
    """
    def __init__(self,gdf,expression,target_field):
        """
        Initializes the calculator.

        Args:
            gdf (gpd.GeoDataFrame): Data to process.
            expression (str): Expression to evaluate (e.g., "!area! / 1000").
            target_field (str): Destination field name.
        """
        self.gdf=gdf
        self.expression=expression
        self.target_field=target_field
    def calculate(self):
        """
        Parses the expression and executes the strategy.
        
        Returns:
            gpd.GeoDataFrame: Data with the new/updated field.
            
        Raises:
            ValueError: If the input GeoDataFrame is invalid.
        """
        try:
            strategy=ExpressionParser.get_strategy(self.expression)
            gdf=strategy.execute(expression=self.expression,gdf=self.gdf,target_field=self.target_field)
            return gdf
        except Exception as e:
            raise ValueError(f"FieldCalculator failed to calculate field '{self.target_field}': {e}")

class GeoserverCalcField:
    def __init__(self):
        """Initializes the Geoserver client.

        The constructor loads credentials and the GeoServer URL from environment
        variables for authentication and connectivity.
        """
        self.username = os.getenv('GEOSERVER_USERNAME')
        self.password = os.getenv('GEOSERVER_PASSWORD')
        self.geoserver_url = os.getenv('GEOSERVER_URL')
    def get_vector_layer(self, workspace, layername):
        """Retrieves the WFS URL for a vector layer.

        :param workspace: The name of the GeoServer workspace containing the layer.
        :type workspace: str
        :param layername: The name of the vector layer.
        :type layername: str
        :returns: The WFS URL for the specified layer, formatted as JSON.
        :rtype: str
        """
        url = (f"{self.geoserver_url}/{workspace}/ows?service=WFS&version=1.0.0&request=GetFeature"
               f"&typeName={workspace}:{layername}&outputFormat=application/json")
        return url
    def create_zipfilepath(self, tmpdir) :
        """Creates the file path for a temporary zip file.

        :param tmpdir: The path of the temporary folder.
        :type tmpdir: str
        :returns: The path of the zip file to be created.
        :rtype: str
        """
        zip_file_path = os.path.join(tmpdir, "output.zip")
        return zip_file_path
    def zip_files(self, zip_file_path, tempdir):
        """Zips shapefiles from a temporary directory into a single zip file.
      
        The process includes standard shapefile components: .shp, .shx, .dbf, .prj, and .cpg.

        :param zip_file_path: The full path for the output zip file.
        :type zip_file_path: str
        :param tempdir: The path of the temporary directory containing the shapefiles.
        :type tempdir: str
        :raises ValueError: If an error occurs during the zipping process.
        """
        try:
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(tempdir):
                    for file in files:
                        if file.split('.')[-1] in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                            file_path = os.path.join(root, file)
                            zf.write(file_path, os.path.basename(file_path))
        except Exception as e:
            raise TimeoutError(f'cannot zip the files: {e}') from e
    def upload_shapefile(self, workspace, layername, zip_path):
        """Uploads a zipped shapefile to GeoServer.
       
        :param workspace: The name of the GeoServer workspace.
        :type workspace: str
        :param datastore: The name of the datastore that will contain the shapefile.
        :type datastore: str
        :param zip_path: The path of the zip file to upload.
        :type zip_path: str
        :raises ValueError: If an error occurs during the upload process.
        :returns: The HTTP response from the GeoServer REST API.
        :rtype: requests.Response
        """
        try:
            url = (f"{self.geoserver_url}/rest/workspaces/"
                   f"{workspace}/datastores/{layername}/file.shp")
            with open(zip_path, "rb") as f:
                headers = {"Content-type": "application/zip"}
                response = requests.put(
                    url,
                    data=f,
                    auth=(self.username, self.password),
                    headers=headers,
                    params={"update": "overwrite"},
                    timeout=30.0
                )
            return response.status_code
        except Exception as e:
            raise ValueError(f'cannot upload the shapefile: {e},{response.content}') from e
         