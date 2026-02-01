import re
import geopandas as gpd
from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

class CalcStrategy(ABC):
    """ every engine should implement this interface """
    @abstractmethod
    def execute(self, payload: dict,gdf: gpd.GeoDataFrame,target_field: str):
        """Executes the calculation strategy."""
        pass        

class SpatialStrategy(CalcStrategy):
    """
    JOB: Handle geometry transformations dynamically via JSON.
    Supports: Properties, Methods, Static Args, and Dynamic Column Args.
    """
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
    
        result = gdf.geometry
        for operation in payload.get('operations', []):
            method_name = operation['method']
            raw_args = operation.get('args', [])
            kwargs = operation.get('kwargs', {}) 
            resolved_args = []
            for arg in raw_args:
                if isinstance(arg, str) and arg.startswith("!") and arg.endswith("!"):
                    col_name = arg.replace("!", "")
                    if col_name not in gdf.columns:
                        raise ValueError(f"Column '{col_name}' not found.")
                    resolved_args.append(gdf[col_name])
                else:
                    resolved_args.append(arg)
            if not hasattr(result, method_name):
                raise ValueError(f"Geometry object has no method/property '{method_name}'")
            attr = getattr(result, method_name)
            if callable(attr):
                result = attr(*resolved_args, **kwargs)
            else:
                result = attr
        gdf[target_field] = result
        if isinstance(result, (gpd.GeoSeries, gpd.GeoDataFrame)):
             gdf.set_geometry(target_field, inplace=True)

        return gdf
class LogicStrategy(CalcStrategy):
    """
   job: Handle logic transformations dynamically via JSON.
    """
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        rules = payload.get("rules", [])
        default_val = payload.get("else", None)

        if not rules:
            raise ValueError("Logic strategy requires a 'rules' list.")
        conditions = []
        choices = []
        try:
            for rule in rules:
                condition_str = rule.get("if")
                result_val = rule.get("then")
                mask = gdf.eval(condition_str)
                conditions.append(mask)
                choices.append(result_val)
            gdf[target_field] = np.select(conditions, choices, default=default_val)
            return gdf

        except Exception as e:
            raise ValueError(f"Rule Engine failed: {e}")
class VectorStrategy(CalcStrategy):
    """JOB: High-speed numerical math using C-optimized pandas/numexpr."""
    
    def execute(self,payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        expression = payload.get("expression", "")
        try:
            fields = re.findall(r'!(.*?)!', expression)
            for field in fields:
                if field not in gdf.columns:
                    raise ValueError(f"Field {field} not found in GeoDataFrame.")
                if not is_numeric_dtype(gdf[field]):
                    raise ValueError(f"Field {field} is not numeric.")
            clean_expr = expression.replace('!', '').replace('[', '').replace(']', '')
            gdf[target_field]=gdf.eval(clean_expr,engine='numexpr')
            return gdf
        except ZeroDivisionError:
            raise ValueError("Division by zero detected in expression.")
        except Exception as e:
            raise ValueError(f"VectorStrategy failed to execute expression '{expression}': {e}")  
class ExpressionParser:
    """
    Utility class to route expressions to the appropriate calculation strategy.
    """
    @staticmethod
    def get_strategy(payload: dict):
        """
        Analyzes the payload to determine the best execution engine.
        """
        try:
            strategy_type = payload.get("strategy", "").lower()
            if strategy_type == "spatial":
                return SpatialStrategy()
            elif strategy_type == "logic":
                return LogicStrategy()
            elif strategy_type == "vector":
                return VectorStrategy()
        except Exception as e:
            raise ValueError(f"ExpressionParser failed to determine strategy for expression '{strategy_type}': {e}")

class FieldCalculator:
    """
    Facade class that provides a high-level API for calculating field values.
    """
    def __init__(self,payload,target_field,url):
        """
        Initializes the calculator.

        Args:
            gdf (gpd.GeoDataFrame): Data to process.
            expression (str): Expression to evaluate (e.g., "!area! / 1000").
            target_field (str): Destination field name.
        """
        self.payload=payload
        self.target_field=target_field
        self.url=url
        self.db_port=os.getenv('DB_PORT')
        self.db_host=os.getenv('DB_HOST')
        self.db_name=os.getenv('DB_NAME')
        self.db_user=os.getenv('DB_USER')
        self.db_password=os.getenv('DB_PASSWORD')
        
    def calculate(self,layer_name):
        """
        Parses the expression and executes the strategy.
        
        Returns:
            gpd.GeoDataFrame: Data with the new/updated field.
            
        Raises:
            ValueError: If the input GeoDataFrame is invalid.
        """
        try:
            gdf=gpd.read_file(self.url)
            strategy=ExpressionParser.get_strategy(self.payload)
            gdf=strategy.execute(payload=self.payload,gdf=gdf,target_field=self.target_field)
            engine=create_engine(f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}")
            gdf.to_postgis(layer_name,engine,if_exists='replace')
         
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
        self.db_port=os.getenv('DB_PORT')
        self.db_host=os.getenv('DB_HOST')
        self.db_name=os.getenv('DB_NAME')
        self.db_user=os.getenv('DB_USER')
        self.db_password=os.getenv('DB_PASSWORD')
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
    def get_store_name(self,workspace, layer_name):
        info_url = f"{self.geoserver_url}/rest/workspaces/{workspace}/featuretypes/{layer_name}.json"
        auth = (self.username, self.password)
        try:
            response = requests.get(info_url, auth=auth)
            if response.status_code == 200:
                data = response.json()
                store_name = data['featureType']['store']['name']
                print(f'store_name: {store_name}')
                store_name=store_name.split(':')[1]
            return store_name   
        except Exception as e:
            print(f"An error occurred: {e}")
    def update_datastore(self,workspace,layername):
        """Updates the datastore in GeoServer to reflect changes in the database.

        :param workspace: The name of the GeoServer workspace.
        :type workspace: str
        :param layername: The name of the layer associated with the datastore.
        :type layername: str
        """
        data_store=self.get_store_name(workspace, layername)
        url = f"{self.geoserver_url}/rest/workspaces/{workspace}/datastores/{data_store}/featuretypes/{layername}/reset"
        try:
            response = requests.post(
                url,
                auth=(self.username, self.password),
                headers={"Accept": "application/json"} 
            )
            if response.status_code == 200:
                print(f"Successfully reset cache for: {layername}")
            else:
                print(f"Failed to reset cache for: {data_store}")
                print(f"Failed to reset. Status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")
                    
        
# test expressions 
expression={
    "strategy":"logic"
    ,"rules":[
        {"if":"Weather_Description=='NO ADVERSE CONDITIONS'","then": "safe"},
        {"if":"Weather_Description=='CLEAR'","then": "hello12"}
        ],"else":"false12"
}

# expression={
#   "strategy": "vector",
#   "expression": "Number_of_Injuries * 2"
# }
# expression={
#   "strategy": "spatial",
#   "operations": [
#     { "method": "buffer", "params": { "distance": 0.5 } },
#     { "method": "centroid" }
#   ]}        
# test run
geoserver=GeoserverCalcField()
layer_url=geoserver.get_vector_layer(workspace="field_calc",layername="hello_test1")
calculator=FieldCalculator(payload=expression,target_field="newfield",url=layer_url)
calculator.calculate(layer_name="hello_test1")
geoserver.update_datastore(workspace="field_calc",layername="hello_test1")