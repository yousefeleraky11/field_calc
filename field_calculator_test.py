import re
import geopandas as gpd
import pandas as pd
from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype, is_string_dtype,is_object_dtype
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
        geom = gdf.geometry
        for operation in payload.get('operations', []):
            method_name = operation['method']
            args = operation.get('args', [])
            kwargs = operation.get('kwargs', {})
            resolved_args = []
            for arg in args:
                if isinstance(arg, str) and arg.startswith("!") and arg.endswith("!"):
                    col_name = arg.replace("!", "")
                    if col_name not in gdf.columns:
                        raise ValueError(f"Column '{col_name}' not found.")
                    resolved_args.append(gdf[col_name])
                else:
                    resolved_args.append(arg)
            if not hasattr(geom, method_name):
                raise ValueError(f"Geometry object has no method/property '{method_name}'")
            attr = getattr(geom, method_name)
            if callable(attr):
                geom  = attr(*resolved_args, **kwargs)
            else:
                geom = attr
        gdf[target_field] = geom
        # if isinstance(result, (gpd.GeoSeries, gpd.GeoDataFrame)):
        #      gdf.set_geometry(target_field, inplace=True)
        #      gdf.drop(columns=['geometry'], inplace=True)

        return gdf
class LogicStrategy(CalcStrategy):
    """
   job: Handle logic transformations dynamically via JSON.
    """
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        rules = payload.get("rules", [])
        default_val = payload.get("else")

        if not rules:
            raise ValueError("Logic strategy requires a 'rules' list.")
        conditions = []
        choices = []
        try:
            for rule in rules:
                condition_str = rule.get("if")
                fields = re.findall(r'!(.*?)!', condition_str)
                for field in fields:
                    if field not in gdf.columns:
                        raise ValueError(f"Field {field} not found in GeoDataFrame.")
                clean_expr = condition_str.replace('!', '`')
                result_val = rule.get("then")
                mask = gdf.eval(clean_expr)
                conditions.append(mask)
                choices.append(result_val)
            gdf[target_field] = np.select(conditions, choices, default=default_val)
            return gdf
        except Exception as e:
            raise ValueError(f"Rule Engine failed: {e}")from e
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
            clean_expr = expression.replace('!', '`')
            gdf[target_field]=gdf.eval(clean_expr,engine='numexpr')
            return gdf
        except ZeroDivisionError:
            raise ValueError("Division by zero detected in expression.")from ZeroDivisionError
        except Exception as e:
            raise ValueError(
                f"VectorStrategy failed to execute expression '{expression}': {e}") from e
class StringStrategy(CalcStrategy):
    """JOB: Handle string concatenation and manipulation via JSON."""
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
       
        try:
            column_a = payload.get("col_A")
            if column_a in gdf.columns:
                column_a=gdf[column_a].astype(str)
            column_b = payload.get("col_B")
            if column_b in gdf.columns:
                column_b=gdf[column_b].astype(str)
            sep=payload.get("separator", " ")
            
            gdf[target_field] = column_a + sep + column_b
            return gdf
        except Exception as e:
            raise ValueError(f"StringStrategy failed to execute expression '{payload}': {e}") from e
class DateStrategy(CalcStrategy):
    """JOB: Handle date manipulations via JSON."""
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        # Placeholder for date manipulation logic
        try:
            date_column = payload.get("date_col")
            series = gdf[date_column]
            operation = payload.get("operation")
            if operation == "extract_year":
                gdf[target_field] = pd.to_datetime(series, errors='coerce').dt.year
            elif operation == "extract_month":
                gdf[target_field] = pd.to_datetime(series, errors='coerce').dt.month
            elif operation == "extract_day":
                gdf[target_field] = pd.to_datetime(series, errors='coerce').dt.day
            elif operation =="to_date":
                date=payload.get("date")
                time_diff= pd.to_datetime(series, format='mixed') - pd.to_datetime(date)
                gdf[target_field]= time_diff.dt.days
            return gdf    
        except Exception as e:
            raise ValueError(f"DateStrategy failed to execute expression '{payload}': {e}")from e         
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
            elif strategy_type == "string":
                return StringStrategy()
            elif strategy_type == "date":
                return DateStrategy()
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
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
            return gdf
         
        except Exception as e:
            raise ValueError(f"FieldCalculator failed to calculate field '{self.target_field}': {e}")

class ExpressionValidator:
    """
    JOB: Validate payload logic against layer schema without loading data.
    """
    def __init__(self, url, payload):
        self.url = url
        self.payload = payload
        try:
            self.gdf_schema=gpd.read_file(self.url,rows=1)
        except Exception as e:
            raise ValueError(f"Could not read layer schema from {url}: {e}")
    def _check_columns(self,columns):
        missing_columns=[col for col in columns if col not in self.gdf_schema.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    def _validate_vector(self):
        try:
            expression = self.payload.get("expression", "")
            fields = re.findall(r'!(.*?)!', expression)
            self._check_columns(fields)
            for field in fields:
                if not is_numeric_dtype(self.gdf_schema[field]):
                    raise ValueError(f"Field {field} is not numeric.")
        except Exception as e:
            raise ValueError(f"Expression validation failed: {e}")        
    def _validate_logic(self):
        try:
            rules = self.payload.get("rules", [])
            for rule in rules:
                condition_str = rule.get("if")
                fields = re.findall(r'!(.*?)!', condition_str)
                self._check_columns(fields)
        except Exception as e:
            raise ValueError(f"Logic validation failed: {e}")
    def _validate_string(self):
        try:
            col_a=self.payload.get("col_A")
            col_b=self.payload.get("col_B")
            if col_a not in self.gdf_schema.columns and col_b not in self.gdf_schema.columns:
                raise ValueError(f"Missing columns: {col_a}, {col_b}")
            if not is_object_dtype(self.gdf_schema[col_a]) and not is_object_dtype(self.gdf_schema[col_b]):   
                raise ValueError(f"Columns {col_a} and {col_b} are not string.")
        except Exception as e:
            raise ValueError(f"String validation failed: {e}") 
    def _validate_date(self):
        try:
            date_col=self.payload.get("date_col")
            self._check_columns([date_col])
            pd.to_datetime(self.gdf_schema[date_col])
            operation=self.payload.get("operation")
            if operation =="to_date":
                date=self.payload.get("date")
                pd.to_datetime(date)
        except Exception as e:
            raise ValueError(f"Date validation failed for column {date_col}: {e}")    
    def validate(self):
        try:
            strategy=self.payload.get("strategy")
            if strategy=="vector":
                self._validate_vector()
            elif strategy=="logic":
                self._validate_logic()
            elif strategy=="string":
                self._validate_string()
            elif strategy=="date":
                self._validate_date()
            print(f"the {strategy} strategy is valid for the given payload and layer schema.")    
            return {"status":"success","message":"Expression is valid."}    
        except Exception as e:
            print(f"Validation error: {e}")
            raise ValueError(f"Validation failed: {e}")            

class GeoserverCalcField:
    """
    Facade class for calculating field values using GeoServer.
    """
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
        """Retrieves the datastore name for a given layer in GeoServer.
        :param workspace: The name of the GeoServer workspace.
        :type workspace: str
        :param layer_name: The name of the layer associated with the datastore.
        :type layer_name: str
        :returns: The name of the datastore associated with the layer.
        :rtype: str
        """
        info_url = f"{self.geoserver_url}/rest/workspaces/{workspace}/featuretypes/{layer_name}.json"
        auth = (self.username, self.password)
        try:
            response = requests.get(info_url, auth=auth,timeout=60)
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
                headers={"Accept": "application/json"},
                timeout=60
            )
            if response.status_code == 200:
                print(f"Successfully reset cache for: {layername}")
            else:
                print(f"Failed to reset cache for: {data_store}")
                print(f"Failed to reset. Status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}") from e
# expression={
#     "strategy":"logic"
#     ,"rules":[
#         {"if":"!AccidentNumber!<0","then": "safe"},
        
#         ],"else":"false12"
# }
expression={
    "strategy":"date",
    "operation":"to_date",
    "date_col":"Date_and_Time",
    "date":"2018-01-01"
}
expression={
  "strategy": "vector",
  "expression": "!Accident_Number! * 0"
}
expression={
  "strategy": "spatial",
  "operations": [
    {"method": "buffer", "kwargs": { "distance": 0.5  }},
    {"method": "centroid"}]}
expression={
  "strategy": "string",
  "method": "concat",
  "col_A": "Weather_Description",
  "col_B": "1234",
  "separator": " - "
}
# #   ]}
# # test run
# geoserver=GeoserverCalcField()
# layer_url=geoserver.get_vector_layer(workspace="field_calc",layername="hello_test1")
# calculator=FieldCalculator(payload=expression,target_field="newfield1",url=layer_url)
# calculator.calculate(layer_name="hello_test1")
# geoserver.update_datastore(workspace="field_calc",layername="hello_test1")
