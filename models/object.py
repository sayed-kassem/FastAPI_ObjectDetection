from pydantic import BaseModel
from typing import List, Tuple
class Object(BaseModel):
    box : Tuple[float,float,float,float]
    label: str

class Objects(BaseModel):
    objects: List[Object]