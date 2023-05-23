import numpy as np
import xml.etree.ElementTree as ET


def get_points(path):
  if path is None or path == '':
    return None

  xml_data = open(path, 'r').read()
  xml_root = ET.fromstring(xml_data)

  dots = []
  types = []

  for item in xml_root.findall('Marker_Data/Marker_Type'):
    markerType = -1
    coord = [0,0]
    itter = 0

    for c in item:
      if c.tag == 'Type':
        # Get the type of the species
        markerType = int(c.text)
        if markerType not in types:
          types.append(markerType)
      if c.tag == 'Marker':
        # Print all subchildren of c
        for sc in c:
          if sc.tag == 'MarkerX':
            itter += 1
            coord[0] = int(sc.text)
          elif sc.tag == 'MarkerY':
            itter += 1
            coord[1] = int(sc.text)
        if markerType != -1 and itter == 2:
          dots.append({'type': markerType, 'coord': coord})
          coord = [0,0]
          itter = 0

  return dots