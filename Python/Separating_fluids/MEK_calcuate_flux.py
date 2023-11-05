from MEK_data_class import *

# Calculating flux through regions given in corners
corners = [(35,160, 70,170),(35,85,70,100),(35,50,70,60)]

data_div_curl = div_curl(u, v, dx=0.5, dy=0.5)
data_div_curl.flux_sides(corners)

# From terminal:
"""
Flux Rectangle 1: 104.8526049082102
Each sides :
 C1: 1556.867943941396 (bottom)
 C2: 21664.567474322168 (right)
 C3: -2059.677184793871 (top)
 C4: -21056.905628561482 (left)

Flux Rectangle 2: -6476.939182097958
Each sides :
 C1: -5187.564033067892 (bottom)
 C2: 14782.532896182347 (right)
 C3: -4074.0522144394345 (top)
 C4: -11997.85583077298 (left)

Flux Rectangle 3: -124.5686660449619
Each sides :
 C1: -195.57014792583357 (bottom)
 C2: 1536.8217966413547 (right)
 C3: 284.9436464350764 (top)
 C4: -1750.7639611955597 (left)
"""
